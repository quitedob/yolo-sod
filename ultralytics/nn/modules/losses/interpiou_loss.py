# /workspace/yolo/ultralytics/nn/modules/losses/interpiou_loss.py
# 主要功能简介: InterpIoU损失函数完整实现 - 专为小目标检测优化的几何损失函数
# 基于2025年最新研究：通过插值构建平滑损失曲面，解决小目标定位精度瓶颈

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def _iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    稳定IoU计算（xyxy坐标），返回逐样本IoU值
    
    Args:
        a: 预测框 [N,4] 或 [K,N,4]
        b: 真实框 [N,4] 或 [K,N,4]
        eps: 数值稳定性参数
        
    Returns:
        IoU值 [N] 或 [K,N]
    """
    # 计算交集左上与右下
    tl = torch.maximum(a[..., :2], b[..., :2])  # 左上角取较大
    br = torch.minimum(a[..., 2:], b[..., 2:])  # 右下角取较小
    inter_wh = (br - tl).clamp(min=0)  # 交集宽高非负
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # 交集面积
    
    # 计算各自面积
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = (area_a + area_b - inter_area).clamp(min=eps)  # 并集面积并加eps防零
    
    return inter_area / union  # 返回IoU

def interpiou_iou_xyxy(
    pred_xyxy: torch.Tensor, 
    gt_xyxy: torch.Tensor, 
    samples: int = 8, 
    eps: float = 1e-9
) -> torch.Tensor:
    """
    插值IoU计算：在预测框与GT框间做线性插值，采样K点计算IoU与GT的均值
    
    设计理念：通过插值构建平滑损失曲面，解决小目标训练不稳定的问题
    
    Args:
        pred_xyxy: [N,4] 预测框（xyxy格式）
        gt_xyxy: [N,4] 真实框（xyxy格式）
        samples: 采样次数K，越大越稳但越慢
        eps: 数值稳定项
        
    Returns:
        iou_interp: [N] 的均值IoU（非损失）
    """
    assert pred_xyxy.shape == gt_xyxy.shape, "pred/gt形状需一致"
    device = pred_xyxy.device
    
    # 生成线性插值系数t∈[0,1]，形状[K,1,1]
    t = torch.linspace(0.0, 1.0, steps=max(int(samples), 1), device=device).view(-1, 1, 1)
    
    # 线性插值盒：B_t = (1-t)*pred + t*gt，得到[K,N,4]
    interp = (1.0 - t) * pred_xyxy.unsqueeze(0) + t * gt_xyxy.unsqueeze(0)
    
    # 与GT的IoU，逐K求IoU后在K维求均值
    ious = _iou_xyxy(interp, gt_xyxy.unsqueeze(0).expand_as(interp), eps=eps)  # [K,N]
    mean_iou = ious.mean(dim=0)  # [N]
    
    return mean_iou

def interpiou_loss_xyxy(
    pred_xyxy: torch.Tensor, 
    gt_xyxy: torch.Tensor, 
    samples: int = 8, 
    eps: float = 1e-9
) -> torch.Tensor:
    """
    插值IoU损失：基于插值IoU的回归损失
    
    Args:
        pred_xyxy: [N,4] 预测框
        gt_xyxy: [N,4] 真实框
        samples: 采样次数
        eps: 数值稳定项
        
    Returns:
        标量损失值
    """
    iou = interpiou_iou_xyxy(pred_xyxy, gt_xyxy, samples=samples, eps=eps)
    return (1.0 - iou).mean()

class InterpIoULoss(nn.Module):
    """
    InterpIoU损失函数模块
    
    核心优势：
    1. 平滑损失曲面：通过插值避免梯度消失
    2. 小目标友好：对微小偏差提供有意义的梯度
    3. 数值稳定：避免极端情况下的训练不稳定
    """
    
    def __init__(self, samples: int = 8, eps: float = 1e-9, reduction: str = "mean"):
        """
        初始化InterpIoU损失
        
        Args:
            samples: 插值采样点数
            eps: 数值稳定性参数
            reduction: 损失聚合方式 ("mean", "sum", "none")
        """
        super().__init__()
        self.samples = samples
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        
        Args:
            pred_boxes: 预测边界框 [N,4] (xyxy格式)
            target_boxes: 目标边界框 [N,4] (xyxy格式)
            
        Returns:
            损失值
        """
        # 计算插值IoU损失
        loss = interpiou_loss_xyxy(pred_boxes, target_boxes, self.samples, self.eps)
        
        # 应用聚合方式
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "mean"
            return loss.mean()

class InterpIoUBboxLoss(nn.Module):
    """
    集成到YOLO训练流程的InterpIoU边界框损失
    
    兼容Ultralytics的BboxLoss接口，可直接替换原有损失函数
    """
    
    def __init__(self, reg_max: int = 16, samples: int = 8):
        """
        初始化InterpIoU边界框损失
        
        Args:
            reg_max: DFL最大回归值
            samples: InterpIoU采样点数
        """
        super().__init__()
        self.reg_max = reg_max
        self.samples = samples
        
        # DFL损失（如果reg_max > 1）
        self.dfl_loss = None
        if reg_max > 1:
            from ultralytics.nn.modules.block import DFL
            self.dfl_loss = DFL(reg_max)
    
    def forward(self, 
                pred_dist: torch.Tensor, 
                pred_bboxes: torch.Tensor, 
                anchor_points: torch.Tensor, 
                target_bboxes: torch.Tensor, 
                target_scores: torch.Tensor, 
                target_scores_sum: torch.Tensor, 
                fg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算InterpIoU损失和DFL损失
        
        Args:
            pred_dist: 预测分布 [B,N,reg_max*4]
            pred_bboxes: 预测边界框 [B,N,4]
            anchor_points: 锚点坐标 [B,N,2]
            target_bboxes: 目标边界框 [B,N,4]
            target_scores: 目标分数 [B,N]
            target_scores_sum: 目标分数总和
            fg_mask: 前景掩码 [B,N]
            
        Returns:
            (loss_iou, loss_dfl): IoU损失和DFL损失
        """
        # 计算权重
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # InterpIoU损失
        iou = interpiou_iou_xyxy(
            pred_bboxes[fg_mask], 
            target_bboxes[fg_mask], 
            samples=self.samples
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        # DFL损失
        if self.dfl_loss:
            from ultralytics.utils.loss import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.reg_max), 
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        
        return loss_iou, loss_dfl

# 导出主要接口
__all__ = [
    "interpiou_iou_xyxy",
    "interpiou_loss_xyxy", 
    "InterpIoULoss",
    "InterpIoUBboxLoss"
]
