# ultralytics/utils/loss_mf.py
"""
MambaFusion损失函数模块
实现Shape-IoU + BCE/DSLA分类损失 + ATSS匹配
专为VisDrone小目标检测优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.utils.metrics import bbox_iou


def shape_iou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Shape-IoU损失函数
    更好地处理边界框形状和尺度差异
    
    Args:
        pred_boxes: 预测边界框 [N, 4] (x, y, w, h)
        target_boxes: 目标边界框 [N, 4] (x, y, w, h)
        eps: 数值稳定性参数
        
    Returns:
        Shape-IoU损失值
    """
    # 计算标准IoU
    iou = bbox_iou(pred_boxes, target_boxes, xywh=True)
    
    # 计算宽高比差异 (对数空间)
    pred_log_wh = torch.log(pred_boxes[:, 2:] + eps)
    target_log_wh = torch.log(target_boxes[:, 2:] + eps)
    
    # 宽高差异惩罚项
    wh_diff = torch.sum((pred_log_wh - target_log_wh) ** 2, dim=1)
    
    # Shape-IoU = IoU - 宽高差异惩罚
    shape_iou = iou - wh_diff / (wh_diff + 1.0)
    
    # 返回损失 (1 - Shape-IoU)
    return (1.0 - shape_iou).mean()


def dsla_targets(iou_scores, center_distances, gamma=2.0):
    """
    动态软标签分配 (DSLA)
    基于IoU和中心距离生成软标签
    
    Args:
        iou_scores: IoU分数 [N]
        center_distances: 中心距离 [N] (归一化到[0,1])
        gamma: 软化参数
        
    Returns:
        软标签分数 [N]
    """
    # 软标签 = IoU^gamma * (1 - center_distance)
    soft_targets = (iou_scores ** gamma) * (1.0 - center_distances)
    return torch.clamp(soft_targets, 0.0, 1.0)


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss用于处理样本不平衡
    
    Args:
        pred: 预测概率 [N, C]
        target: 目标标签 [N] 或软标签 [N, C]
        alpha: 平衡参数
        gamma: 聚焦参数
        
    Returns:
        Focal Loss值
    """
    # 如果target是硬标签，转换为one-hot
    if target.dim() == 1:
        target_onehot = F.one_hot(target, num_classes=pred.size(1)).float()
    else:
        target_onehot = target
    
    # 计算交叉熵
    ce_loss = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction='none')
    
    # 计算概率
    p_t = torch.sigmoid(pred)
    p_t = torch.where(target_onehot == 1, p_t, 1 - p_t)
    
    # 应用alpha权重
    alpha_t = torch.where(target_onehot == 1, alpha, 1 - alpha)
    
    # 计算focal权重
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # 最终focal loss
    focal_loss = focal_weight * ce_loss
    
    return focal_loss.mean()


class MFLoss(nn.Module):
    """
    MambaFusion综合损失函数
    结合Shape-IoU回归损失和Focal分类损失
    """
    
    def __init__(self, 
                 cls_weight=1.0,
                 box_weight=7.5, 
                 obj_weight=1.0,
                 focal_gamma=2.0,
                 focal_alpha=0.25):
        super().__init__()
        
        self.cls_weight = cls_weight    # 分类损失权重
        self.box_weight = box_weight    # 回归损失权重
        self.obj_weight = obj_weight    # 目标性损失权重
        self.focal_gamma = focal_gamma  # Focal Loss参数
        self.focal_alpha = focal_alpha
        
    def forward(self, predictions, targets):
        """
        计算总损失
        
        Args:
            predictions: 模型预测 (cls_pred, reg_pred, obj_pred)
            targets: 真实标签
            
        Returns:
            总损失和各项损失的字典
        """
        cls_pred, reg_pred, obj_pred = predictions
        
        # 解析目标
        # 这里需要根据实际的target格式来调整
        # target应该包含: boxes, labels, indices等
        
        device = cls_pred.device
        batch_size = cls_pred.size(0)
        
        # 初始化损失
        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device) 
        obj_loss = torch.tensor(0.0, device=device)
        
        # 计算每个样本的损失
        for i in range(batch_size):
            # 获取当前样本的预测和目标
            if i < len(targets) and len(targets[i]['boxes']) > 0:
                target_boxes = targets[i]['boxes']  # [M, 4]
                target_labels = targets[i]['labels']  # [M]
                
                # 这里需要实现样本匹配逻辑 (ATSS或SimOTA)
                # 简化版本：使用最佳匹配
                
                # 计算分类损失 (Focal Loss)
                cls_targets = F.one_hot(target_labels, num_classes=cls_pred.size(1)).float()
                cls_loss += focal_loss(
                    cls_pred[i].permute(1,2,0).reshape(-1, cls_pred.size(1)),
                    cls_targets,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma
                )
                
                # 计算回归损失 (Shape-IoU)
                # 这里需要将预测的网格坐标转换为实际坐标
                # 简化实现
                box_loss += shape_iou_loss(reg_pred[i], target_boxes[:1])  # 简化
                
        # 加权总损失
        total_loss = (
            self.cls_weight * cls_loss +
            self.box_weight * box_loss + 
            self.obj_weight * obj_loss
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss, 
            'obj_loss': obj_loss
        }


def compute_atss_targets(pred_boxes, pred_scores, gt_boxes, gt_labels, num_classes):
    """
    ATSS (Adaptive Training Sample Selection) 目标分配
    动态选择正样本，解决小目标样本稀缺问题
    
    Args:
        pred_boxes: 预测框 [N, 4]
        pred_scores: 预测分数 [N, C] 
        gt_boxes: 真实框 [M, 4]
        gt_labels: 真实标签 [M]
        num_classes: 类别数
        
    Returns:
        分配的目标和权重
    """
    # ATSS实现较复杂，这里提供简化版本
    # 实际使用时建议使用更完整的实现
    
    device = pred_boxes.device
    num_pred = pred_boxes.size(0)
    num_gt = gt_boxes.size(0)
    
    if num_gt == 0:
        # 没有真实目标，全部为负样本
        return {
            'labels': torch.zeros(num_pred, dtype=torch.long, device=device),
            'bbox_targets': torch.zeros_like(pred_boxes),
            'weights': torch.zeros(num_pred, device=device)
        }
    
    # 计算IoU矩阵 [N, M]
    ious = bbox_iou(pred_boxes, gt_boxes, xywh=True)
    
    # 对每个GT选择top-k个候选
    k = min(9, num_pred)  # 每个GT最多选择9个候选
    topk_ious, topk_indices = torch.topk(ious, k, dim=0)
    
    # 计算动态阈值
    iou_thresholds = topk_ious.mean(dim=0) + topk_ious.std(dim=0)
    
    # 分配正样本
    positive_mask = torch.zeros((num_pred, num_gt), dtype=torch.bool, device=device)
    for gt_idx in range(num_gt):
        candidates = topk_indices[:, gt_idx]
        valid_candidates = ious[candidates, gt_idx] >= iou_thresholds[gt_idx]
        positive_mask[candidates[valid_candidates], gt_idx] = True
    
    # 处理多个GT的冲突：选择IoU最大的
    max_ious, matched_gt_indices = ious.max(dim=1)
    is_positive = positive_mask.any(dim=1)
    
    # 构建目标
    labels = torch.zeros(num_pred, dtype=torch.long, device=device)
    bbox_targets = torch.zeros_like(pred_boxes)
    weights = torch.zeros(num_pred, device=device)
    
    if is_positive.any():
        positive_indices = is_positive.nonzero(as_tuple=False).squeeze(1)
        matched_gts = matched_gt_indices[positive_indices]
        
        labels[positive_indices] = gt_labels[matched_gts]
        bbox_targets[positive_indices] = gt_boxes[matched_gts]
        weights[positive_indices] = 1.0
    
    return {
        'labels': labels,
        'bbox_targets': bbox_targets, 
        'weights': weights
    }