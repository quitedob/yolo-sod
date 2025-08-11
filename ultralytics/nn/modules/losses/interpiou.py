# /workspace/yolo/ultralytics/nn/modules/losses/interpiou.py  # 文件路径
# 功能简介：提供 InterpIoU 近似（在预测框与GT框间做线性插值，取 IoU 均值作为更平滑的几何项），更稳健于小目标与偏置情况  # 中文注释

import torch  # 引入PyTorch主库  # 中文注释


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """稳定IoU计算（xyxy坐标），返回逐样本IoU值。a,b形状[N,4]或[K,N,4]对齐。  # 中文注释"""
    # 计算交集左上与右下  # 中文注释
    tl = torch.maximum(a[..., :2], b[..., :2])  # 左上角取较大  # 中文注释
    br = torch.minimum(a[..., 2:], b[..., 2:])  # 右下角取较小  # 中文注释
    inter_wh = (br - tl).clamp(min=0)  # 交集宽高非负  # 中文注释
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # 交集面积  # 中文注释
    # 计算各自面积  # 中文注释
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)  # 中文注释
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)  # 中文注释
    union = (area_a + area_b - inter_area).clamp(min=eps)  # 并集面积并加eps防零  # 中文注释
    return inter_area / union  # 返回IoU  # 中文注释


def interpiou_iou_xyxy(
    pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, samples: int = 8, eps: float = 1e-9
) -> torch.Tensor:
    """返回 InterpIoU 的IoU近似（而非损失），形状与样本N对齐。  # 中文注释

    做法：在xyxy空间对预测框与GT框做线性插值，采样K点计算IoU与GT的均值，作为稳定IoU估计。  # 中文注释

    参数：  # 中文注释
    - pred_xyxy: [N,4] 预测框（xyxy）  # 中文注释
    - gt_xyxy:   [N,4] GT框（xyxy）    # 中文注释
    - samples:   采样次数K，越大越稳但越慢  # 中文注释
    - eps:       数值稳定项  # 中文注释
    返回：  # 中文注释
    - iou_interp: [N] 的均值IoU（非损失）  # 中文注释
    """
    assert pred_xyxy.shape == gt_xyxy.shape, "pred/gt形状需一致"  # 中文注释
    device = pred_xyxy.device  # 读取设备  # 中文注释
    # 生成线性插值系数t∈[0,1]，形状[K,1,1]  # 中文注释
    t = torch.linspace(0.0, 1.0, steps=max(int(samples), 1), device=device).view(-1, 1, 1)  # 中文注释
    # 线性插值盒：B_t = (1-t)*pred + t*gt，得到[K,N,4]  # 中文注释
    interp = (1.0 - t) * pred_xyxy.unsqueeze(0) + t * gt_xyxy.unsqueeze(0)  # 中文注释
    # 与GT的IoU，逐K求IoU后在K维求均值  # 中文注释
    ious = _iou_xyxy(interp, gt_xyxy.unsqueeze(0).expand_as(interp), eps=eps)  # [K,N]  # 中文注释
    mean_iou = ious.mean(dim=0)  # [N]  # 中文注释
    return mean_iou  # 返回均值IoU  # 中文注释


def interpiou_loss_xyxy(
    pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, samples: int = 8, eps: float = 1e-9
) -> torch.Tensor:
    """返回 InterpIoU 损失标量：1 - mean(InterpIoU)。便于独立调用做回归项。  # 中文注释"""
    iou = interpiou_iou_xyxy(pred_xyxy, gt_xyxy, samples=samples, eps=eps)  # 计算插值IoU  # 中文注释
    return (1.0 - iou).mean()  # 平均为标量损失  # 中文注释


