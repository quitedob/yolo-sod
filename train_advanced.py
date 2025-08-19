# /workspace/yolo/train_advanced.py
# -*- coding: utf-8 -*-
"""
YOLO-SOD-Advanced 训练脚本
功能:
1. 注册所有新的自定义模块 (DCNv3, BiFormer, RFLA, NWD等)。
2. 定义并使用 AdvancedDetectionTrainer，该Trainer集成了RFLA分配器和NWD损失。
3. 从命令行接收新的模型配置和超参数文件。
"""
import os
import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils import yaml_load, checks
from ultralytics.nn.modules import Detect, C2f
import math

# ========== 1. 自定义模块实现（现在从 advanced_blocks.py 导入） ==========
from ultralytics.nn.modules.advanced_blocks import C2f_DCNv3, BiFormerLiteBlock as BiFormerBlock

# --- 1.3 NWD Loss ---
class NWD_BboxLoss(nn.Module):
    def __init__(self, C=1.0, reduction='mean'):
        super().__init__()
        self.C = C
        self.reduction = reduction

    def forward(self, pred_bboxes, target_bboxes, eps=1e-7):
        # cx, cy, w, h
        mu_pred, var_pred = pred_bboxes[..., :2], (pred_bboxes[..., 2:]**2) / 4.0
        mu_target, var_target = target_bboxes[..., :2], (target_bboxes[..., 2:]**2) / 4.0

        dist_mu = torch.sum((mu_pred - mu_target)**2, dim=-1)
        dist_var = torch.sum((torch.sqrt(var_pred) - torch.sqrt(var_target))**2, dim=-1)
        w2_squared = dist_mu + dist_var
        
        loss = 1.0 - torch.exp(-torch.sqrt(w2_squared.clamp(min=eps)) / self.C)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# --- 1.4 RFLA Assigner (概念性实现) ---
class RFLA_TaskAlignedAssigner(TaskAlignedAssigner):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__(topk, num_classes, alpha, beta, eps)
        print("[INFO] Using RFLA_TaskAlignedAssigner (conceptual).")

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        RFLA的核心思想是用基于感受野的度量替代IoU。
        这里我们用一个中心点距离的指数衰减函数来模拟RFD。
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.zeros(0, device=device), torch.zeros(0, 4, device=device), torch.zeros(0, self.num_classes, device=device), torch.zeros(0, device=device, dtype=torch.bool), torch.zeros(0, device=device))

        # 1. 计算中心点距离作为RFD的代理
        gt_xy = gt_bboxes[..., :2]
        pd_xy = pd_bboxes[..., :2]
        
        # [b, n_max_boxes, num_anchors]
        pairwise_dist = torch.cdist(gt_xy, pd_xy, p=2)

        # 归一化距离并转换为相似度 (RFD-like)
        # 这里的归一化因子应该是与感受野/尺度相关的, 此处简化
        rfd_sim = torch.exp(-pairwise_dist / 20.0)
        
        # 2. 结合分类得分
        # `get_matching_matrix` is not a standard method in TaskAlignedAssigner.
        # Let's use a simplified logic.
        b, max_num_obj, _ = gt_labels.shape
        n_anchors = pd_scores.shape[1]
        
        # [b, n_anchors, n_cls] -> [b, n_anchors, 1] -> [b, 1, n_anchors]
        pd_scores_max, _ = pd_scores.max(dim=-1)
        pd_scores_max = pd_scores_max.unsqueeze(1).repeat(1, max_num_obj, 1)

        # 3. 计算对齐度量 (alignment_metric)
        alignment_metric = pd_scores_max.pow(self.alpha) * rfd_sim.pow(self.beta)
        
        # 4. 选择top-k候选
        # `select_topk_candidates` is also not standard. Re-implementing logic.
        topk_ious, _ = torch.topk(alignment_metric, self.topk, dim=-1)
        dynamic_ks = torch.clamp(topk_ious.sum(2).int(), min=1)
        
        mask_topk = torch.zeros_like(alignment_metric, dtype=torch.bool)
        for i in range(b):
            for j in range(max_num_obj):
                if mask_gt[i, j, 0]:
                    _, topk_idx = torch.topk(alignment_metric[i, j, :], dynamic_ks[i, j], dim=-1)
                    mask_topk[i, j, topk_idx] = True

        mask_topk = mask_topk.logical_and(mask_gt)

        # 5. 获取最终分配结果
        fg_mask, gt_idx = mask_topk.max(1)
        
        target_gt_idx = gt_idx[fg_mask]
        
        batch_idx = torch.arange(b, device=gt_bboxes.device).unsqueeze(1).repeat(1, n_anchors)[fg_mask]
        
        target_bboxes = gt_bboxes[batch_idx, target_gt_idx]
        target_labels = gt_labels[batch_idx, target_gt_idx]
        
        target_scores = torch.zeros((b, n_anchors, self.num_classes), device=gt_bboxes.device)
        target_scores[batch_idx, fg_mask, target_labels.long().squeeze()] = 1.0
        
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


# ========== 2. 自定义训练器和损失 ==========
class v8AdvancedDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = RFLA_TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.nwd_loss = NWD_BboxLoss(C=12.8, reduction='none') # 根据论文建议值
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=True).to(self.device)
        print("[INFO] v8AdvancedDetectionLoss initialized with RFLA and NWD+CIoU hybrid loss.")

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        bs = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = self.preprocess(batch['cls'].squeeze(-1), batch['bboxes'], bs)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        
        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # CIoU Loss and DFL
            loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes,
                                                target_scores, target_scores_sum, fg_mask)
            loss[0] = loss_iou
            loss[2] = loss_dfl

            # NWD Loss
            pred_bboxes_xywh = self.bbox_decode(anchor_points, pred_distri, xywh=True)
            target_bboxes_xywh = self.xyxy2xywh(target_bboxes)
            loss_nwd = self.nwd_loss(pred_bboxes_xywh[fg_mask], target_bboxes_xywh[fg_mask])
            loss_nwd = (loss_nwd * target_scores.sum(-1)[fg_mask]).sum() / target_scores_sum
            
            loss[0] += loss_nwd * self.hyp.get('nwd', 5.0) # 添加NWD损失
        
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * bs, loss.detach()

    def xyxy2xywh(self, x):
        y = torch.zeros_like(x)
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]
        return y


class AdvancedDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, ch=3, nc=self.data['nc'], verbose=verbose and RANK in (-1, 0))
        if weights:
            model.load(weights)
        return model
        
    def get_criterion(self):
        return v8AdvancedDetectionLoss(self.model)


# ========== 3. 模块注册与主函数 ==========
def register_custom_modules():
    """注册所有自定义模块到Ultralytics命名空间"""
    import ultralytics.nn.modules as U
    try:
        from ultralytics.nn.modules.detect_stable import DetectStable
        U.DetectStable = DetectStable
        from ultralytics.nn.modules.stable_fuse import HyperACEBlockStable
        U.HyperACEBlockStable = HyperACEBlockStable
    except ImportError:
        print(" [INFO] Could not import baseline custom modules. Assuming they are not needed.")

    U.C2f_DCNv3 = C2f_DCNv3
    # Note: The YAML uses BiFormerBlock, so we map BiFormerLiteBlock to this name.
    U.BiFormerBlock = BiFormerBlock 
    U.DetectStable = Detect # 临时用标准Detect
    U.HyperACEBlockStable = C2f # 临时用C2f
    print("[INFO] Successfully registered advanced custom modules.")

def main():
    parser = argparse.ArgumentParser(description="YOLO-SOD-Advanced Training Script")
    parser.add_argument('--cfg', required=True, help='Model structure YAML path')
    parser.add_argument('--hyp', required=True, help='Hyperparameters YAML path')
    parser.add_argument('--data', required=True, help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default='runs_advanced')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--weights', default='', help='Path to pretrained weights')
    args = parser.parse_args()

    register_custom_modules()

    overrides = yaml_load(checks.check_yaml(args.hyp))
    overrides.update(vars(args))
    
    trainer = AdvancedDetectionTrainer(overrides=overrides)
    trainer.train()

if __name__ == "__main__":
    RANK = int(os.environ.get('RANK', -1))
    main()
