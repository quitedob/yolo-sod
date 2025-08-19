# /workspace/yolo/ultralytics/utils/tal_rfla.py
# -*- coding: utf-8 -*-
"""
RFLA_TaskAlignedAssigner: 基于高斯感受野 + KLD 的标签分配器（含两阶段 HLA）
与 Ultralytics TaskAlignedAssigner 接口一致，可直接替换使用。

论文依据（ECCV'22 RFLA）：
- 高斯 ERF/GT 建模与 TRF->ERF 近似（式(2)(3)(4)）
- KLD 闭式解并归一化为 RFD（式(6)(7)(8)）
- 两阶段 HLA（式(9)）
参考：RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection. :contentReference[oaicite:1]{index=1}
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 与 Ultralytics 基类保持一致的返回签名
    from ultralytics.utils.tal import TaskAlignedAssigner
except Exception:  # pragma: no cover
    # 若环境中类名不同，可把此处替换为你所在分支的 TAL 基类
    class TaskAlignedAssigner(nn.Module):
        def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
            super().__init__()
            self.topk, self.num_classes = topk, num_classes
            self.alpha, self.beta, self.eps = alpha, beta, eps


class RFLA_TaskAlignedAssigner(TaskAlignedAssigner):
    """
    针对微小目标（P2/P3）友好的标签分配器：
    - 用 RFD(=1/(1+KLD)) 替代 IoU 定位项
    - 两阶段 HLA：top-k + 衰减 er_n 的补样
    - 与 TaskAlignedAssigner forward 的输入/输出完全一致

    重要超参：
    - rf_ratio:   用来把“层 stride”近似到 ERF 半径 er_n ≈ rf_ratio * stride
                  （论文式(2)给出 TRF 半径，ERF≈TRF/2；这里以 rf_ratio 作为可调近似）
    - beta_hla:   HLA 第 2 阶段的 ERF 衰减因子（论文表 3 推荐 0.9）
    - strides:    各层步长（如 [4, 8, 16, 32]，包含 P2 时以 4 起）
    - hw_list:    各层特征图尺寸 [(H2,W2),(H3,W3),... ]，用于把 anc_points 拆回各层
    """
    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
        *,
        strides: List[int],
        hw_list: List[Tuple[int, int]],
        rf_ratio: float = 2.0,
        beta_hla: float = 0.9,
    ):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        # === 关键配置（中文注释） ===
        self.strides = list(strides)              # 每个层级的 stride
        self.hw_list = list(hw_list)              # 每个层级的 (H,W)
        self.rf_ratio = float(rf_ratio)           # er_n ~ rf_ratio * stride 的比例
        self.beta_hla = float(beta_hla)           # HLA 二阶段的 ERF 衰减因子
        # 预计算每层 anchor 数，以及每个 anchor 的 ERF 半径（像素，随层不同）
        self._level_num_anchors = [h * w for h, w in self.hw_list]
        self.register_buffer(
            "_ern_per_anchor",
            self._build_erf_radius_per_anchor(self.strides, self._level_num_anchors, self.rf_ratio),
            persistent=False,
        )

    # =========================
    # 核心：前向（接口与 TA 一致）
    # =========================
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores: (B, A, C)  分类分数（未sigmoid或已sigmoid都可，这里用原分数配合幂次）
            pd_bboxes: (B, A, 4)  预测框（xywh 或 xyxy 不影响本分配器，因为我们用 anc_points）
            anc_points:(A, 2)     anchor 网格中心（已被乘上 stride，处于输入分辨率坐标系）
            gt_labels: (B, N, 1)  GT 类别 id
            gt_bboxes: (B, N, 4)  GT 框 (xyxy)，注意：这里我们只用中心与宽高
            mask_gt:   (B, N, 1)  有效 gt 掩码
        Returns: 与 TaskAlignedAssigner 保持一致的 5 个张量：
            target_labels (B, A), target_bboxes (B, A, 4), target_scores (B, A, C),
            fg_mask (B, A), target_gt_idx (B, A)
        """
        device = gt_bboxes.device
        bs, A, C = pd_scores.shape
        N = gt_bboxes.shape[1]

        if N == 0:
            # 与 Ultralytics 空返回保持一致
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),  # target_labels
                torch.zeros_like(pd_bboxes),                           # target_bboxes
                torch.zeros_like(pd_scores),                           # target_scores
                torch.zeros_like(pd_scores[..., 0]),                   # fg_mask
                torch.zeros_like(pd_scores[..., 0]),                   # target_gt_idx
            )

        # ---------- 1) 计算基于 KLD 的 RFD（第一阶段） ----------
        # ERF 半径（像素）：每个 anchor 一个值，形状 (A,)
        ern = self._ern_per_anchor.to(device)  # (A,)
        # GT 中心与宽高
        xg = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5  # (B,N)
        yg = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5  # (B,N)
        wg = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_(min=self.eps)
        hg = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_(min=self.eps)

        # anchor 中心
        xa = anc_points[:, 0]  # (A,)
        ya = anc_points[:, 1]  # (A,)

        # 维度对齐以一次性广播：B,N,A
        xa = xa.view(1, 1, A)
        ya = ya.view(1, 1, A)
        ern2 = (ern ** 2).view(1, 1, A)

        xg = xg.unsqueeze(-1)
        yg = yg.unsqueeze(-1)
        wg = wg.unsqueeze(-1)
        hg = hg.unsqueeze(-1)

        # —— KLD 化简式（论文式(7)）——
        # D_KL = er^2/(8 w^2) + er^2/(8 h^2) + 2*(xa-xg)^2/w^2 + 2*(ya-yg)^2/h^2 + ln(2w/er) + ln(2h/er) - 1
        term_var = ern2 / (8.0 * (wg ** 2)) + ern2 / (8.0 * (hg ** 2))
        term_pos = 2.0 * ((xa - xg) ** 2) / (wg ** 2) + 2.0 * ((ya - yg) ** 2) / (hg ** 2)
        # 为数值稳定，所有 log() 前都 clamp
        term_log = torch.log((2.0 * wg).clamp_min(self.eps)) - torch.log(ern.view(1, 1, A).clamp_min(self.eps)) \
                 + torch.log((2.0 * hg).clamp_min(self.eps)) - torch.log(ern.view(1, 1, A).clamp_min(self.eps))
        d_kl_stage1 = term_var + term_pos + term_log - 1.0
        rfd_stage1 = 1.0 / (1.0 + d_kl_stage1.clamp_min(0.0))   # 归一化到 (0,1)

        # ---------- 2) 分类-定位对齐度（按论文思想，把分类与 RFD 结合） ----------
        # 提取对应 GT 类别的分类分数，并做 α/β 幂（与 TAL 一致的“task-aligned”范式）
        # pd_scores: (B,A,C) -> (B,N,A) 只取每个 gt 的类通道
        gt_cls = gt_labels.long().squeeze(-1).clamp_(min=0)      # (B,N)
        # gather 前把维度凑齐：index 形状 (B,N,1) 与 (B,A,C) 对齐
        cls_gather = gt_cls.unsqueeze(-1).expand(-1, -1, A)      # (B,N,A) 作为每个 gt 的类 id
        # 把 (B,A,C) 先换到 (B,A,1) 再广播到 (B,N,A) 不直观；更直接做 one-hot 再点乘
        one_hot = F.one_hot(gt_cls, num_classes=C).float()       # (B,N,C)
        # (B,A,C) × (B,N,C) -> (B,N,A)（通过爱因斯坦求和更省显存）
        cls_scores = torch.einsum("bac,bnc->bna", pd_scores.sigmoid(), one_hot)

        # 仅保留有效 gt
        valid = mask_gt.bool().squeeze(-1)                       # (B,N)
        rfd_stage1 = rfd_stage1 * valid.unsqueeze(-1)            # (B,N,A)
        cls_scores = cls_scores * valid.unsqueeze(-1)

        # 对齐度（类似 TAL 的 score^alpha * iou^beta，这里把 iou 换成 rfd）
        align_metric_s1 = (cls_scores.clamp_min(self.eps) ** self.alpha) * (rfd_stage1.clamp_min(self.eps) ** self.beta)

        # ---------- 3) HLA - 第一阶段：对每个 gt 取 top-k ----------
        mask_topk_s1 = self._select_topk(align_metric_s1)        # (B,N,A) 0/1

        # ---------- 4) HLA - 第二阶段：衰减 ERF 半径，补 1 个正样本 ----------
        ern_decay2 = (ern * self.beta_hla).view(1, 1, A)
        d_kl_stage2 = (ern_decay2 ** 2) / (8.0 * (wg ** 2)) + (ern_decay2 ** 2) / (8.0 * (hg ** 2)) \
                    + 2.0 * ((xa - xg) ** 2) / (wg ** 2) + 2.0 * ((ya - yg) ** 2) / (hg ** 2) \
                    + torch.log((2.0 * wg).clamp_min(self.eps)) - torch.log(ern_decay2.clamp_min(self.eps)) \
                    + torch.log((2.0 * hg).clamp_min(self.eps)) - torch.log(ern_decay2.clamp_min(self.eps)) - 1.0
        rfd_stage2 = 1.0 / (1.0 + d_kl_stage2.clamp_min(0.0))
        align_metric_s2 = (cls_scores.clamp_min(self.eps) ** self.alpha) * (rfd_stage2.clamp_min(self.eps) ** self.beta)

        # 只对“第一阶段无正样本”的 GT 进行补样，且每个 GT 仅补 1 个
        has_pos = mask_topk_s1.bool().any(dim=-1, keepdim=True)  # (B,N,1)
        # 在未被 S1 选中的位置上找 S2 的 argmax
        metric_s2_masked = align_metric_s2.masked_fill(mask_topk_s1.bool(), -1e9)
        idx_top1_s2 = metric_s2_masked.argmax(dim=-1)            # (B,N)
        mask_top1_s2 = torch.zeros_like(mask_topk_s1, dtype=torch.bool)  # (B,N,A)
        mask_top1_s2.scatter_(-1, idx_top1_s2.unsqueeze(-1), True)
        # 只有在 S1 没有正样本的 gt 上才启用 S2
        mask_s2_effective = (~has_pos) & valid.unsqueeze(-1)
        mask_top1_s2 = mask_top1_s2 & mask_s2_effective

        # ---------- 5) 合并 HLA 两阶段的结果（论文式(9)思想） ----------
        final_mask = mask_topk_s1.bool() | mask_top1_s2.bool()   # (B,N,A)
        # 最终用于解决冲突的打分：把无效位置置为 -inf，方便后续 argmax 选 winner
        metric_final = torch.where(final_mask, align_metric_s1, torch.full_like(align_metric_s1, -1e9))
        # 对于来自 S2 的补样，把相应位置的分数替换为 S2 的分数，避免 S1 的 -inf 干扰
        metric_final = torch.where(mask_top1_s2, align_metric_s2, metric_final)

        # ---------- 6) 解决“同一 anchor 被多个 gt 选中”的冲突：按分数取最大 ----------
        # (B,N,A) -> (B,A): 每个 anchor 只保留分数最高的那个 gt
        max_metric, matched_gt = metric_final.max(dim=1)         # (B,A), (B,A)
        fg_mask = (max_metric > 0)                                # (B,A)

        # ---------- 7) 生成与 Ultralytics 对齐的返回张量 ----------
        # target_gt_idx: 无前景的位置置零；有前景的位置是 “该 anchor 对应的 gt 索引”
        target_gt_idx = matched_gt * fg_mask.long()

        # 收集目标标签与框
        target_labels = torch.full((bs, A), self.num_classes, dtype=torch.long, device=device)
        target_bboxes = torch.zeros((bs, A, 4), dtype=gt_bboxes.dtype, device=device)
        target_scores = torch.zeros((bs, A, C), dtype=pd_scores.dtype, device=device)

        # 便于索引的批次下标
        b_idx = torch.arange(bs, device=device).unsqueeze(-1)     # (B,1)
        a_idx = torch.nonzero(fg_mask, as_tuple=False)            # (M,2) -> [b,a]

        if a_idx.numel() > 0:
            b_pick = a_idx[:, 0]
            a_pick = a_idx[:, 1]
            g_pick = matched_gt[b_pick, a_pick]                   # (M,)
            # 标签
            tgt_cls_pick = gt_labels[b_pick, g_pick, 0].long().clamp_(min=0)
            target_labels[b_pick, a_pick] = tgt_cls_pick
            # 框（xyxy 直接拷贝）
            target_bboxes[b_pick, a_pick] = gt_bboxes[b_pick, g_pick]
            # 分数：one-hot × 对齐度（与 TAL 的做法一致）
            score_val = max_metric[b_pick, a_pick].clamp_min(self.eps)  # (M,)
            target_scores[b_pick, a_pick, :] = 0.0
            target_scores[b_pick, a_pick, tgt_cls_pick] = score_val

        return target_labels, target_bboxes, target_scores, fg_mask.float(), target_gt_idx

    # =========================
    # 辅助：top-k 选择（等价于 TAL 的 select_topk_candidates）
    # =========================
    def _select_topk(self, metrics: torch.Tensor) -> torch.Tensor:
        """
        metrics: (B,N,A) 每个 gt 对每个 anchor 的对齐度
        return : (B,N,A) 0/1 的 top-k mask
        """
        topk_val, topk_idx = torch.topk(metrics, k=self.topk, dim=-1, largest=True)
        # 避免全 0：若一个 gt 最高分也很低，仍保留其 top-k 的位置
        mask = torch.zeros_like(metrics, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        return mask

    # =========================
    # 辅助：构建每个 anchor 的 ERF 半径（像素）
    # =========================
    @staticmethod
    def _build_erf_radius_per_anchor(strides: List[int], n_per_level: List[int], rf_ratio: float) -> torch.Tensor:
        """
        把每层的 stride 近似映射为 ERF 半径：er_n ≈ rf_ratio * stride
        然后按层重复到每个 anchor。

        更精确的计算可按式(2)累计 TRF，再除以 2 得 ERF 半径；这里提供等效的可调近似，
        便于在 YOLOv12 的 FPN（含 P2）中快速落地。
        """
        er_list = []
        for s, n in zip(strides, n_per_level):
            er = float(rf_ratio) * float(s)
            er_list.append(torch.full((n,), er, dtype=torch.float32))
        return torch.cat(er_list, dim=0)  # (A,)
