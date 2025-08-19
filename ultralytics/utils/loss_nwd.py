# 功能: 实现Normalized Wasserstein Distance (NWD)损失函数
# 理论依据: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection" (arXiv:2110.13389)

import torch

def _box_to_gaussian(boxes: torch.Tensor, eps: float = 1e-9):
    """
    将边界框 (cx, cy, w, h) 转换为二维高斯分布 N(μ, Σ).
    Args:
        boxes (torch.Tensor): 形状为 (*, 4) 的边界框张量.
        eps (float): 防止除零的小常数.
    Returns:
        tuple: 均值μ (shape=(*, 2)) 和协方差Σ (shape=(*, 2, 2)).
    """
    # 中心点 (cx, cy) 作为均值 μ
    mean = boxes[..., :2]
    
    # 宽高 (w, h) 用于计算协方差矩阵 Σ
    # Σ = diag(w^2/4, h^2/4)
    hw = boxes[..., 2:].clamp(min=eps)
    var = (hw ** 2) / 4
    
    # 创建对角协方差矩阵
    B = boxes.shape[:-1]
    cov = torch.zeros(*B, 2, 2, device=boxes.device, dtype=boxes.dtype)
    cov[..., 0, 0] = var[..., 0]
    cov[..., 1, 1] = var[..., 1]
    
    return mean, cov

def _wasserstein_distance_diag(mean1, cov1, mean2, cov2, eps: float = 1e-9):
    """
    计算两个对角协方差矩阵的二维高斯分布之间的2阶瓦瑟斯坦距离的平方.
    W_2^2(N_1, N_2) = ||μ_1 - μ_2||_2^2 + ||Σ_1^{1/2} - Σ_2^{1/2}||_F^2
    Args:
        mean1, mean2 (torch.Tensor): 形状为 (*, 2) 的均值向量.
        cov1, cov2 (torch.Tensor): 形状为 (*, 2, 2) 的对角协方差矩阵.
        eps (float): 防止计算不稳定的常数.
    Returns:
        torch.Tensor: 瓦瑟斯坦距离的平方，形状为 (*,).
    """
    # ||μ_1 - μ_2||_2^2
    mean_dist_sq = ((mean1 - mean2) ** 2).sum(dim=-1)
    
    # ||Σ_1^{1/2} - Σ_2^{1/2}||_F^2
    # 对于对角矩阵, Σ^{1/2} = diag(sqrt(σ_11), sqrt(σ_22))
    cov1_sqrt_diag = torch.sqrt(torch.diagonal(cov1, dim1=-2, dim2=-1))
    cov2_sqrt_diag = torch.sqrt(torch.diagonal(cov2, dim1=-2, dim2=-1))
    cov_dist_sq = ((cov1_sqrt_diag - cov2_sqrt_diag) ** 2).sum(dim=-1)
    
    return mean_dist_sq + cov_dist_sq

def nwd_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-7, constant: float = 12.8) -> torch.Tensor:
    """
    计算NWD损失. L_NWD = 1 - exp(-sqrt(W_2^2) / C).
    Args:
        pred_boxes (torch.Tensor): 预测框 (cx, cy, w, h), 形状 (N, 4).
        gt_boxes (torch.Tensor): 真实框 (cx, cy, w, h), 形状 (N, 4).
        eps (float): 数值稳定性常数.
        constant (float): NWD归一化常数 C.
    Returns:
        torch.Tensor: NWD损失，形状 (N,).
    """
    # 1. 将边界框转换为高斯分布
    mean_pred, cov_pred = _box_to_gaussian(pred_boxes, eps)
    mean_gt, cov_gt = _box_to_gaussian(gt_boxes, eps)
    
    # 2. 计算瓦瑟斯坦距离
    wasserstein_sq = _wasserstein_distance_diag(mean_pred, cov_pred, mean_gt, cov_gt, eps)
    
    # 3. 计算NWD相似度
    # 添加eps防止sqrt(0)的梯度问题
    nwd_similarity = torch.exp(-torch.sqrt(wasserstein_sq.clamp(min=eps)) / constant)
    
    # 4. 计算NWD损失
    loss = 1 - nwd_similarity
    
    return loss
