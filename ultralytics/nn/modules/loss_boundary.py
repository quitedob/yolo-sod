# /workspace/yolo/ultralytics/nn/modules/loss_boundary.py
# 作用：对预测/GT做Sobel边缘图再BCE，增强轮廓可分性；可选加IoU混合
# 依据：BASNet的边界/混合损失可显著改善SOD的边界质量，减少锯齿和粘连现象
# 动机：在显著性目标检测中，边界质量对视觉效果至关重要
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareLoss(nn.Module):
    """边界感知损失，结合Sobel边缘检测和二元交叉熵损失"""
    def __init__(self, edge_weight: float = 1.0, bce_weight: float = 1.0, iou_weight: float = 0.0):
        super().__init__()
        self.edge_weight = edge_weight  # 边缘损失权重
        self.bce_weight = bce_weight    # BCE损失权重
        self.iou_weight = iou_weight    # IoU损失权重
        
        # 定义Sobel算子卷积核用于边缘检测
        # X方向Sobel核
        sobel_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2], 
            [1, 0, -1]
        ], dtype=torch.float32)
        
        # Y方向Sobel核  
        sobel_y = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32)
        
        # 注册为缓冲区，不参与梯度计算但会跟随模型设备迁移
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def _compute_edge_map(self, mask):
        """
        使用Sobel算子计算边缘图
        Args:
            mask: (B, 1, H, W) 输入掩码
        Returns:
            edge_map: (B, 1, H, W) 归一化的边缘图
        """
        # 应用Sobel算子计算X和Y方向梯度
        grad_x = F.conv2d(mask, self.sobel_x, padding=1)
        grad_y = F.conv2d(mask, self.sobel_y, padding=1)
        
        # 计算梯度幅值（边缘强度）
        gradient_magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        
        # 归一化到[0,1]范围，避免数值不稳定
        batch_size = gradient_magnitude.size(0)
        normalized_edges = []
        
        for i in range(batch_size):
            single_edge = gradient_magnitude[i]
            min_val = single_edge.min()
            max_val = single_edge.max()
            
            # 避免除零错误
            if (max_val - min_val) < 1e-6:
                normalized_edge = torch.zeros_like(single_edge)
            else:
                normalized_edge = (single_edge - min_val) / (max_val - min_val)
            
            normalized_edges.append(normalized_edge)
        
        return torch.stack(normalized_edges, dim=0)
    
    def _compute_iou_loss(self, pred_mask, gt_mask):
        """
        计算IoU损失
        Args:
            pred_mask: (B, 1, H, W) 预测掩码
            gt_mask: (B, 1, H, W) 真实掩码
        Returns:
            iou_loss: IoU损失值
        """
        # 计算交集和并集
        intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
        union = (pred_mask + gt_mask - pred_mask * gt_mask).sum(dim=(1, 2, 3)) + 1e-6
        
        # 计算IoU并转为损失（1 - IoU）
        iou = intersection / union
        iou_loss = 1 - iou.mean()
        
        return iou_loss
    
    def forward(self, pred_mask, gt_mask):
        """
        计算边界感知损失
        Args:
            pred_mask: (B, 1, H, W) 预测掩码，值在[0,1]范围
            gt_mask: (B, 1, H, W) 真实掩码，值在[0,1]范围  
        Returns:
            total_loss: 总损失值
        """
        # 确保输入张量的维度正确
        if pred_mask.dim() == 3:
            pred_mask = pred_mask.unsqueeze(1)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        
        # 如果通道数不是1，取第一个通道（假设为前景通道）
        if pred_mask.size(1) != 1:
            pred_mask = pred_mask[:, :1]
        if gt_mask.size(1) != 1:
            gt_mask = gt_mask[:, :1]
        
        total_loss = 0.0
        
        # 1. 基础二元交叉熵损失
        if self.bce_weight > 0:
            bce_loss = F.binary_cross_entropy(pred_mask, gt_mask, reduction='mean')
            total_loss += self.bce_weight * bce_loss
        
        # 2. IoU损失（可选）
        if self.iou_weight > 0:
            iou_loss = self._compute_iou_loss(pred_mask, gt_mask)
            total_loss += self.iou_weight * iou_loss
        
        # 3. 边界感知损失
        if self.edge_weight > 0:
            # 计算预测掩码和真实掩码的边缘图
            pred_edges = self._compute_edge_map(pred_mask)
            gt_edges = self._compute_edge_map(gt_mask)
            
            # 在边缘图上计算二元交叉熵损失
            edge_loss = F.binary_cross_entropy(pred_edges, gt_edges, reduction='mean')
            total_loss += self.edge_weight * edge_loss
        
        return total_loss
    
    def get_edge_maps(self, pred_mask, gt_mask):
        """
        获取边缘图用于可视化
        Args:
            pred_mask: (B, 1, H, W) 预测掩码
            gt_mask: (B, 1, H, W) 真实掩码
        Returns:
            pred_edges: 预测掩码的边缘图
            gt_edges: 真实掩码的边缘图
        """
        with torch.no_grad():
            pred_edges = self._compute_edge_map(pred_mask)
            gt_edges = self._compute_edge_map(gt_mask)
            return pred_edges, gt_edges
