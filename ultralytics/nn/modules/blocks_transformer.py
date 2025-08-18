# /workspace/yolo/ultralytics/nn/modules/blocks_transformer.py
# 作用：在P2/P3等尺度做窗口化注意力融合（简化版Swin Transformer）
# 参考：Swin Transformer通过Shifted Windows实现局部与全局特征连接
# 动机：局部窗口自注意力提供线性复杂度与多尺度特征表达，显著提升密集预测性能
import torch
import torch.nn as nn

def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口，支持不整除的情况
    Args:
        x: (B, C, H, W) 输入特征图
        window_size: 窗口大小
    Returns:
        windows: (B*num_windows, window_size*window_size, C) 窗口化后的特征
        original_size: (H, W) 原始尺寸
    """
    B, C, H, W = x.shape
    
    # 处理不能整除的情况，使用较小的窗口大小
    actual_window_h = min(window_size, H)
    actual_window_w = min(window_size, W)
    
    # 如果特征图太小，直接全局处理
    if H <= window_size and W <= window_size:
        # 直接将整个特征图作为一个窗口
        windows = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return windows, (H, W)
    
    # 计算需要的padding，确保能被窗口大小整除
    pad_h = (actual_window_h - H % actual_window_h) % actual_window_h
    pad_w = (actual_window_w - W % actual_window_w) % actual_window_w
    
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w
    
    # 重新计算窗口数量
    num_windows_h = H // actual_window_h
    num_windows_w = W // actual_window_w
    
    # 将特征图重新组织为窗口形式
    x = x.view(B, C, H // actual_window_h, actual_window_h, W // actual_window_w, actual_window_w)
    # 调整维度顺序并展平为窗口序列
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = windows.view(-1, actual_window_h * actual_window_w, C)
    return windows, (H, W)

def window_reverse(windows, original_size, window_size):
    """
    将窗口序列重新组合为特征图，支持小特征图的情况
    Args:
        windows: (B*num_windows, window_size*window_size, C) 窗口化特征
        original_size: (H, W) 原始特征图尺寸  
        window_size: 窗口大小
    Returns:
        x: (B, C, H, W) 恢复的特征图
    """
    H, W = original_size
    
    # 处理小特征图的情况
    if H <= window_size and W <= window_size:
        B, HW, C = windows.shape
        x = windows.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x
    
    # 使用实际的窗口大小
    actual_window_h = min(window_size, H)
    actual_window_w = min(window_size, W)
    
    num_windows_h = H // actual_window_h  
    num_windows_w = W // actual_window_w
    B = int(windows.shape[0] / (num_windows_h * num_windows_w))
    
    # 重新组织窗口为特征图形式
    x = windows.view(B, num_windows_h, num_windows_w, actual_window_h, actual_window_w, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, -1, H, W)
    return x

class WindowAttention(nn.Module):
    """窗口内自注意力机制"""
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 7, mlp_ratio: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        
        # Layer Normalization 和 Multi-Head Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP层（Feed Forward Network）
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),  # 使用GELU激活函数，在Transformer中效果更好
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x):
        """
        窗口注意力前向传播
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            x: (B, C, H, W) 处理后的特征图
        """
        # 保存原始尺寸
        B, C, orig_H, orig_W = x.shape
        
        # 将特征图划分为窗口
        windows, padded_size = window_partition(x, self.window_size)
        
        # 在每个窗口内应用自注意力
        norm_windows = self.norm1(windows)
        attn_windows, _ = self.attn(norm_windows, norm_windows, norm_windows, need_weights=False)
        
        # 添加残差连接
        windows = windows + attn_windows
        
        # 应用MLP并添加残差连接
        windows = windows + self.mlp(self.norm2(windows))
        
        # 将窗口重新组合为特征图
        y = window_reverse(windows, padded_size, self.window_size)
        
        # 裁剪回原始尺寸（去除padding）
        if y.shape[-2] != orig_H or y.shape[-1] != orig_W:
            y = y[:, :, :orig_H, :orig_W]
            
        return y

class SwinBlock(nn.Module):
    """Swin Transformer块，结合深度卷积和窗口注意力"""
    def __init__(self, c: int, num_heads: int = 4, window_size: int = 7):
        super().__init__()
        # 深度可分离卷积，保留局部特征
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        
        # 窗口注意力模块
        self.window_attn = WindowAttention(c, num_heads=num_heads, window_size=window_size)
        
        # 点卷积、批归一化和激活函数
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        """
        SwinBlock前向传播
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            output: (B, C, H, W) 输出特征图
        """
        # 保存输入用于残差连接
        identity = x
        
        # 深度卷积提取局部特征
        y = self.dw(x)
        
        # 窗口注意力提取全局特征
        y = self.window_attn(y)
        
        # 点卷积、批归一化和激活
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        
        # 残差连接，提升训练稳定性和梯度流
        return identity + y
