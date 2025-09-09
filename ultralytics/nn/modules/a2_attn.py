# Area Attention (A2_Attn) Block Implementation
# Reference: "YOLOv12: Attention-Centric Real-Time Object Detectors" concept

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

class A2_Attn(nn.Module):
    """
    Area Attention (A2_Attn) Block
    参考论文: "YOLOv12: Attention-Centric Real-Time Object Detectors" (arXiv:2502.12524)
    通过将特征图划分为区域并对区域进行注意力计算，以降低计算复杂度。
    """
    def __init__(self, c1, c2=None, num_areas=4, num_heads=4):
        super().__init__()
        if c2 is None:
            c2 = c1
            
        self.num_areas = num_areas
        self.num_heads = num_heads
        
        # 确保通道数可以被头数整除
        assert c1 % num_heads == 0, f"Input channels {c1} must be divisible by num_heads {num_heads}"
        
        self.proj = Conv(c1, c1, 1) # 1x1卷积进行特征投影
        
        # 使用标准多头注意力
        self.attention = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        
        self.out_proj = Conv(c1, c2, 1) # 输出投影
        self.layer_norm = nn.LayerNorm(c1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        
        # 1. 特征投影
        x_proj = self.proj(x)
        
        # 2. 区域划分与池化 (此处以垂直划分为例)
        # 将特征图在高度上划分为 num_areas 个区域
        # 使用AdaptiveAvgPool2d将每个区域池化为一个向量
        pooled_x = F.adaptive_avg_pool2d(x_proj, (self.num_areas, w)) # (B, C, num_areas, W)
        
        # 3. 准备注意力输入
        # (B, C, num_areas, W) -> (B, C, num_areas * W) -> (B, num_areas * W, C)
        seq = pooled_x.flatten(2).transpose(1, 2)
        
        # 4. 多头自注意力
        # LayerNorm有助于稳定训练
        seq_norm = self.layer_norm(seq)
        attn_output, _ = self.attention(seq_norm, seq_norm, seq_norm)
        
        # 5. 上采样并还原形状
        # (B, num_areas * W, C) -> (B, C, num_areas * W) -> (B, C, num_areas, W)
        attn_output = attn_output.transpose(1, 2).reshape(b, c, self.num_areas, w)
        
        # 使用插值上采样回原始高度
        upsampled_attn = F.interpolate(attn_output, size=(h, w), mode='bilinear', align_corners=False)
        
        # 6. 输出投影与残差连接
        output = self.out_proj(upsampled_attn)
        
        # 如果输出通道数与输入不同，则不使用残差连接
        if output.shape[1] == identity.shape[1]:
            return output + identity
        else:
            return output