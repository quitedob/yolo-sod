# 文件路径：/workspace/yolo/ultralytics/nn/modules/smallobj_modules_stable.py  # 小目标稳定模块集成  # 中文注释

from __future__ import annotations  # 未来注解支持  # 中文注释

import torch  # 导入PyTorch主库  # 中文注释
import torch.nn as nn  # 导入神经网络模块  # 中文注释
import torch.nn.functional as F  # 导入函数式API  # 中文注释

from .conv import Conv  # 从本包导入Conv卷积模块  # 中文注释
from .stable_fuse import ChannelNorm, ScaleAdd  # 导入稳定归一化与可伸缩加法  # 中文注释
from .detect_stable import DetectStable  # 导入稳定检测头（可按尺度屏蔽）  # 中文注释


class FusionLockTSS_Stable(nn.Module):  # 定义稳定版Fusion-Lock TSS  # 中文注释
    """MGDFIS的Fusion-Lock TSS稳定版：在多头注意力与tanh之间加入LayerNorm以稳定输出尺度"""  # 类文档  # 中文注释

    def __init__(self, channels: int):  # 初始化函数，传入通道数  # 中文注释
        super().__init__()  # 调用父类构造  # 中文注释
        self.channels = channels  # 保存通道数  # 中文注释
        self.attn = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)  # 单头自注意力，批次优先  # 中文注释
        self.norm = nn.LayerNorm(channels)  # 关键修复：LayerNorm稳定注意力输出  # 中文注释
        self.tanh = nn.Tanh()  # Tanh激活用于门控  # 中文注释

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播定义  # 中文注释
        b, c, h, w = x.shape  # 读取批次与空间尺寸  # 中文注释
        q = x.flatten(2).transpose(1, 2)  # 展平到序列(B, H*W, C)  # 中文注释
        attn_out, _ = self.attn(q, q, q)  # 自注意力输出  # 中文注释
        attn_out = self.norm(attn_out)  # 关键：归一化注意力结果  # 中文注释
        g = self.tanh(attn_out)  # Tanh门控  # 中文注释
        g = g.transpose(1, 2).view(b, c, h, w)  # 还原到(B,C,H,W)  # 中文注释
        return g * x  # 门控与残差相乘形成输出  # 中文注释


class HyperACEBlockStable(nn.Module):  # 定义稳定版HyperACEBlock  # 中文注释
    """高阶注意力与上下文增强（稳定版）：投影+ChannelNorm→卷积融合→稳定注意力→ScaleAdd残差"""  # 文档说明  # 中文注释

    def __init__(self, ch_high: int, ch_low: int, ch_out: int) -> None:  # 初始化，指定高低层通道与输出通道  # 中文注释
        super().__init__()  # 调用父类构造  # 中文注释
        self.ph = Conv(ch_high, ch_out, 1)  # 高层1x1投影到统一通道  # 中文注释
        self.pl = Conv(ch_low, ch_out, 1)  # 低层1x1投影到统一通道  # 中文注释
        self.norm_h = ChannelNorm()  # 对高层特征做通道归一化  # 中文注释
        self.norm_l = ChannelNorm()  # 对低层特征做通道归一化  # 中文注释
        self.fuse_conv = Conv(ch_out, ch_out, 3)  # 初步融合后的3x3卷积  # 中文注释
        self.attn = FusionLockTSS_Stable(ch_out)  # 稳定自注意力  # 中文注释
        self.scale_add = ScaleAdd(init_alpha=0.2)  # 可学习残差，初值较小更稳  # 中文注释

    def forward(self, x_high: torch.Tensor | list | tuple, x_low: torch.Tensor | None = None) -> torch.Tensor:  # 前向：融合两路特征  # 中文注释
        # 兼容Ultralytics的多源输入：当 from 为列表时，输入会以list/tuple形式传入  # 中文注释
        if x_low is None and isinstance(x_high, (list, tuple)):  # 若只给了一个列表  # 中文注释
            assert len(x_high) == 2, "HyperACEBlockStable 期望两个输入特征 (x_high, x_low)"  # 校验长度  # 中文注释
            x_high, x_low = x_high[0], x_high[1]  # 拆包两路  # 中文注释
        assert x_low is not None, "HyperACEBlockStable 缺少第二个输入特征"  # 安全断言  # 中文注释
        if x_high.shape[-2:] != x_low.shape[-2:]:  # 若空间尺寸不一致  # 中文注释
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode="nearest")  # 上采样对齐尺寸  # 中文注释
        h = self.norm_h(self.ph(x_high))  # 高层投影+归一化  # 中文注释
        l = self.norm_l(self.pl(x_low))  # 低层投影+归一化  # 中文注释
        fused = self.fuse_conv(h + l)  # 初步相加后卷积融合  # 中文注释
        a = self.attn(fused)  # 稳定注意力建模  # 中文注释
        y = self.scale_add(fused, a)  # 残差融合（可学习幅度）  # 中文注释
        return y  # 返回融合结果  # 中文注释


__all__ = [  # 导出符号列表  # 中文注释
    "FusionLockTSS_Stable",  # 稳定注意力模块  # 中文注释
    "HyperACEBlockStable",  # 稳定融合块  # 中文注释
    "ChannelNorm",  # 通道归一化  # 中文注释
    "ScaleAdd",  # 可伸缩加法  # 中文注释
    "DetectStable",  # 稳定检测头  # 中文注释
]


