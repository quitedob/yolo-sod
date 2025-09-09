# /workspace/yolo/ultralytics/nn/modules/smallobj_modules_stable.py
# -*- coding: utf-8 -*-
"""
包含 HyperACEBlockStable 和 DetectStable 的实现.
这些模块是为小目标检测而优化的稳定版本.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv 是官方通用卷积；ChannelNorm/ScaleAdd/FusionLockTSS_Stable 多见于分支定制模块
try:
    from ultralytics.nn.modules.conv import Conv  # 官方 Conv
except Exception:
    # 极端兜底，避免导入失败（不建议长期使用）
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            p = (k // 2) if p is None else p
            self.cv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()
        def forward(self, x):  # 简化版
            return self.act(self.bn(self.cv(x)))

# 可选：从你工程现有模块里导入；若缺失则提供轻量替代实现
try:
    from ultralytics.nn.modules.stable_ops import ChannelNorm, ScaleAdd, FusionLockTSS_Stable  # 你分支中的命名示意
except Exception:
    # ------ 轻量替代实现，确保能跑通（若你工程已有正式实现，会走上面的 try 分支） ------
    class ChannelNorm(nn.Module):
        """通道归一化：对每个通道做零均值单位方差标准化。"""
        def __init__(self, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x:(B,C,H,W)
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            return (x - mean) / (var + self.eps).sqrt()

    class ScaleAdd(nn.Module):
        """可学习残差缩放：y = x + alpha * z"""
        def __init__(self, init_alpha: float = 0.2):
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return x + self.alpha * z

    class FusionLockTSS_Stable(nn.Module):
        """稳定注意力占位：1x1 -> MHA -> 1x1，替代你工程内的稳定注意力模块。"""
        def __init__(self, c: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
            super().__init__()
            assert c % num_heads == 0, "通道数必须能整除 heads"
            self.qkv = nn.Conv2d(c, c * 3, 1, 1)
            self.mha = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True, dropout=attn_drop)
            self.proj = nn.Conv2d(c, c, 1, 1)
            self.drop = nn.Dropout(proj_drop)
            self.ln = nn.LayerNorm(c)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            qkv = self.qkv(x).view(B, 3, C, H * W).permute(1, 0, 3, 2).contiguous()  # (3,B,HW,C)
            q, k, v = qkv[0], qkv[1], qkv[2]  # (B,HW,C)
            out, _ = self.mha(q, k, v)
            out = self.ln(out + q)
            out = out.transpose(1, 2).contiguous().view(B, C, H, W)
            return self.drop(self.proj(out))

# -------------------- Detect 基类导入（多版本兼容） --------------------
DetectBase = None
try:
    from ultralytics.nn.modules.head import Detect as _Detect  # 官方常见命名
    DetectBase = _Detect
except Exception:
    try:
        from ultralytics.nn.modules.head import BaseDetect as _BaseDetect  # 某些分支命名
        DetectBase = _BaseDetect
    except Exception:
        DetectBase = None  # 若都不存在，下面会给出兜底提示


# =========================================================
# 1) 稳定版 HyperACEBlock：适配YAML v2，c1,c2,c_out 从 args 传入
# =========================================================
class HyperACEBlockStable(nn.Module):
    """高阶注意力与上下文增强（稳定版）：投影+ChannelNorm→卷积融合→稳定注意力→ScaleAdd残差"""

    def __init__(self, c_in_high, c_in_low, c_out, **kwargs):
        super().__init__()
        self.ph = Conv(c_in_high, c_out, 1)    # 高层 1x1 投影
        self.pl = Conv(c_in_low,  c_out, 1)    # 低层 1x1 投影
        self.norm_h = ChannelNorm()           # 高层通道归一化
        self.norm_l = ChannelNorm()           # 低层通道归一化
        self.fuse_conv = Conv(c_out, c_out, 3) # 3x3 融合卷积
        self.attn = FusionLockTSS_Stable(c_out) # 稳定自注意力
        self.scale_add = ScaleAdd(init_alpha=0.2) # 可学习残差

    def forward(self, x: list | tuple) -> torch.Tensor:
        # 兼容 from: [a,b] 列表输入
        assert isinstance(x, (list, tuple)) and len(x) == 2, \
            "HyperACEBlockStable 期望一个包含两个输入特征 (x_high, x_low) 的列表或元组"
        x_high, x_low = x[0], x[1]

        # 尺寸对齐
        if x_high.shape[-2:] != x_low.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode="nearest")

        # 投影 + 归一化
        h = self.norm_h(self.ph(x_high))
        l = self.norm_l(self.pl(x_low))

        # 融合卷积
        fused = self.fuse_conv(h + l)

        # 稳定注意力
        a = self.attn(fused)

        # 残差融合（可学习幅度）
        y = self.scale_add(fused, a)
        return y


# =========================================================
# 2) DetectStable：训练期按 active_mask 选择性关闭某些尺度（阻断该头梯度）
# =========================================================
class DetectStable(DetectBase if DetectBase is not None else nn.Module):
    """扩展版 Detect：通过 active_mask 选择性关闭某些尺度在训练期的损失与回传"""

    def __init__(self, nc=80, ch=(), **kwargs):
        if DetectBase is None:
            raise RuntimeError("未找到 Ultralytics Detect/BaseDetect 基类，请确认版本。")
        super().__init__(nc=nc, ch=ch, **kwargs)  # 基类完成 nl/cv2/cv3/dfl 等构建
        # 注册缓冲区：布尔掩码，长度与多尺度头数量一致，默认全开启
        self.register_buffer("active_mask", torch.ones(self.nl, dtype=torch.bool))

    def set_active_mask(self, mask: list[bool]):
        """设置激活掩码，mask 长度需与 nl 一致"""
        assert len(mask) == self.nl, "active_mask 长度需与尺度数一致"
        with torch.no_grad():
            self.active_mask[:] = torch.tensor(mask, dtype=torch.bool, device=self.active_mask.device)

    def forward(self, x):
        """
        训练期：逐尺度生成 y_i，若该尺度被关闭则 y_i = (detach()*0) 阻断梯度与损失
        推理期：复用基类 _inference 解码保持行为一致
        说明：官方 Detect 训练/推理路径的一致性参考文档/源码。cv2/cv3 分别为 box/cls 分支。
        """
        outs = []
        for i in range(self.nl):
            # 与官方 Detect 的逐尺度头保持一致：cat(box, cls)
            yi = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            if self.training and (not bool(self.active_mask[i])):
                # 训练期关闭该尺度：阻断梯度并置零，避免参与损失（对上游不反传）
                yi = yi.detach() * 0.0
            outs.append(yi)

        if self.training:
            # 训练路径：返回 per-level 列表，交由 Loss/Assigner 使用
            return outs

        # 推理路径：交给 Detect 基类的解码逻辑（拼接 + DFL/解码等）
        y = self._inference(outs)
        return y if self.export else (y, outs)


# =========================================================
# 3) LayerNorm2d for BiFormerLiteBlock
# =========================================================
class LayerNorm2d(nn.Module):
    """(B, C, H, W) channel-wise layer normalization."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# =========================================================
# 4) BiFormer-Lite (Window Attention + Global Routing Tokens)
# =========================================================
class BiFormerLiteBlock(nn.Module):
    """BiFormer-Lite Block."""
    def __init__(self, c, **kwargs):
        super().__init__()
        # Placeholder for arguments from YAML, like win, topk, num_heads
        win = kwargs.get('win', 8)
        topk = kwargs.get('topk', 64)
        num_heads = kwargs.get('num_heads', 4)
        attn_drop = kwargs.get('attn_drop', 0.0)
        proj_drop = kwargs.get('proj_drop', 0.0)

        assert c % num_heads == 0, "c must be divisible by num_heads"
        self.c = c
        self.win = win
        self.topk = topk
        self.num_heads = num_heads

        self.norm1 = LayerNorm2d(c)
        self.qkv_local = nn.Conv2d(c, c * 3, 1, 1)
        self.attn_local = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.proj_local = nn.Conv2d(c, c, 1, 1)
        self.drop_local = nn.Dropout(proj_drop)

        self.norm2 = LayerNorm2d(c)
        self.q_proj_g = nn.Conv2d(c, c, 1, 1)
        self.kv_proj_g = nn.Conv2d(c, c * 2, 1, 1)
        self.attn_global = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.proj_global = nn.Conv2d(c, c, 1, 1)
        self.drop_global = nn.Dropout(proj_drop)

        self.norm3 = LayerNorm2d(c)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, 1),
            nn.GELU(),
            nn.Conv2d(c * 2, c, 1, 1),
        )

    @staticmethod
    def _window_partition(x: torch.Tensor, win: int):
        B, C, H, W = x.shape
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hn, Wn = x.shape[2], x.shape[3]
        x = x.view(B, C, Hn // win, win, Wn // win, win).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, win, win)
        return x, (Hn, Wn)

    @staticmethod
    def _window_merge(x_win: torch.Tensor, B: int, grid_hw: tuple[int, int], win: int):
        Hn, Wn = grid_hw
        C = x_win.shape[1]
        x = x_win.view(B, Hn // win, Wn // win, C, win, win).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, Hn, Wn)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x1 = self.norm1(x)
        qkv = self.qkv_local(x1)
        q, k, v = qkv.chunk(3, dim=1)

        q_win, grid_hw = self._window_partition(q, self.win)
        k_win, _ = self._window_partition(k, self.win)
        v_win, _ = self._window_partition(v, self.win)
        
        q_flat = q_win.flatten(2).transpose(1, 2)
        k_flat = k_win.flatten(2).transpose(1, 2)
        v_flat = v_win.flatten(2).transpose(1, 2)

        out_local, _ = self.attn_local(q_flat, k_flat, v_flat)
        out_local = out_local.transpose(1, 2).view_as(q_win)

        out_local = self._window_merge(out_local, B, grid_hw, self.win)
        if grid_hw[0] != H or grid_hw[1] != W:
            out_local = out_local[:, :, :H, :W]

        x = x + self.drop_local(self.proj_local(out_local))

        x2 = self.norm2(x)
        score = self.q_proj_g(x2)
        score_map = score.pow(2).sum(1)
        k_g, v_g = self.kv_proj_g(x2).chunk(2, dim=1)

        topk = min(self.topk, H * W)
        _, idx = torch.topk(score_map.view(B, -1), k=topk, dim=1)
        
        k_g_flat = k_g.view(B, C, -1).transpose(1, 2)
        v_g_flat = v_g.view(B, C, -1).transpose(1, 2)

        idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)
        K_global = torch.gather(k_g_flat, 1, idx_exp)
        V_global = torch.gather(v_g_flat, 1, idx_exp)

        q_glb = self.q_proj_g(x2).view(B, C, -1).transpose(1, 2)
        out_global, _ = self.attn_global(q_glb, K_global, V_global)
        out_global = out_global.transpose(1, 2).view(B, C, H, W)
        x = x + self.drop_global(self.proj_global(out_global))

        x = x + self.ffn(self.norm3(x))
        return x


