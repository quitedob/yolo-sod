# /workspace/yolo/ultralytics/nn/modules/advanced_blocks.py
# -*- coding: utf-8 -*-
"""
中文简介：
- 提供 DCNv3/4 的统一包装（优先 DCNv4 -> mmcv -> 普通Conv 回退）
- 提供 C2f_DCNv3：正确替换 Bottleneck 的 3x3 卷积（cv2）
- 提供 BiFormerLiteBlock：窗口注意 + 少量全局路由 token 的轻量版 BRA
参考：
- DCNv3/4 & InternImage 自定义算子需求（需编译/安装）: https://github.com/OpenGVLab/InternImage
- timm 的 FlashInternImage PR（提到 PyTorch 版替代 DCNv3/4）：https://github.com/huggingface/pytorch-image-models/pull/2167
- BiFormer 论文/官方实现：BRA 双层路由注意（区域级筛选+token级注意）
"""

from typing import Optional, Tuple
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 0. 尝试导入 DCNv4 或 mmcv.ops，失败则回退 ======
_HAS_DCNV4 = False
_HAS_MMCV = False
try:
    # pip install DCNv4   （若可用，速度/精度最佳）
    # 备注：不同包的符号名可能不同，实际以安装包暴露的API为准
    import DCNv4  # noqa: F401
    from DCNv4 import dcnv4 as _dcnv4  # 假定暴露 dcnv4 函数（不同版本可能有差异）
    _HAS_DCNV4 = True
except Exception:
    try:
        from mmcv.ops import modulated_deform_conv2d as _mdcn
        _HAS_MMCV = True
    except Exception:
        _HAS_DCNV4 = False
        _HAS_MMCV = False


# ====== 1. DCN 卷积包装：统一接口 ======
class DCNConv2d(nn.Module):
    """
    中文说明：
    - 统一包装 DCNv4 / DCNv3(modulated deform conv) / 普通 Conv2d
    - 若 DCN 后端可用：使用 offset+mask 的 modulated deformable conv
    - 若不可用：回退到标准 Conv2d，并打印一次警告
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: bool = True):
        super().__init__()
        assert kernel_size in (3, 5, 7), "建议使用 3/5/7 kernel 以匹配常见 DCN 参数"
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        self.stride = stride
        self.pad = kernel_size // 2 if padding is None else padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # DCN 核心权重（与普通 conv2d 权重一致）
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size) * 0.02)

        # offset & mask 预测头（DCN 专用）
        # offset 通道数 = 2*k*k*deform_groups；mask 通道数 = k*k*deform_groups
        off_ch = 2 * kernel_size * kernel_size * deform_groups
        msk_ch = kernel_size * kernel_size * deform_groups
        self.conv_offset = nn.Conv2d(in_channels, off_ch, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv_mask = nn.Conv2d(in_channels, msk_ch, kernel_size=3, stride=stride, padding=1, bias=True)

        # 回退标识避免重复 warn
        self._warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_DCNV4:
            # 伪代码：不同 DCNv4 包的 API 略有差异，需按实际安装的函数签名调用
            # 这里保留接口占位（项目中按实际包修正）
            offset = self.conv_offset(x)
            mask = torch.sigmoid(self.conv_mask(x))
            # 假定 _dcnv4.forward(input, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups)
            return _dcnv4(x, offset, mask, self.weight, self.bias, self.stride, self.pad, self.dilation, self.groups, self.deform_groups)

        if _HAS_MMCV:
            # mmcv 的 modulated_deform_conv2d
            offset = self.conv_offset(x)
            mask = torch.sigmoid(self.conv_mask(x))
            return _mdcn(x, offset, mask, self.weight, self.bias, stride=self.stride,
                         padding=self.pad, dilation=self.dilation, groups=self.groups,
                         deform_groups=self.deform_groups)

        # 回退：普通卷积（功能正确但无 DCN 变形能力）
        if not self._warned and self.training:
            warnings.warn("DCN 后端不可用，已回退为普通 Conv2d（性能会下降）。建议安装 DCNv4 或 mmcv.ops。", stacklevel=1)
            self._warned = True
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.pad,
                        dilation=self.dilation, groups=self.groups)


# ====== 2. C2f_DCNv3：正确替换 Bottleneck 的 3x3（cv2），非 cv1 ======
class C2f_DCNv3(nn.Module):
    """
    中文说明：
    - 基于 Ultralytics 的 C2f 搭建：将每个 Bottleneck 的 3x3 卷积（cv2）替换成 DCNConv2d
    - 替换 cv2（3x3 主卷积）而不是 cv1（1x1），否则通道/结构不匹配
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3, deform_groups=1):
        super().__init__()
        from ultralytics.nn.modules import Conv, Bottleneck  # 复用官方实现
        self.c = int(c2 * e)

        # 入口 1x1：与原 C2f 一致
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # 出口 1x1：与原 C2f 一致
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        # n 个残差瓶颈，每个把 3x3 主卷积替换为 DCNConv2d
        self.m = nn.ModuleList()
        for _ in range(n):
            b = Bottleneck(self.c, self.c, shortcut, g, k=3, e=1.0)  # 先按默认构造
            # 用 DCN 替换 b.cv2（3x3）
            b.cv2 = DCNConv2d(self.c, self.c, kernel_size=k, stride=1, padding=k // 2, groups=g, deform_groups=deform_groups, bias=True)
            self.m.append(b)

    def forward(self, x):
        # 与 Ultralytics C2f 结构一致
        y = list(self.cv1(x).chunk(2, 1))
        for b in self.m:
            y.append(b(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ====== 3. 工具：LayerNorm2d（适配 (B,C,H,W)）=====
class LayerNorm2d(nn.Module):
    """中文：对 (B, C, H, W) 的通道归一化，便于替代 LayerNorm。"""
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


# ====== 4. BiFormer-Lite（窗口注意 + 全局路由 token）=====
class BiFormerLiteBlock(nn.Module):
    """
    中文说明（BRA-Lite近似）：
    - 第一步：窗口内自注意力（局部，降低复杂度，类似 Swin 的 window-MSA 思想）
    - 第二步：选取全局 top-k“路由 token”（通过内容分数），对窗口输出做一次跨窗口稀疏注意
    - 仅使用 PyTorch 的 MultiheadAttention + 稠密 matmul，满足 GPU 友好实现
    - 超参：
        win：窗口大小（如 8）
        topk：全局路由 token 数（如 64）
    - 该实现保留“路由注意”的核心思想，但未完全复刻官方 BRA 的区域级-令牌级两段筛选细节
      （官方实现与论文请参考： https://github.com/rayleizhu/BiFormer ）。
    """
    def __init__(self, c: int, win: int = 8, topk: int = 64, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert c % num_heads == 0, "通道数必须能被 num_heads 整除"
        self.c = c
        self.win = win
        self.topk = topk
        self.num_heads = num_heads

        # 局部注意：用 MHA 实现窗口内注意
        self.norm1 = LayerNorm2d(c)
        self.qkv_local = nn.Conv2d(c, c * 3, 1, 1)
        self.attn_local = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.proj_local = nn.Conv2d(c, c, 1, 1)
        self.drop_local = nn.Dropout(proj_drop)

        # 全局路由注意：从全图选 top-k token 作为 KV
        self.norm2 = LayerNorm2d(c)
        self.q_proj_g = nn.Conv2d(c, c, 1, 1)
        self.kv_proj_g = nn.Conv2d(c, c * 2, 1, 1)
        self.attn_global = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True, dropout=attn_drop)
        self.proj_global = nn.Conv2d(c, c, 1, 1)
        self.drop_global = nn.Dropout(proj_drop)

        # FFN
        self.norm3 = LayerNorm2d(c)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, 1),
            nn.GELU(),
            nn.Conv2d(c * 2, c, 1, 1),
        )

    @staticmethod
    def _window_partition(x: torch.Tensor, win: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hn, Wn = x.shape[2], x.shape[3]
        x = x.unfold(2, win, win).unfold(3, win, win)  # B, C, Hn/win, Wn/win, win, win
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()   # B, Nh, Nw, C, win, win
        x = x.view(B, -1, C, win, win)                 # B, num_win, C, win, win
        return x, (Hn, Wn)

    @staticmethod
    def _window_merge(x_win: torch.Tensor, grid_hw: Tuple[int, int], win: int) -> torch.Tensor:
        B, num_win, C, _, _ = x_win.shape
        Hn, Wn = grid_hw
        Nh, Nw = Hn // win, Wn // win
        x = x_win.view(B, Nh, Nw, C, win, win).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, C, Nh, win, Nw, win
        x = x.view(B, C, Hn, Wn)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ---------- 1) 局部窗口注意 ----------
        x1 = self.norm1(x)
        qkv = self.qkv_local(x1)
        q, k, v = qkv.chunk(3, dim=1)  # (B,C,H,W)

        # 切窗口 -> (B*win_num, win*win, C)
        def _prep(t):
            tw, grid = self._window_partition(t, self.win)
            B0, Nw, C0, wh, ww = tw.shape
            flat = tw.view(B0 * Nw, C0, wh * ww).transpose(1, 2).contiguous()
            return flat, (B0, Nw, C0, wh, ww), grid

        qf, meta, grid_hw = _prep(q)
        kf, _, _ = _prep(k)
        vf, _, _ = _prep(v)

        out_local, _ = self.attn_local(qf, kf, vf)  # (B*Nw, wh*ww, C)
        # 还原窗口 -> (B, C, H, W)
        B0, Nw, C0, wh, ww = meta
        out_local = out_local.transpose(1, 2).contiguous().view(B0, Nw, C0, wh, ww)
        out_local = self._window_merge(out_local, grid_hw, self.win)
        out_local = self.drop_local(self.proj_local(out_local))
        x = x + out_local  # 残差

        # ---------- 2) 全局路由注意（选 top-k token 作为全局 KV） ----------
        x2 = self.norm2(x)
        # 路由打分：简单用 1x1 投影后对空间求范数
        score = self.q_proj_g(x2)  # (B,C,H,W)
        score_map = score.pow(2).sum(1)  # (B,H,W)
        k_g, v_g = self.kv_proj_g(x2).chunk(2, dim=1)  # (B,C,H,W)

        # 选出 top-k 全局 token 的坐标
        topk = min(self.topk, H * W)
        vals, idx = torch.topk(score_map.view(B, -1), k=topk, dim=1, largest=True, sorted=False)  # (B,topk)
        # gather 全局 KV
        def _gather_tokens(feat):  # (B,C,H,W) -> (B, topk, C)
            feat_flat = feat.view(B, C, -1).permute(0, 2, 1).contiguous()
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)
            return torch.gather(feat_flat, 1, idx_exp)  # (B, topk, C)

        K_global = _gather_tokens(k_g)
        V_global = _gather_tokens(v_g)

        # 全局注意：Q 用全图，K/V 用 top-k
        q_glb = self.q_proj_g(x2).view(B, C, -1).transpose(1, 2).contiguous()  # (B, H*W, C)
        out_global, _ = self.attn_global(q_glb, K_global, V_global)            # (B, H*W, C)
        out_global = out_global.transpose(1, 2).contiguous().view(B, C, H, W)
        out_global = self.drop_global(self.proj_global(out_global))
        x = x + out_global  # 残差

        # ---------- 3) FFN ----------
        x = x + self.ffn(self.norm3(x))
        return x
