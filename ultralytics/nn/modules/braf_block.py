# /workspace/yolo/ultralytics/nn/modules/braf_block.py
# 主要功能简介: 实现双层路由注意力融合模块(BRAF), 在高分辨率特征图上以低复杂度建模长程依赖, 提升小目标检测表现

import math  # 数学工具  # 中文注释
from typing import Tuple  # 类型注解  # 中文注释

import torch  # 导入PyTorch  # 中文注释
import torch.nn as nn  # 神经网络模块  # 中文注释
import torch.nn.functional as F  # 函数式API  # 中文注释

from .conv import Conv  # 引入常规卷积封装  # 中文注释
from .block import C2f  # 引入C2f结构用于轻量特征提炼  # 中文注释


class BiLevelRoutingAttention(nn.Module):
    """
    双层路由注意力(Bi-Level Routing Attention, BRA)核心实现
    设计要点:
      1) 区域级路由: 先将特征划分为不重叠网格窗口, 以区域均值作为区域token, 计算区域间相关性并为每个查询区域选出top-k相关区域
      2) 令牌级注意力: 仅在被路由到的少量目标区域内, 对查询区域内的每个像素token执行细粒度注意力, 显著降低计算量
    复杂度优势:
      - 标准MHSA复杂度~O((HW)^2); BRA复杂度~O(HW * (k * T)), 其中T为窗口内token数, k远小于总区域数
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, topk: int = 4):  # 初始化传入通道/头/窗口/TopK  # 中文注释
        super().__init__()  # 调用父类构造  # 中文注释
        assert dim % num_heads == 0, "dim必须能被num_heads整除"  # 保证每头维度整除  # 中文注释
        self.dim = dim  # 保存总通道  # 中文注释
        self.num_heads = num_heads  # 注意力头数  # 中文注释
        self.head_dim = dim // num_heads  # 每头通道数  # 中文注释
        self.scale = self.head_dim ** -0.5  # 缩放因子用于稳定点积  # 中文注释
        self.window_size = int(window_size)  # 区域窗口尺寸(正方形)  # 中文注释
        self.topk = int(max(1, topk))  # 每个区域选择的top-k目标区域数  # 中文注释

        self.qkv = Conv(dim, dim * 3, 1, act=False)  # 1x1卷积生成Q/K/V拼接通道  # 中文注释
        self.proj = Conv(dim, dim, 1, act=False)  # 输出投影保持通道一致  # 中文注释

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:  # 内部:补齐到窗口整除  # 中文注释
        b, c, h, w = x.shape  # 读取形状  # 中文注释
        ws = self.window_size  # 快捷引用窗口尺寸  # 中文注释
        ph = (ws - h % ws) % ws  # 高度补齐量  # 中文注释
        pw = (ws - w % ws) % ws  # 宽度补齐量  # 中文注释
        if ph or pw:  # 若需要补齐  # 中文注释
            x = F.pad(x, (0, pw, 0, ph))  # 右/下零填充  # 中文注释
        return x, (ph, pw, h, w)  # 返回补齐张量与原始信息  # 中文注释

    def _window_partition(self, x: torch.Tensor, ws: int) -> torch.Tensor:  # 将特征划分为窗口并展平  # 中文注释
        b, c, h, w = x.shape  # 形状  # 中文注释
        gh, gw = h // ws, w // ws  # 网格行列数  # 中文注释
        x = x.view(b, c, gh, ws, gw, ws)  # 拆分为网格与窗口内像素  # 中文注释
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (b, gh, gw, ws, ws, c)  # 中文注释
        x = x.view(b, gh * gw, ws * ws, c)  # (b, R, T, c) 其中R=区域数,T=窗口token数  # 中文注释
        return x  # 返回区域-令牌展平表示  # 中文注释

    def _window_reverse(self, x: torch.Tensor, ws: int, gh: int, gw: int) -> torch.Tensor:  # 将窗口还原为特征图  # 中文注释
        b, r, t, c = x.shape  # 读取窗口序列形状  # 中文注释
        x = x.view(b, gh, gw, ws, ws, c)  # 重整为网格-窗口-通道  # 中文注释
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (b, c, gh, ws, gw, ws)  # 中文注释
        x = x.view(b, c, gh * ws, gw * ws)  # 合并回H/W  # 中文注释
        return x  # 返回重建特征图  # 中文注释

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播  # 中文注释
        b, c, h0, w0 = x.shape  # 读取输入形状  # 中文注释
        x_pad, (ph, pw, h, w) = self._pad_to_window(x)  # 补齐到窗口整除  # 中文注释
        ws = self.window_size  # 窗口大小快捷变量  # 中文注释
        gh, gw = (h + ph) // ws, (w + pw) // ws  # 计算补齐后的网格数  # 中文注释

        qkv = self.qkv(x_pad)  # 生成QKV拼接特征  # 中文注释
        q, k, v = torch.chunk(qkv, 3, dim=1)  # 沿通道切分为Q/K/V  # 中文注释

        # 重排为(B, Heads, D, H, W)以便窗口化处理  # 中文注释
        q = q.view(b, self.num_heads, self.head_dim, (h + ph), (w + pw))  # 形状重排  # 中文注释
        k = k.view(b, self.num_heads, self.head_dim, (h + ph), (w + pw))  # 形状重排  # 中文注释
        v = v.view(b, self.num_heads, self.head_dim, (h + ph), (w + pw))  # 形状重排  # 中文注释

        # 变换为(B*H, C_h, H, W)以便统一窗口划分  # 中文注释
        bh = b * self.num_heads  # 合并批次与头  # 中文注释
        q = q.permute(0, 1, 3, 4, 2).contiguous().view(bh, (h + ph), (w + pw), self.head_dim).permute(0, 3, 1, 2)
        k = k.permute(0, 1, 3, 4, 2).contiguous().view(bh, (h + ph), (w + pw), self.head_dim).permute(0, 3, 1, 2)
        v = v.permute(0, 1, 3, 4, 2).contiguous().view(bh, (h + ph), (w + pw), self.head_dim).permute(0, 3, 1, 2)

        # 窗口划分: 得到(BH, R, T, D)  # 中文注释
        q_w = self._window_partition(q, ws)  # 查询窗口token  # 中文注释
        k_w = self._window_partition(k, ws)  # 键窗口token  # 中文注释
        v_w = self._window_partition(v, ws)  # 值窗口token  # 中文注释

        bh_, r, t, d = q_w.shape  # 读取窗口化后的形状  # 中文注释

        # 区域级token: 按窗口内token均值进行池化 (BH, R, D)  # 中文注释
        q_r = q_w.mean(dim=2)  # 查询区域token  # 中文注释
        k_r = k_w.mean(dim=2)  # 键区域token  # 中文注释

        # 区域相关性(BH, R, R)并选Top-K(BH, R, K)  # 中文注释
        attn_r = torch.einsum("brd,bkd->brk", q_r, k_r) * self.scale  # 点积与缩放  # 中文注释
        topk = min(self.topk, r)  # 防止K大于区域数  # 中文注释
        topk_val, topk_idx = torch.topk(attn_r, k=topk, dim=-1)  # 选择最相关区域  # 中文注释

        # 令牌级注意力: 仅在被路由到的目标区域token集合内计算(BH, R, T, D)  # 中文注释
        out_w = q_w.new_zeros(bh_, r, t, d)  # 预分配输出窗口token  # 中文注释

        # 将(BH, R, T, D)视作BH批次列表, 对每个样本与每个区域进行小批计算, 保持梯度  # 中文注释
        for i in range(bh_):  # 遍历BH维度  # 中文注释
            k_win = k_w[i]  # (R, T, D) 键窗口库  # 中文注释
            v_win = v_w[i]  # (R, T, D) 值窗口库  # 中文注释
            q_win = q_w[i]  # (R, T, D) 查询窗口库  # 中文注释
            sel = topk_idx[i]  # (R, K) 每个查询区域的目标区域索引  # 中文注释
            # 遍历每个查询区域, 聚合其令牌到被路由区域的令牌上  # 中文注释
            for r_idx in range(r):  # 遍历区域  # 中文注释
                tgt_regions = sel[r_idx]  # (K,) 被选目标区域ID  # 中文注释
                k_sel = k_win.index_select(0, tgt_regions).reshape(topk * t, d)  # (K*T, D)  # 中文注释
                v_sel = v_win.index_select(0, tgt_regions).reshape(topk * t, d)  # (K*T, D)  # 中文注释
                q_tok = q_win[r_idx]  # (T, D) 查询区域内token  # 中文注释
                attn = torch.matmul(q_tok, k_sel.transpose(0, 1)) * self.scale  # (T, K*T) 注意力分数  # 中文注释
                attn = attn.softmax(dim=-1)  # softmax归一化  # 中文注释
                out_w[i, r_idx] = torch.matmul(attn, v_sel)  # (T, D) 输出令牌  # 中文注释

        # 将窗口序列还原回(BH, C_h, H_pad, W_pad)  # 中文注释
        out = self._window_reverse(out_w, ws, gh, gw)  # 反窗口化  # 中文注释
        out = out.permute(0, 2, 3, 1).contiguous().view(b, self.num_heads, (h + ph), (w + pw), self.head_dim)
        out = out.permute(0, 1, 4, 2, 3).contiguous().view(b, self.dim, (h + ph), (w + pw))  # 合并头通道  # 中文注释

        # 移除补齐并投影输出  # 中文注释
        if ph or pw:  # 若存在补齐  # 中文注释
            out = out[:, :, :h, :w]  # 裁剪回原尺寸  # 中文注释
        out = self.proj(out)  # 线性投影  # 中文注释
        return out  # 返回BRA结果  # 中文注释


class BiLevelRoutingAttentionFusionBlock(nn.Module):
    """
    双层路由注意力融合块(BRAF): C2f轻量提炼 + BRA全局依赖建模 + 1x1投影
    用途: 插入于颈部最高分辨率(P2/P3)路径, 强化小目标细节与长程上下文
    参数:
      c1: 输入通道; c2: 输出通道; num_heads: 头数; window_size: 窗口尺寸; topk: 区域路由Top-K; e: 隐藏通道比例
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 8,
        window_size: int = 8,
        topk: int = 4,
        e: float = 0.5,
    ):
        super().__init__()  # 父类构造  # 中文注释
        c_hidden = int(c2 * e)  # 隐藏通道数  # 中文注释
        c_hidden = max(32, (c_hidden // num_heads) * num_heads)  # 保证能被头数整除且不少于32  # 中文注释

        self.cv_in = Conv(c1, c_hidden, 1)  # 通道对齐到隐藏维度  # 中文注释
        self.c2f = C2f(c_hidden, c_hidden, n=1, shortcut=True)  # 轻量特征提炼  # 中文注释
        self.bra = BiLevelRoutingAttention(c_hidden, num_heads=num_heads, window_size=window_size, topk=topk)  # BRA  # 中文注释
        self.cv_out = Conv(c_hidden, c2, 1, act=False)  # 输出通道恢复  # 中文注释

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播  # 中文注释
        x = self.cv_in(x)  # 通道对齐  # 中文注释
        x = self.c2f(x)  # 轻量提炼  # 中文注释
        x = self.bra(x)  # 路由注意力  # 中文注释
        x = self.cv_out(x)  # 输出投影  # 中文注释
        return x  # 返回融合结果  # 中文注释


