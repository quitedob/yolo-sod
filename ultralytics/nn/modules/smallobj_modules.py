"""
/workspace/yolo/ultralytics/nn/modules/smallobj_modules.py
小目标增强模块集合：SE、MixedAttention、SpaceToDepth、OmniKernelFusion、HyperACEBlock、DecoupledHeadLite、Add
可选集成 SageAttention 2.2.0（若 /workspace/SageAttention 或已安装环境可用）
"""

from __future__ import annotations

import os  # 引入os库，用于读取环境变量和路径
import sys  # 引入sys库，用于动态添加依赖路径
from typing import Tuple  # 引入类型注解Tuple，提高代码可读性

import torch  # 引入PyTorch主库
import torch.nn as nn  # 引入神经网络模块
import torch.nn.functional as F  # 引入函数式API


# —— 可选导入 SageAttention ——
_has_sageattention: bool = False  # 定义布尔标记，指示是否可用SageAttention
sageattn = None  # 预留sageattn句柄

try:
    # 优先尝试已安装环境
    from sageattention import sageattn as _sageattn  # 尝试从已安装包导入
    sageattn = _sageattn  # 绑定到本地变量
    _has_sageattention = True  # 标记可用
except Exception:
    # 尝试从指定本地路径加载
    sa_root = "/workspace/SageAttention"  # 指定SageAttention源代码根路径
    if os.path.isdir(sa_root):  # 判断目录是否存在
        if sa_root not in sys.path:  # 若未在sys.path中则添加
            sys.path.append(sa_root)  # 动态添加搜索路径
        try:
            from sageattention import sageattn as _sageattn  # 再次尝试导入
            sageattn = _sageattn  # 绑定到本地变量
            _has_sageattention = True  # 标记可用
        except Exception:
            _has_sageattention = False  # 保持不可用状态


class Add(nn.Module):
    """特征相加模块：接受列表或元组输入，逐元素相加输出"""

    def __init__(self) -> None:
        super().__init__()  # 调用父类构造，初始化模块

    def forward(self, xs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # 处理输入为列表/元组或单张量情况
        if isinstance(xs, (list, tuple)):  # 若为列表或元组
            y = xs[0]  # 取第一个张量作为初始和
            for t in xs[1:]:  # 遍历后续张量
                y = y + t  # 执行逐元素相加
            return y  # 返回相加结果
        return xs  # 若为单张量，直接返回


class SE(nn.Module):
    """SE 通道注意力（轻量）：全局池化 -> 两层1x1 -> Sigmoid 权重

    惰性构建（lazy init）：首次前向按输入通道创建权重，并在后续复用。
    注意：为兼容 AMP，参数始终保持 float32，仅对输入特征做临时类型适配，
    避免 GradScaler 报 “Attempting to unscale FP16 gradients”。
    """

    def __init__(self, reduction: int = 16) -> None:
        super().__init__()  # 初始化父类
        self.reduction = reduction  # 保存通道压缩比例
        self.fc1: nn.Conv2d | None = None  # 延迟创建的降维卷积
        self.fc2: nn.Conv2d | None = None  # 延迟创建的升维卷积
        self.in_channels: int | None = None  # 记录构建时的输入通道

    def _maybe_build(self, c: int, device):
        """按需构建或重建1x1卷积，并对齐 device（dtype 固定为 float32）"""
        hidden = max(c // self.reduction, 4)  # 计算隐藏层通道数，至少为4
        if (self.fc1 is None) or (self.in_channels != c):  # 首次或通道变更
            self.fc1 = nn.Conv2d(c, hidden, 1, bias=True)  # 创建降维1x1
            self.fc2 = nn.Conv2d(hidden, c, 1, bias=True)  # 创建升维1x1
            self.in_channels = c  # 记录输入通道
        # 每次前向都确保对齐 device，dtype 固定为 float32 以兼容 AMP
        assert self.fc1 is not None and self.fc2 is not None  # 静态检查
        self.fc1.to(device=device, dtype=torch.float32)  # 同步设备，保持FP32
        self.fc2.to(device=device, dtype=torch.float32)  # 同步设备，保持FP32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape  # 解析输入维度
        self._maybe_build(c, x.device)  # 惰性构建并对齐 device（保持权重FP32）
        m = F.adaptive_avg_pool2d(x, 1)  # 全局平均池化到1x1
        m32 = m.to(dtype=torch.float32)  # 临时转FP32与权重匹配
        a = F.relu(self.fc1(m32), inplace=True)  # 第一次1x1 + ReLU
        a = torch.sigmoid(self.fc2(a))  # 第二次1x1 + Sigmoid
        a = a.to(dtype=x.dtype)  # 回转到输入 dtype
        return x * a  # 对输入按通道加权


class MixedAttention(nn.Module):
    """通道+空间混合注意力（类似CBAM），可选用于Neck融合块内"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()  # 初始化父类
        hidden = max(channels // reduction, 4)  # 计算隐藏通道
        self.ca_conv1 = nn.Conv2d(channels, hidden, 1)  # 通道注意力降维
        self.ca_conv2 = nn.Conv2d(hidden, channels, 1)  # 通道注意力升维
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3)  # 空间注意力7x7卷积

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.adaptive_avg_pool2d(x, 1)  # 全局平均池化
        w = torch.sigmoid(self.ca_conv2(F.relu(self.ca_conv1(g), inplace=True)))  # 通道权重
        x = x * w  # 通道加权
        avg_map = torch.mean(x, dim=1, keepdim=True)  # 求通道均值图
        max_map, _ = torch.max(x, dim=1, keepdim=True)  # 求通道最大图
        s = torch.sigmoid(self.sa_conv(torch.cat([avg_map, max_map], dim=1)))  # 空间权重
        return x * s  # 空间加权


class SpaceToDepth(nn.Module):
    """空间转深度：以因子r将(H,W)折叠到通道，尺寸/ r"""

    def __init__(self, r: int = 2) -> None:
        super().__init__()  # 初始化父类
        self.r = r  # 保存下采样因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(x, downscale_factor=self.r)  # 使用像素反洗牌实现


class OmniKernelFusion(nn.Module):
    """OmniKernel 三分支融合：全局(池化+1x1) + 大核DW(5x5) + 局部DW(3x3) + 可选SageAttention分支"""

    def __init__(self, in_ch: int, out_ch: int, attn_heads: int = 4) -> None:
        super().__init__()  # 初始化父类
        self.gp_conv = nn.Conv2d(in_ch, out_ch, 1)  # 全局池化后的1x1卷积
        self.dw5 = nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch)  # 5x5深度可分离卷积
        self.pw5 = nn.Conv2d(in_ch, out_ch, 1)  # 1x1逐点卷积
        self.dw3 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)  # 3x3深度可分离卷积
        self.pw3 = nn.Conv2d(in_ch, out_ch, 1)  # 1x1逐点卷积
        self.out_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # 融合后的3x3卷积

        # 可选：SageAttention 投影层（仅当检测到可用且显式开启时有效）
        self.use_sage = (
            _has_sageattention
            and os.environ.get("SAGEATTN_ENABLE", "0") == "1"
        )  # 需设置环境变量 SAGEATTN_ENABLE=1 才启用
        self.attn_heads = attn_heads  # 注意力头数
        if self.use_sage:  # 若可用
            self.qkv = nn.Conv2d(in_ch, out_ch * 3, 1, bias=False)  # 生成QKV的1x1卷积
            self.proj = nn.Conv2d(out_ch, out_ch, 1, bias=False)  # 注意力输出投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局分支：全局池化+1x1上采
        g = F.adaptive_avg_pool2d(x, (1, 1))  # 全局池化到1x1
        g = self.gp_conv(g)  # 1x1卷积
        g = F.interpolate(g, size=x.shape[-2:], mode="nearest")  # 上采样回原尺寸

        # 卷积分支：大核与局部
        l5 = self.pw5(F.relu(self.dw5(x), inplace=True))  # 5x5 DW + 1x1 PW
        l3 = self.pw3(F.relu(self.dw3(x), inplace=True))  # 3x3 DW + 1x1 PW

        y = g + l5 + l3  # 融合三分支

        # 可选：SageAttention 空间注意力（将HxW视作序列）
        if self.use_sage and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            b, c, h, w = x.shape  # 解析维度
            qkv = self.qkv(x.contiguous())  # 生成QKV三倍通道
            q, k, v = torch.chunk(qkv, 3, dim=1)  # 切分为Q/K/V
            heads = self.attn_heads  # 读取注意力头数
            head_dim = max(c // heads, 16)  # 计算每头维度（下限16）
            # 调整到 (B, heads, seq_len, head_dim) 的布局
            def reshape_to_hnd(t: torch.Tensor) -> torch.Tensor:
                t = t.contiguous().view(b, heads, -1, (t.shape[1] // heads))  # (B,H,HW,C//H)
                t = t[:, :, : h * w, : head_dim]  # 截断到匹配维度
                return t.contiguous()  # 返回连续张量

            q_hnd = reshape_to_hnd(q)  # 重排Q
            k_hnd = reshape_to_hnd(k)  # 重排K
            v_hnd = reshape_to_hnd(v)  # 重排V

            try:
                torch.cuda.synchronize()  # 同步以更好地捕获潜在内核错误
                attn_out = sageattn(q_hnd, k_hnd, v_hnd, tensor_layout="HND", is_causal=False)  # 调用SageAttention
                torch.cuda.synchronize()  # 同步确保完成
                attn_out = attn_out.reshape(b, -1, h, w).contiguous()  # 还原空间布局
                y = y + self.proj(attn_out)  # 投影并与卷积融合
            except Exception:
                # 若注意力失败则忽略该分支，保障健壮性
                pass  # 直接跳过SageAttention

        return F.relu(self.out_conv(y), inplace=True)  # 输出3x3卷积并激活


class HyperACEBlock(nn.Module):
    """高阶相关增强融合块：x_high(上采样/高层) + x_low(低层) -> out"""

    def __init__(self, ch_high: int, ch_low: int, ch_out: int) -> None:
        super().__init__()  # 初始化父类
        self.ph = nn.Conv2d(ch_high, ch_out, 1, bias=False)  # 高层特征投影到统一通道
        self.pl = nn.Conv2d(ch_low, ch_out, 1, bias=False)  # 低层特征投影到统一通道
        self.bn_h = nn.BatchNorm2d(ch_out)  # BN归一化（高层）
        self.bn_l = nn.BatchNorm2d(ch_out)  # BN归一化（低层）
        self.hyper = nn.Conv2d(ch_out * 2, ch_out, 1, bias=False)  # 近似“超边”融合
        self.fuse = nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False)  # 3x3融合卷积
        self.bn_f = nn.BatchNorm2d(ch_out)  # 融合后BN

    def forward(self, x_high: torch.Tensor | list | tuple, x_low: torch.Tensor | None = None) -> torch.Tensor:
        # 兼容Ultralytics解析：当上游以list/tuple传入两路特征时，自动拆包
        if x_low is None and isinstance(x_high, (list, tuple)):
            assert len(x_high) == 2, "HyperACEBlock 期望两个输入特征 (x_high, x_low)"
            x_high, x_low = x_high[0], x_high[1]
        if x_high.shape[-2:] != x_low.shape[-2:]:  # 若空间尺寸不一致
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode="nearest")  # 上采样对齐
        h = F.relu(self.bn_h(self.ph(x_high)), inplace=True)  # 高层投影并激活
        l = F.relu(self.bn_l(self.pl(x_low)), inplace=True)  # 低层投影并激活
        z = torch.cat([h, l], dim=1)  # 拼接形成高阶“超边”输入
        z = F.relu(self.hyper(z), inplace=True)  # 1x1融合高阶相关
        z = F.relu(self.bn_f(self.fuse(z)), inplace=True)  # 3x3整合并归一
        return z  # 返回融合结果


class DWConv(nn.Module):
    """深度可分离卷积：DW(3x3) + PW(1x1)，降低FLOPs"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()  # 初始化父类
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)  # 深度卷积
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)  # 逐点卷积
        self.bn = nn.BatchNorm2d(out_ch)  # BN归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)  # 执行DW卷积
        x = self.pw(x)  # 执行PW卷积
        return F.relu(self.bn(x), inplace=True)  # BN+ReLU激活


class DecoupledHeadLite(nn.Module):
    """轻量解耦头：分类/置信 与 回归 分离；Anchor-Free（点到框）"""

    def __init__(self, in_ch: int, mid_ch: int, num_classes: int) -> None:
        super().__init__()  # 初始化父类
        # 分类+置信/质量共享特征
        self.cls1 = DWConv(in_ch, mid_ch)  # 第一层DW卷积（分类分支）
        self.cls2 = DWConv(mid_ch, mid_ch)  # 第二层DW卷积（分类分支）
        # 回归特征
        self.reg1 = DWConv(in_ch, mid_ch)  # 第一层DW卷积（回归分支）
        self.reg2 = DWConv(mid_ch, mid_ch)  # 第二层DW卷积（回归分支）
        # 输出头
        self.cls_out = nn.Conv2d(mid_ch, num_classes, 1)  # 分类输出
        self.obj_out = nn.Conv2d(mid_ch, 1, 1)  # 对象置信输出
        self.ctr_out = nn.Conv2d(mid_ch, 1, 1)  # 中心度输出
        self.iou_out = nn.Conv2d(mid_ch, 1, 1)  # IoU质量输出
        self.box_out = nn.Conv2d(mid_ch, 4, 1)  # 框回归输出（xywh）

    def forward(self, x: torch.Tensor):
        c = self.cls2(self.cls1(x))  # 分类分支特征
        r = self.reg2(self.reg1(x))  # 回归分支特征
        cls = self.cls_out(c)  # 分类热图
        obj = self.obj_out(c)  # 置信热图
        ctr = self.ctr_out(c)  # 中心度热图
        iou = self.iou_out(c)  # IoU质量热图
        box = self.box_out(r)  # 坐标回归
        return cls, obj, ctr, iou, box  # 返回五头输出


# Create alias for SE_Block to match YAML configuration
SE_Block = SE

__all__ = [
    "Add",  # 导出Add用于YAML解析
    "SE",  # 导出SE注意力模块
    "SE_Block",  # 导出SE_Block别名
    "MixedAttention",  # 导出混合注意力模块
    "SpaceToDepth",  # 导出空间转深度模块
    "OmniKernelFusion",  # 导出三分支融合模块
    "HyperACEBlock",  # 导出高阶相关融合块
    "DecoupledHeadLite",  # 导出轻量解耦检测头
]


