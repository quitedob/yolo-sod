# /workspace/yolo/ultralytics/nn/modules/recurrent_attention_fusion_block.py
# 主要功能简介: 提供循环-注意力融合模块(RAFB)与卷积GRU单元, 用于在颈部进行上下文感知的小目标增强融合, 提升稳定性与小目标检测性能  

import torch  # 导入PyTorch主包  
import torch.nn as nn  # 导入神经网络模块  
import torch.nn.functional as F  # 导入常用函数接口  

from .conv import Conv  # 从本工程卷积封装中导入Conv  
from .block import C2f  # 从本工程模块中导入C2f块  
from .stable_fuse import ChannelNorm, ScaleAdd  # 导入通道归一化与可学习残差缩放  
from .sageattention2 import SageAttention2, should_use_sageattention2_once, mark_sageattention2_choice  # 导入Sage注意力与一次性选择工具  


class ConvGRUCell(nn.Module):  # 定义卷积GRU单元类  
    """
    一个简化版的卷积门控循环单元（GRU）Cell，用于在特征图层面处理循环状态。  
    设计目标: 在多尺度/层级上携带上下文状态, 平衡小目标的上下文依赖与数值稳定性  
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):  # 初始化函数, 指定输入维度/隐藏维度/卷积核  
        super().__init__()  # 调用父类初始化  
        self.input_dim = input_dim  # 记录输入通道数  
        self.hidden_dim = hidden_dim  # 记录隐藏状态通道数  
        self.kernel_size = kernel_size  # 记录卷积核大小  
        self.padding = kernel_size // 2  # 设定padding使输出尺寸不变  

        # 门控卷积: 同时计算更新门与重置门 (2 * hidden_dim)  
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,  # 输入为当前输入与上一隐藏态拼接  
            out_channels=2 * hidden_dim,  # 输出包含两组门  
            kernel_size=self.kernel_size,  # 卷积核大小  
            padding=self.padding,  # 边缘填充  
            bias=True,  # 使用偏置  
        )  

        # 候选隐藏状态卷积: 生成候选h~  
        self.conv_can = nn.Conv2d(
            in_channels=input_dim + hidden_dim,  # 输入为当前输入与重置后的隐藏态拼接  
            out_channels=hidden_dim,  # 输出为隐藏态通道数  
            kernel_size=self.kernel_size,  # 卷积核大小  
            padding=self.padding,  # 边缘填充  
            bias=True,  # 使用偏置  
        )  

        # 采用通道归一化提升数值稳定性  
        self.cn_gates = ChannelNorm(2 * hidden_dim)  # 对门控输出做通道归一化  
        self.cn_can = ChannelNorm(hidden_dim)  # 对候选态做通道归一化  

    def forward(self, input_tensor: torch.Tensor, h_cur: torch.Tensor) -> torch.Tensor:  # 前向传播实现  
        # input_tensor: [B, Cin, H, W] 当前层级输入  
        # h_cur: [B, Hc, H, W] 上一层级隐藏状态  
        combined = torch.cat([input_tensor, h_cur], dim=1)  # 沿通道维拼接输入与隐藏态  
        combined_conv = self.conv_gates(combined)  # 通过门控卷积计算门值  
        combined_conv = self.cn_gates(combined_conv)  # 通道归一化以稳态  

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)  # 按通道拆分为两组门  
        reset_gate = torch.sigmoid(gamma)  # 重置门采用Sigmoid  
        update_gate = torch.sigmoid(beta)  # 更新门采用Sigmoid  

        combined_reset = torch.cat([input_tensor, reset_gate * h_cur], dim=1)  # 重置门调制隐藏态并与输入拼接  
        cc_cnm = self.conv_can(combined_reset)  # 计算候选隐藏态卷积输出  
        cc_cnm = self.cn_can(cc_cnm)  # 对候选态做通道归一化  
        h_next_candidate = torch.tanh(cc_cnm)  # 候选态通过tanh限制幅值  

        h_next = (1 - update_gate) * h_cur + update_gate * h_next_candidate  # 按更新门混合旧态与候选态  
        return h_next  # 返回下一隐藏态  


class RecurrentAttentionFusionBlock(nn.Module):  # 定义循环-注意力融合模块  
    """
    循环-注意力融合模块 (RAFB)  
    融合了ConvGRU上下文通路与轻量自注意力的细化通路, 并以上下文引导注意力输出, 强化小目标上下文建模。  

    Args:  
        c1 (int): 输入通道数(由解析器注入).  
        c2 (int): 输出通道数.  
        gru_hidden_dim (int): ConvGRU隐藏维度, 建议与c2同量级.  
        n_heads (int): 多头自注意力头数, 要整除内部嵌入维度.  
        e (float): 注意力前C2f的通道缩放因子, 控制嵌入维度.  
    """

    def __init__(self, c1: int, c2: int, gru_hidden_dim: int = 128, n_heads: int = 4, e: float = 0.5):  # 初始化RAFB  
        super().__init__()  # 调用父类初始化  

        # 保存配置  
        self.gru_hidden_dim = int(gru_hidden_dim)  # 记录GRU隐藏通道数  
        self.out_channels = int(c2)  # 记录输出通道数  
        # 计算注意力嵌入维度并保证能被多头数整除  
        base_embed = max(32, int(c2 * e))  # 基础嵌入维度下限32  
        if base_embed % int(n_heads) != 0:  # 若不能整除则上调到最近倍数  
            base_embed = (base_embed // int(n_heads) + 1) * int(n_heads)  
        self.embed_channels = int(base_embed)  # 最终嵌入维度  

        # 循环上下文通路: 使用ConvGRU建模跨层级上下文  
        self.conv_gru = ConvGRUCell(c1, self.gru_hidden_dim, kernel_size=3)  # 定义卷积GRU单元  

        # 注意力前置特征提取: 轻量C2f提取局部结构, 降低注意力成本  
        self.attn_c2f = C2f(c1, self.embed_channels, n=1, shortcut=True)  # C2f提取并压缩通道  

        # 注意力分支: 优先尝试SageAttention2, 失败则回退到MHSA  
        self.use_sage = should_use_sageattention2_once()  # 读取一次性启用标志  
        if self.use_sage:  # 若启用SageAttention2  
            try:  # 捕获潜在异常  
                self.attn = SageAttention2(self.embed_channels)  # 构建Sage注意力  
                self.attn_type = "sage"  # 记录模式  
                mark_sageattention2_choice(True)  # 标记选择成功  
            except Exception:  # 若构建失败则回退  
                self.use_sage = False  # 标记不可用  
                mark_sageattention2_choice(False)  # 记录回退  
        if not self.use_sage:  # 回退使用标准MHSA  
            # 多头自注意力: 在(H*W)范围做长程依赖建模  
            self.ln1 = nn.LayerNorm(self.embed_channels)  # 注意力前层归一化  
            self.mhsa = nn.MultiheadAttention(
                embed_dim=self.embed_channels,  # 嵌入维度  
                num_heads=int(n_heads),  # 头数  
                batch_first=True,  # 使用(B, L, E)格式  
            )  
            self.ln2 = nn.LayerNorm(self.embed_channels)  # FFN前层归一化  
            self.ffn = nn.Sequential(  # 前馈网络增强非线性表达  
                nn.Linear(self.embed_channels, self.embed_channels * 2),  # 线性升维  
                nn.GELU(),  # GELU激活  
                nn.Linear(self.embed_channels * 2, self.embed_channels),  # 线性降维回嵌入  
            )  
            self.attn_type = "mhsa"  # 记录模式  

        # 上下文引导: 由GRU隐藏态生成空间调制图, 引导注意力结果  
        self.context_gate = nn.Sequential(  # 上下文引导门  
            Conv(self.gru_hidden_dim, self.embed_channels, 1),  # 1x1卷积对齐通道  
            nn.Sigmoid(),  # Sigmoid生成[0,1]调制权重  
        )  

        # 输出整合: 使用可学习缩放残差, 提升初期数值稳定性  
        self.out_conv = Conv(self.embed_channels, self.out_channels, 1)  # 输出1x1卷积整合通道  
        self.res_proj = Conv(c1, self.out_channels, 1)  # 残差分支通道对齐投影  
        self.residual_add = ScaleAdd(init_alpha=0.5)  # 可学习残差缩放  

    def forward(self, x):  # 定义前向传播  
        # Ultralytics解析器传入可能是Tensor或list, 此处兼容处理  
        if isinstance(x, (list, tuple)):  # 如果是多输入  
            f_curr = x[0]  # 取第一个作为主特征  
            # 尝试从第二输入获取外部隐藏态(可选)  
            h_prev = None  
            if len(x) > 1 and isinstance(x[1], torch.Tensor) and x[1].dim() == 4:  # 二维特征图隐藏态  
                if x[1].shape[1] == self.gru_hidden_dim:  # 通道匹配才使用  
                    h_prev = x[1]  # 采用外部隐藏态  
        else:  # 否则直接使用Tensor  
            f_curr = x  # 指定当前特征图  
            h_prev = None  

        b, c, h, w = f_curr.shape  # 读取批量/通道/高/宽  
        if h_prev is None:  # 若无外部隐藏态则初始化为0  
            h_prev = torch.zeros(b, self.gru_hidden_dim, h, w, device=f_curr.device, dtype=f_curr.dtype)  # 初始化隐藏态为0  

        # 1) 循环上下文更新  
        h_curr = self.conv_gru(f_curr, h_prev)  # 通过ConvGRU更新隐藏状态  

        # 2) 注意力细化  
        f_attn_in = self.attn_c2f(f_curr)  # 先经C2f提取局部信息  
        if self.attn_type == "sage":  # 若使用Sage注意力  
            f_attn_out = self.attn(f_attn_in)  # 直接在[C,H,W]域内进行注意力  
        else:  # 使用MHSA  
            f_seq = f_attn_in.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, L, C]  
            f_seq = self.ln1(f_seq)  # 层归一化稳定注意力  
            attn_out, _ = self.mhsa(f_seq, f_seq, f_seq)  # 多头自注意力  
            f_seq = f_seq + attn_out  # 残差连接1  
            f_ffn = self.ffn(self.ln2(f_seq))  # 前馈网络  
            f_seq = f_seq + f_ffn  # 残差连接2  
            f_attn_out = f_seq.transpose(1, 2).view(b, self.embed_channels, h, w)  # 还原为[B, C, H, W]  

        # 3) 上下文引导融合  
        context_map = self.context_gate(h_curr)  # 由隐藏态生成调制图  
        f_fused = f_attn_out * context_map  # 逐元素调制注意力特征  

        # 4) 输出与稳定残差  
        f_out = self.out_conv(f_fused)  # 1x1卷积整合输出通道  
        res = self.res_proj(f_curr)  # 残差分支通道对齐  
        f_out = self.residual_add(f_out, res)  # 与投影后的残差做可学习缩放融合  

        return f_out  # 返回融合后的输出特征  


