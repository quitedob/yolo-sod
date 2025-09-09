# /workspace/yolo/ultralytics/nn/modules/blocks_mamba.py
# 作用：在骨干/颈部插入 Mamba 长序列状态空间建模模块，无 mamba-ssm 时回退到GLU门控卷积块
# 参考：Mamba 选择性SSM（线性时空复杂度）
# 动机：高分辨率特征图展平成序列后进行建模，弥补纯注意力在超长序列下的计算瓶颈
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1BN(nn.Sequential):
    """1x1卷积+BatchNorm+SiLU激活的组合模块"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, 1, bias=False),  # 1x1卷积用于通道变换
            nn.BatchNorm2d(c_out),  # 批归一化稳定训练
            nn.SiLU(inplace=True)  # SiLU激活函数
        )

class GLUBlock(nn.Module):
    """GLU门控卷积块，作为Mamba的回退方案"""
    def __init__(self, c: int, expansion: int = 2):
        super().__init__()
        hidden = c * expansion  # 扩展隐藏维度
        self.pw1 = nn.Conv2d(c, hidden * 2, 1, bias=False)  # 点卷积产生门控和内容
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)  # 深度卷积
        self.bn = nn.BatchNorm2d(hidden)  # 批归一化
        self.pw2 = nn.Conv2d(hidden, c, 1, bias=False)  # 输出投影
        self.act = nn.SiLU(inplace=True)  # 激活函数
    
    def forward(self, x):
        # GLU门控机制：将输入分为激活项和门控项
        a, g = self.pw1(x).chunk(2, dim=1)
        x = torch.sigmoid(g) * a  # 门控融合
        x = self.dw(x)  # 深度卷积特征提取
        x = self.bn(x)  # 批归一化
        x = self.act(x)  # 激活
        x = self.pw2(x)  # 输出投影
        return x

class MambaBlock(nn.Module):
    """Mamba状态空间模块，支持线性复杂度的长序列建模"""
    def __init__(self, c: int, c_hidden: int = 256, seq_reduction: int = 2):
        super().__init__()
        self.in_proj = Conv1x1BN(c, c_hidden)  # 输入投影
        self.out_proj = Conv1x1BN(c_hidden, c)  # 输出投影
        self.reduction = seq_reduction  # 序列长度缩减因子，降低计算复杂度
        
        # 测试MambaBlock回退机制
        print("Testing MambaBlock fallback mechanism...")
        try:
            # 尝试导入mamba-ssm库
            from mamba_ssm import Mamba
            # 测试causal_conv1d兼容性
            import torch
            test_input = torch.randn(1, c_hidden, 16)
            test_mamba = Mamba(d_model=c_hidden, d_state=16, d_conv=4, expand=2)
            _ = test_mamba(test_input)  # 测试前向传播
            
            self.use_mamba = True
            self.mamba = test_mamba
            print(f"[INFO] MambaBlock: 成功加载mamba-ssm，使用Mamba SSM进行长序列建模")
        except Exception as e:
            # 无法导入mamba-ssm或兼容性问题时使用GLU回退
            self.use_mamba = False
            self.fallback = GLUBlock(c_hidden, expansion=2)
            print(f"[WARNING] MambaBlock: mamba-ssm不兼容 ({e})，回退到GLU门控卷积")
    
    def forward(self, x):
        B, C, H, W = x.shape
        # 输入投影到隐藏维度
        y = self.in_proj(x)
        
        # 如果设置了缩减因子，先进行下采样减少序列长度
        if self.reduction > 1:
            y = F.avg_pool2d(y, self.reduction, self.reduction)
        
        Bh, Ch, Hh, Wh = y.shape
        # 将2D特征图展平为1D序列用于Mamba处理
        y = y.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        if self.use_mamba:
            # 使用Mamba进行长序列建模
            y = self.mamba(y)
        else:
            # 使用GLU回退方案
            y = y.transpose(1, 2).reshape(Bh, Ch, Hh, Wh)
            y = self.fallback(y)
            y = y.flatten(2).transpose(1, 2)
        
        # 重新整形为2D特征图
        y = y.transpose(1, 2).reshape(Bh, -1, Hh, Wh)
        
        # 如果进行了下采样，需要上采样回原尺寸
        if self.reduction > 1:
            y = F.interpolate(y, size=(H, W), mode='nearest')
        
        # 输出投影并添加残差连接
        y = self.out_proj(y)
        return x + y  # 残差连接保持特征稳定性
