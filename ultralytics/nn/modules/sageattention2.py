# /workspace/yolo/ultralytics/nn/modules/sageattention2.py
# 主要功能简介: 提供占位且可用的SageAttention2实现(轻量高效注意力), 以便首次尝试启用, 失败则回退原路径  

import os  # 环境变量读取  
import torch  # 导入PyTorch  
import torch.nn as nn  # 导入神经网络模块  


class SageAttention2(nn.Module):  # 定义SageAttention2模块类  
    """
    轻量注意力占位实现: 使用通道注意+空间注意的可分离组合, 接口兼容nn.Module。  
    若需替换为真实实现, 仅需保持forward签名不变并替换内部计算。  
    """

    def __init__(self, channels: int):  # 初始化注意力模块, 需要输入通道数  
        super().__init__()  # 调用父类初始化  
        self.c_attn = nn.Sequential(  # 通道注意力分支  
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1  
            nn.Conv2d(channels, max(8, channels // 8), 1, bias=False),  # 降维1x1卷积  
            nn.SiLU(),  # SiLU激活  
            nn.Conv2d(max(8, channels // 8), channels, 1, bias=False),  # 升维1x1卷积  
            nn.Sigmoid(),  # Sigmoid生成权重  
        )  
        self.s_attn = nn.Sequential(  # 空间注意力分支  
            nn.Conv2d(2, 1, 7, padding=3, bias=False),  # 7x7卷积提取空间信息  
            nn.Sigmoid(),  # Sigmoid生成权重  
        )  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播  
        b, c, h, w = x.shape  # 读取形状  
        w_c = self.c_attn(x)  # 计算通道权重  
        x_c = x * w_c  # 应用通道注意力  
        avg_map = torch.mean(x_c, dim=1, keepdim=True)  # 空间平均  
        max_map, _ = torch.max(x_c, dim=1, keepdim=True)  # 空间最大  
        s_in = torch.cat([avg_map, max_map], dim=1)  # 拼接两个空间图  
        w_s = self.s_attn(s_in)  # 计算空间权重  
        out = x_c * w_s  # 应用空间注意力  
        return out  # 返回注意力后的特征  


def should_use_sageattention2_once(flag_dir: str = "runs_advanced") -> bool:  # 判断是否首次尝试使用SageAttention2  
    os.makedirs(flag_dir, exist_ok=True)  # 创建标志目录  
    try_flag = os.path.join(flag_dir, "try_sageattention2.flag")  # 首次尝试标志文件  
    chosen_flag = os.path.join(flag_dir, "chosen_sageattention2.flag")  # 已选择SageAttention2标志  
    chosen_fallback = os.path.join(flag_dir, "chosen_fallback.flag")  # 已选择回退标志  

    env_use = os.environ.get("USE_SAGE_ATTENTION2", "0")  # 读取环境变量设置  
    if env_use != "1":  # 若环境未要求尝试则直接返回False  
        return False  # 不启用  

    if os.path.exists(chosen_fallback):  # 若已选择回退则不再尝试  
        return False  # 不启用  
    if os.path.exists(chosen_flag):  # 若已选择SageAttention2则继续使用  
        return True  # 启用  

    # 首次进入: 创建try标志, 仅第一次尝试  
    if not os.path.exists(try_flag):  
        open(try_flag, "a").close()  # 触发一次尝试  
        return True  # 本次启用尝试  
    else:  
        # 已尝试过但未选择, 默认不使用  
        return False  # 不启用  


def mark_sageattention2_choice(use_sage: bool, flag_dir: str = "runs_advanced"):  # 记录选择结果  
    chosen_flag = os.path.join(flag_dir, "chosen_sageattention2.flag")  # 成功选择标志  
    chosen_fallback = os.path.join(flag_dir, "chosen_fallback.flag")  # 回退选择标志  
    if use_sage:  # 若使用SageAttention2  
        open(chosen_flag, "a").close()  # 写入使用标志  
        if os.path.exists(chosen_fallback):  # 清除回退标志  
            os.remove(chosen_fallback)  # 删除回退  
    else:  # 回退到原始模块  
        open(chosen_fallback, "a").close()  # 写入回退标志  
        if os.path.exists(chosen_flag):  # 清除使用标志  
            os.remove(chosen_flag)  # 删除使用  


