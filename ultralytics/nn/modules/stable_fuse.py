# 文件路径：/workspace/yolo/ultralytics/nn/modules/stable_fuse.py  # 提供数值稳定的融合与轻归一化模块
import torch  # 引入PyTorch主库  # 中文注释
import torch.nn as nn  # 引入神经网络模块  # 中文注释


class ChannelNorm(nn.Module):  # 定义通道归一化类  # 中文注释
    """通道归一化：按通道维做标准化，独立于批次，稳定特征分布"""  # 类说明  # 中文注释

    def __init__(self, eps: float = 1e-5):  # 初始化，设置数值稳定项eps  # 中文注释
        super().__init__()  # 调用父类构造函数  # 中文注释
        self.eps = eps  # 保存eps以避免除零  # 中文注释

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播接口  # 中文注释
        # 计算按通道的均值（B,C,H,W -> 在C维上求均值）  # 中文注释
        mean = x.mean(dim=1, keepdim=True)  # 通道均值  # 中文注释
        # 计算按通道的方差（无偏=False 更平滑）  # 中文注释
        var = x.var(dim=1, keepdim=True, unbiased=False)  # 通道方差  # 中文注释
        # 标准化：减均值除以标准差  # 中文注释
        x = (x - mean) / (var.add(self.eps).sqrt())  # 数值稳定的归一化  # 中文注释
        return x  # 返回归一化结果  # 中文注释


class ScaleAdd(nn.Module):  # 定义带可学习缩放的加法融合  # 中文注释
    """带可学习缩放的加法：z = x + alpha * y，alpha 初值小以避免早期方差过大"""  # 类说明  # 中文注释

    def __init__(self, init_alpha: float = 0.5):  # 初始化，设置alpha初值  # 中文注释
        super().__init__()  # 调用父类构造函数  # 中文注释
        # 使用可学习参数alpha并以float初始化  # 中文注释
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))  # 可学习缩放因子  # 中文注释

    def forward(self, x: torch.Tensor | list | tuple, y: torch.Tensor | None = None) -> torch.Tensor:  # 前向  # 中文注释
        # 兼容YAML传入列表/元组的两路输入  # 中文注释
        if y is None and isinstance(x, (list, tuple)):  # 如果只传了列表/元组  # 中文注释
            assert len(x) == 2, "ScaleAdd 期望两个输入特征"  # 断言两个输入  # 中文注释
            a, b = x[0], x[1]  # 拆包两路特征  # 中文注释
            return a + self.alpha * b  # 返回融合结果  # 中文注释
        # 常规两输入形式  # 中文注释
        assert y is not None, "ScaleAdd 缺少第二个输入"  # 检查第二输入  # 中文注释
        return x + self.alpha * y  # 返回融合结果  # 中文注释


