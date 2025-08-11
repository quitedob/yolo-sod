# 文件路径：/workspace/yolo/ultralytics/nn/modules/detect_stable.py  # 可控Detect，按尺度屏蔽训练参与
import torch  # 引入PyTorch主库  # 中文注释
import torch.nn as nn  # 引入神经网络模块  # 中文注释
from ultralytics.nn.modules.head import Detect as BaseDetect  # 复用Ultralytics原Detect  # 中文注释


class DetectStable(BaseDetect):  # 扩展Detect，支持active_mask控制尺度参与  # 中文注释
    """扩展版 Detect：通过 active_mask 选择性关闭某些尺度在训练期的损失与回传"""  # 类说明  # 中文注释

    def __init__(self, nc=80, ch=(), **kwargs):  # 初始化，保持与基类参数一致  # 中文注释
        super().__init__(nc=nc, ch=ch, **kwargs)  # 调用基类构造（此时已知 nl/ch）  # 中文注释
        # 注册缓冲区：布尔掩码，长度与多尺度头数量一致，默认全开启  # 中文注释
        self.register_buffer("active_mask", torch.ones(self.nl, dtype=torch.bool))  # 掩码  # 中文注释

    def set_active_mask(self, mask: list[bool]):  # 设置激活掩码的接口  # 中文注释
        assert len(mask) == self.nl, "active_mask 长度需与尺度数一致"  # 长度校验  # 中文注释
        with torch.no_grad():  # 无梯度区以避免污染  # 中文注释
            self.active_mask[:] = torch.tensor(mask, dtype=torch.bool, device=self.active_mask.device)  # 赋值  # 中文注释

    def forward(self, x):  # 前向：复用 Detect 的实现并在每尺度输出处应用掩码  # 中文注释
        outs = []  # 存放各尺度输出列表  # 中文注释
        for i in range(self.nl):  # 遍历各尺度  # 中文注释
            yi = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 与基类一致的逐尺度头  # 中文注释
            if self.training and (not bool(self.active_mask[i])):  # 训练期关闭时  # 中文注释
                yi = yi.detach() * 0.0  # 阻断梯度并置零，避免参与损失  # 中文注释
            outs.append(yi)  # 收集  # 中文注释
        if self.training:  # 训练路径直接返回列表  # 中文注释
            return outs  # 中文注释
        y = self._inference(outs)  # 推理路径：解码与拼接  # 中文注释
        return y if self.export else (y, outs)  # 与基类保持一致的返回  # 中文注释


