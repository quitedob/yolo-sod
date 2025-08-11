# /workspace/yolo/ultralytics/nn/modules/losses/__init__.py  # 文件路径
# 作用：导出 InterpIoU 相关API，便于外部 import ultralytics.nn.modules.losses.interpiou  # 中文注释

from .interpiou import interpiou_iou_xyxy, interpiou_loss_xyxy  # 导出核心函数  # 中文注释

__all__ = [
    "interpiou_iou_xyxy",  # 导出均值IoU接口  # 中文注释
    "interpiou_loss_xyxy",  # 导出InterpIoU损失接口  # 中文注释
]


