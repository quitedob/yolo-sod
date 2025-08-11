#!/bin/bash
# /workspace/yolo/start.sh  # 文件路径：训练启动脚本（两阶段+日志），使用 Bash 数组避免续行空格问题  # 中文注释
set -euo pipefail  # 严格模式：出错即停  # 中文注释

RUN_NAME="v12_$(date +%Y%m%d_%H%M%S)"  # 生成唯一运行名  # 中文注释

# 组装参数为数组，避免续行空格/CRLF带来的“空白参数”  # 中文注释
ARGS=(
  --model /workspace/yolo/ultralytics/cfg/models/new/yolov12-mambafusion-smallobj-640.yaml  # 模型YAML  # 中文注释
  --data  /workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml                             # 数据集YAML（小写）  # 中文注释
  --hyp   /workspace/yolo/ultralytics/cfg/models/new/hyp_smallobj_stable.yaml               # 更稳超参  # 中文注释
  --imgsz 640                                                                                # 输入分辨率  # 中文注释
  --epochs 300                                                                               # 总轮数  # 中文注释
  --stage1_epochs 40                                                                         # 阶段1轮数  # 中文注释
  --freeze_backbone 14                                                                       # 冻结层数  # 中文注释
  --batch -1                                                                                 # 自动批大小  # 中文注释
  --device 0                                                                                 # GPU设备  # 中文注释
  --workers 8                                                                                # DataLoader线程  # 中文注释
  --amp                                                                                      # 启用AMP  # 中文注释
  --lr_s1 0.002                                                                              # 阶段1学习率  # 中文注释
  --lr_s2 0.0015                                                                             # 阶段2学习率  # 中文注释
  --warmup_epochs 10                                                                         # Warmup轮数  # 中文注释
  --clip 1.0                                                                                 # 梯度裁剪阈值  # 中文注释
  --project runs/train                                                                       # 输出目录  # 中文注释
  --name "$RUN_NAME"                                                                         # 实验名  # 中文注释
)

# 调用两阶段训练入口（内置自动日志）  # 中文注释
python /workspace/yolo/train_visdrone_smallobj.py "${ARGS[@]}"