#!/bin/bash
# /workspace/yolo/start.sh
# 主要功能简介: 一键启动高级分层训练流程, 使用自定义模型与超参在VisDrone上训练并稳定小目标检测  

set -e  # 遇错退出, 确保流程安全  

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # 设置默认可见GPU为0  

echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"  # 打印使用GPU信息  

PYTHON=${PYTHON:-python}  # 选择Python解释器  

CFG=${CFG:-/workspace/yolo/ultralytics/cfg/models/new/yolov12-smallobj-advanced.yaml}  # 指定模型配置YAML  
USE_SAGE=${USE_SAGE:-0}  # 是否尝试启用SageAttention2(1启用/0关闭)  

echo "[INFO] Using model cfg: $CFG"  # 打印模型配置路径  
echo "[INFO] Try SageAttention2: $USE_SAGE"  # 打印注意力开关  

# 确保日志目录存在  
mkdir -p runs_advanced  # 创建运行日志目录  

# 注意: 不使用续行反斜杠以避免意外空参数传入argparse  
$PYTHON /workspace/yolo/train_advanced.py --cfg "$CFG" --use_sageattention2 "$USE_SAGE" 2>&1 | tee -a runs_advanced/train.log  # 启动分层训练并记录日志  


