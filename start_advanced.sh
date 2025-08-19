#!/bin/bash

# YOLO-SOD-Advanced 训练启动脚本

# --- 配置区 ---
# 数据集配置文件路径 (请根据您的实际路径修改)
DATASET_YAML="/workspace/yolo/ultralytics/cfg/VisDrone.yaml"

# 预训练权重 (可选, 留空则从头训练)
# 例如: PRETRAINED_WEIGHTS="yolov8l.pt"
PRETRAINED_WEIGHTS=""

# 训练设备 (例如: '0' or '0,1,2,3')
DEVICE='0'

# 实验名称
PROJECT_NAME="runs/YOLO-SOD-Advanced"
EXPERIMENT_NAME="exp_visdrone_500e"

# --- 执行区 ---
python train_advanced.py \
    --cfg /workspace/yolo/ultralytics/cfg/new/yolov12-sod-advanced.yaml \
    --hyp /workspace/yolo/ultralytics/cfg/new/hyp_advanced_500.yaml \
    --data ${DATASET_YAML} \
    --epochs 500 \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --device ${DEVICE} \
    --project ${PROJECT_NAME} \
    --name ${EXPERIMENT_NAME} \
    --weights ${PRETRAINED_WEIGHTS}

echo "训练任务已启动。请监控 '${PROJECT_NAME}/${EXPERIMENT_NAME}' 目录下的输出。"
