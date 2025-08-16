#!/bin/bash
# /workspace/yolo/start.sh
# 主要功能简介: 一键启动YOLO-SOD-Fusion混合架构的HCP-400分层收敛协议训练
# 基于2025年最新研究：MambaFusion、BiCAN、InterpIoU、DetectStable

set -e  # 遇错退出，确保流程安全

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # 设置默认可见GPU为0

echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"  # 打印使用GPU信息

PYTHON=${PYTHON:-python}  # 选择Python解释器

# 配置文件路径
CFG=${CFG:-/workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion.yaml}  # YOLO-SOD-Fusion架构配置
HYP=${HYP:-/workspace/yolo/ultralytics/cfg/models/new/hyp_hcp_400.yaml}  # HCP-400超参数配置
DATA=${DATA:-/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml}  # VisDrone数据集配置

# 训练参数
EPOCHS=${EPOCHS:-400}  # 训练轮数(默认400，HCP-400协议)
BATCH_SIZE=${BATCH_SIZE:-16}  # 批次大小
IMG_SIZE=${IMG_SIZE:-640}  # 图像尺寸
DEVICE=${DEVICE:-0}  # 训练设备

# 训练模式选择
TRAIN_MODE=${TRAIN_MODE:-"simple"}  # 训练模式：simple(测试) / hcp400(完整) / standard(标准)

echo "[INFO] ========================================="
echo "[INFO] YOLO-SOD-Fusion 小目标检测训练启动"
echo "[INFO] ========================================="
echo "[INFO] 模型配置: $CFG"
echo "[INFO] 超参数配置: $HYP"
echo "[INFO] 数据集配置: $DATA"
echo "[INFO] 训练模式: $TRAIN_MODE"
echo "[INFO] 训练轮数: $EPOCHS"
echo "[INFO] 批次大小: $BATCH_SIZE"
echo "[INFO] 图像尺寸: $IMG_SIZE"
echo "[INFO] 训练设备: $DEVICE"
echo "[INFO] ========================================="

# 确保日志目录存在
mkdir -p runs_hcp_400  # 创建HCP-400运行日志目录
mkdir -p runs_standard  # 创建标准训练日志目录
mkdir -p runs_test      # 创建测试运行日志目录

# 根据训练模式选择启动方式
if [ "$TRAIN_MODE" = "simple" ]; then
    echo "[INFO] 启动简化测试训练模式..."
    
    # 启动简化训练（用于测试）
    $PYTHON /workspace/yolo/train_hcp_400_simple.py \
        2>&1 | tee -a runs_test/simple_test.log

elif [ "$TRAIN_MODE" = "hcp400" ]; then
    echo "[INFO] 启动HCP-400分层收敛协议训练..."
    
    # 启动HCP-400训练
    $PYTHON /workspace/yolo/train_hcp_400.py \
        --cfg "$CFG" \
        --hyp "$HYP" \
        --data "$DATA" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        2>&1 | tee -a runs_hcp_400/hcp400_train.log

elif [ "$TRAIN_MODE" = "standard" ]; then
    echo "[INFO] 启动标准训练流程..."
    
    # 启动标准训练
    $PYTHON /workspace/yolo/train.py \
        --cfg "$CFG" \
        --hyp "$HYP" \
        --data "$DATA" \
        --epochs "$EPOCHS" \
        --batch "$BATCH_SIZE" \
        --imgsz "$IMG_SIZE" \
        --device "$DEVICE" \
        2>&1 | tee -a runs_standard/standard_train.log

else
    echo "[ERROR] 未知的训练模式: $TRAIN_MODE"
    echo "[INFO] 支持的模式: simple(测试), hcp400(完整), standard(标准)"
    exit 1
fi

echo "[INFO] 训练完成！"
echo "[INFO] 日志文件保存在: runs_${TRAIN_MODE}/"


