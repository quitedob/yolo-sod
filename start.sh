#!/bin/bash
# /workspace/yolo/start.sh
# 作用：YOLO-SOD-Fusion增强版训练启动脚本，集成所有先进模块
# 功能：Mamba+Swin+DETR-Aux+边界感知损失+P2启停控制
# 说明：这是项目的唯一训练启动入口点

echo "=============================================="
echo "  YOLO-SOD-Fusion 增强版训练系统"
echo "=============================================="
echo "集成模块："
echo "  ★ MambaBlock - 长序列状态空间建模"
echo "  ★ SwinBlock - 窗口注意力融合"
echo "  ★ DETR-Aux - 辅助检测头"
echo "  ★ BoundaryAwareLoss - 边界感知损失"
echo "  ★ MultiObjectTracker - 时序后处理"
echo "  ★ DetectStable - P2尺度启停控制"
echo "=============================================="

# 默认训练配置参数
DEFAULT_DATA="/workspace/yolo/ultralytics/cfg/datasets/VisDrone.yaml"  # 请修改为您的数据集配置文件路径
DEFAULT_EPOCHS=500
DEFAULT_IMGSZ=640
DEFAULT_BATCH=16
DEFAULT_DEVICE=0
DEFAULT_CLOSE_P2_UNTIL=30

# 检查命令行参数
DATA_PATH=${1:-$DEFAULT_DATA}
EPOCHS=${2:-$DEFAULT_EPOCHS}
BATCH_SIZE=${3:-$DEFAULT_BATCH}
IMG_SIZE=${4:-$DEFAULT_IMGSZ}
DEVICE=${5:-$DEFAULT_DEVICE}
CLOSE_P2_UNTIL=${6:-$DEFAULT_CLOSE_P2_UNTIL}

echo "训练配置参数："
echo "  数据集配置: $DATA_PATH"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  图像尺寸: $IMG_SIZE"
echo "  训练设备: $DEVICE"
echo "  P2关闭轮数: $CLOSE_P2_UNTIL"
echo "=============================================="

# 检查数据集配置文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误：数据集配置文件未找到: $DATA_PATH"
    echo "请确保数据集配置文件存在，或修改 start.sh 中的 DEFAULT_DATA 变量"
    exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误：未找到Python解释器"
    exit 1
fi

# 检查必要的Python库
echo "检查依赖库..."
python -c "import torch, ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误：缺少必要的Python库（torch, ultralytics）"
    echo "请运行：pip install -r requirements.txt"
    exit 1
fi

echo "依赖检查通过，开始训练..."
echo "=============================================="

# 执行训练脚本，启用所有增强功能
python /workspace/yolo/train.py \
    --cfg /workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-v5-all.yaml \
    --data "$DATA_PATH" \
    --epochs $EPOCHS \
    --imgsz $IMG_SIZE \
    --batch $BATCH_SIZE \
    --device $DEVICE \
    --workers 8 \
    --close_p2_until $CLOSE_P2_UNTIL \
    --use_boundary_loss \
    --use_detr_aux \
    --edge_weight 1.0 \
    --bce_weight 1.0 \
    --iou_weight 0.0 \
    --boundary_loss_weight 0.2 \
    --lr0 0.01 \
    --lrf 0.01 \
    --optimizer auto \
    --project runs_fusion \
    --name yolo_sod_fusion_exp

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "训练完成！"
    echo "结果保存在: runs_fusion/yolo_sod_fusion_exp/"
    echo "=============================================="
else
    echo "=============================================="
    echo "训练过程中发生错误，请检查日志"
    echo "=============================================="
    exit 1
fi