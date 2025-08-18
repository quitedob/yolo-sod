#!/bin/bash
# /workspace/yolo/start.sh
# 作用：YOLO-SOD-Fusion增强版训练启动脚本（最小修复版：暂时移除 --hyp，确保可跑）

echo "=============================================="
echo "  YOLO-SOD-Fusion 增强版训练系统"
echo "=============================================="
echo "集成模块："
echo "  ★ MambaBlock / SwinBlock / DETR-Aux / BoundaryAwareLoss / DetectStable"
echo "=============================================="

# 【重要】如需恢复自定义超参文件：
#   1) 确认文件真实存在（例如：/workspace/yolo/ultralytics/cfg/new/hyp_hcp_400.yaml）
#   2) 在后面的 python 命令末尾追加：--hyp /workspace/yolo/ultralytics/cfg/new/hyp_hcp_400.yaml
#   3) train.py 会把它作为 cfg 覆盖（Ultralytics 机制）

# 默认训练配置参数（注意大小写保持和实际文件一致）
DEFAULT_DATA="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml"
DEFAULT_EPOCHS=500
DEFAULT_IMGSZ=640
DEFAULT_BATCH=16
DEFAULT_DEVICE=0
DEFAULT_CLOSE_P2_UNTIL=30

# 读入命令行参数或使用默认
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

# 检查数据集配置是否存在
if [ ! -f "$DATA_PATH" ]; then
  echo "错误：数据集配置文件未找到: $DATA_PATH"
  echo "请修正 DEFAULT_DATA 或传入正确路径"
  exit 1
fi

# 检查Python与依赖
if ! command -v python &> /dev/null; then
  echo "错误：未找到Python解释器"
  exit 1
fi

echo "检查依赖库..."
python -c "import torch, ultralytics" 2>/dev/null || {
  echo "错误：缺少必要的Python库（torch, ultralytics），请先：pip install -r requirements.txt"
  exit 1
}

echo "依赖检查通过，开始训练..."
echo "=============================================="

# 启动训练（去掉 --hyp，先确保训练跑通）
python /workspace/yolo/train.py \
  --cfg /workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-v5-stable.yaml \
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

# 结果检查
if [ $? -eq 0 ]; then
  echo "=============================================="
  echo "训练完成！结果保存在: runs_fusion/yolo_sod_fusion_exp/"
  echo "=============================================="
else
  echo "=============================================="
  echo "训练过程中发生错误，请检查日志"
  echo "=============================================="
  exit 1
fi
