#!/bin/bash
# /workspace/yolo/start.sh
# 作用：YOLO-SOD-Fusion增强版训练启动脚本（最小修复版：暂时移除 --hyp，确保可跑）

echo "=============================================="
echo "  YOLOv12-SOD-Fusion-v5 增强版训练系统"
echo "=============================================="
echo "集成模块："
echo "  ★ SE_Block / CBAM_Block / CA_Block / A2_Attn"
echo "  ★ SwinBlock / DETR-Aux / BoundaryAwareLoss / DetectStable"
echo "  ★ MambaBlock（自动回退到GLU门控卷积）"
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

# 启动训练（使用新的YOLOv12-SOD-Fusion-v5分段训练脚本）
# 注意：MambaBlock现在有自动回退机制，可以使用完整版本
python /workspace/yolo/train_yolov12_staged.py \
  --cfg /workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-v5-simple.yaml \
  --data "$DATA_PATH" \
  --epochs $EPOCHS \
  --imgsz $IMG_SIZE \
  --batch $BATCH_SIZE \
  --device $DEVICE \
  --workers 8 \
  --close_p2_until $CLOSE_P2_UNTIL \
  --project runs_yolov12_staged \
  --name yolov12_sod_fusion_v5_exp

# 结果检查
if [ $? -eq 0 ]; then
  echo "=============================================="
  echo "训练完成！结果保存在: runs_yolov12_staged/yolov12_sod_fusion_v5_exp/"
  echo "=============================================="
else
  echo "=============================================="
  echo "训练过程中发生错误，请检查日志"
  echo "=============================================="
  exit 1
fi
