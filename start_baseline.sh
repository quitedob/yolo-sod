#!/bin/bash
# /workspace/yolo/start_baseline.sh
# 作用：YOLOv12n 基线训练脚本 - 使用官方配置进行 VisDrone 数据集训练

echo "=============================================="
echo "      YOLOv12n 基线训练脚本"
echo "=============================================="
echo "配置详情："
echo "  - 模型: yolov12n.pt (Nano版本)"
echo "  - 配置: yolov12.yaml (官方配置)"
echo "  - 数据集: VisDrone2019 (航拍小目标检测)"
echo "  - 训练轮数: 200"
echo "  - 早停耐心: 30 (30轮无提升则停止)"
echo "=============================================="

# 默认训练配置参数
DEFAULT_DATA="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml"
DEFAULT_EPOCHS=200
DEFAULT_IMGSZ=640
DEFAULT_BATCH=16
DEFAULT_DEVICE=0
DEFAULT_PATIENCE=30
DEFAULT_PROJECT="runs_baseline"
DEFAULT_NAME="yolov12n_visdrone_baseline"

# 读入命令行参数或使用默认值
DATA_PATH=${1:-$DEFAULT_DATA}
EPOCHS=${2:-$DEFAULT_EPOCHS}
BATCH_SIZE=${3:-$DEFAULT_BATCH}
IMG_SIZE=${4:-$DEFAULT_IMGSZ}
DEVICE=${5:-$DEFAULT_DEVICE}
PATIENCE=${6:-$DEFAULT_PATIENCE}
PROJECT=${7:-$DEFAULT_PROJECT}
NAME=${8:-$DEFAULT_NAME}

echo "训练配置参数："
echo "  数据集配置: $DATA_PATH"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  图像尺寸: $IMG_SIZE"
echo "  训练设备: $DEVICE"
echo "  早停耐心: $PATIENCE"
echo "  项目名称: $PROJECT"
echo "  实验名称: $NAME"
echo "=============================================="

# 检查必要的文件是否存在
if [ ! -f "$DATA_PATH" ]; then
  echo "错误：数据集配置文件未找到: $DATA_PATH"
  echo "请确认 VisDrone 数据集配置正确"
  exit 1
fi

if [ ! -f "/workspace/yolo/yolov12n.pt" ]; then
  echo "错误：预训练模型未找到: /workspace/yolo/yolov12n.pt"
  exit 1
fi

if [ ! -f "/workspace/yolo/ultralytics/cfg/models/v12/yolov12.yaml" ]; then
  echo "错误：模型配置文件未找到: /workspace/yolo/ultralytics/cfg/models/v12/yolov12.yaml"
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

# 启动训练 - 使用官方 YOLO 训练接口
python -c "
from ultralytics import YOLO
import torch

print('正在加载 YOLOv12n 模型...')
model = YOLO('/workspace/yolo/yolov12n.pt')

print('开始训练...')
results = model.train(
    data='$DATA_PATH',
    epochs=$EPOCHS,
    imgsz=$IMG_SIZE,
    batch=$BATCH_SIZE,
    device='$DEVICE',
    workers=8,
    project='$PROJECT',
    name='$NAME',
    exist_ok=True,
    pretrained=True,
    patience=$PATIENCE,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    verbose=True
)

print('训练完成！')
print(f'最佳模型: {model.trainer.best}')
print(f'最终模型: {model.trainer.last}')
"

# 检查训练结果
if [ $? -eq 0 ]; then
  echo "=============================================="
  echo "基线训练完成！"
  echo "结果保存在: $PROJECT/$NAME/"
  echo "=============================================="
else
  echo "=============================================="
  echo "训练过程中发生错误，请检查日志"
  echo "=============================================="
  exit 1
fi
