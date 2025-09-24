#!/bin/bash
# 文件路径: /workspace/yolo/ablation/start.sh
# 作用：检查环境 -> 顺序运行 /workspace/yolo/ablation/train.py（会自动把 E1~E6 全部跑完）
# 更新：默认 batch size 已修改为 10

set -e

echo "================ Ablation Runner (E1-E6) ================"

# ---- 默认参数（可由命令行位置参数覆盖） ----
DATA_DEFAULT="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml"
EPOCHS_DEFAULT=400
IMGSZ_DEFAULT=640
BATCH_DEFAULT=10   # <-- 已按您的要求修改为 10
DEVICE_DEFAULT=0
P2CLOSE_DEFAULT=30

# ---- 解析命令行参数 ----
# 如果命令行提供了参数，则使用该参数，否则使用默认值
DATA_PATH=${1:-$DATA_DEFAULT}
EPOCHS=${2:-$EPOCHS_DEFAULT}
BATCH=${3:-$BATCH_DEFAULT}
IMGSZ=${4:-$IMGSZ_DEFAULT}
DEVICE=${5:-$DEVICE_DEFAULT}
P2CLOSE=${6:-$P2CLOSE_DEFAULT}

echo "[配置] 数据集=$DATA_PATH, 轮数=$EPOCHS, 批大小=$BATCH, 图像尺寸=$IMGSZ, 设备=$DEVICE, P2延迟=$P2CLOSE"

# ---- 基础环境检查 ----
if [ ! -f "$DATA_PATH" ]; then
  echo "错误：数据集配置文件不存在: $DATA_PATH"
  exit 1
fi

if ! command -v python &>/dev/null; then
  echo "错误：找不到 Python 解释器"
  exit 1
fi

echo "正在检查 Python 依赖库..."
# 使用 Here Document 的方式在 shell 中执行一小段 Python 代码进行检查
python - <<'PY'
try:
    import torch, ultralytics
    print("依赖库检查通过: torch & ultralytics 已安装")
except ImportError as e:
    print(f"错误：缺少依赖库 -> {e}")
    raise SystemExit("请先通过 pip install -r requirements.txt 安装依赖")
PY

# ---- 启动顺序训练（E1→E6 的调度由 train.py 内部完成）----
echo ">>> 即将启动消融实验 E1 -> E6 ..."
python /workspace/yolo/train_all.py \
  --data "$DATA_PATH" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --close_p2_until "$P2CLOSE"

echo "✅ 全部消融实验（E1-E6）已完成。结果目录：runs/ablation/"