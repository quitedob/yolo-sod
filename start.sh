#!/bin/bash
# /workspace/yolo/start.sh
# 作用：一键启动 HCP-400 分层训练；全模式统一调用 train_hcp_400.py；自动评估
# 备注：DetectStable 才支持“前N轮关闭P2”，兼容版(纯 Detect)请把 CLOSE_P2_UNTIL=0

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PYTHON=${PYTHON:-python}

# ===== 默认路径，可通过环境变量覆盖 =====  # 中文注释
CFG=${CFG:-/workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-compatible.yaml}
HYP=${HYP:-/workspace/yolo/ultralytics/cfg/models/new/hyp_hcp_400_v2.yaml}
DATA=${DATA:-/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml}

EPOCHS=${EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-16}
IMG_SIZE=${IMG_SIZE:-640}
DEVICE=${DEVICE:-0}

TRAIN_MODE=${TRAIN_MODE:-"hcp400"}   # simple / hcp400 / standard / enhanced
CLOSE_P2_UNTIL=${CLOSE_P2_UNTIL:-0}  # 兼容版(Detect)务必设 0；DetectStable 可用 30

echo "[INFO] ========================================="
echo "[INFO] 模型配置: $CFG"
echo "[INFO] 超参配置: $HYP"
echo "[INFO] 数据配置: $DATA"
echo "[INFO] 模式: $TRAIN_MODE | EPOCHS=$EPOCHS | BATCH=$BATCH_SIZE | IMG=$IMG_SIZE | DEV=$DEVICE"
echo "[INFO] CLOSE_P2_UNTIL=$CLOSE_P2_UNTIL"
echo "[INFO] ========================================="

mkdir -p runs_hcp_400 runs_standard runs_test runs_enhanced

case "$TRAIN_MODE" in
  simple)   LOGDIR="runs_test" ;;
  hcp400)   LOGDIR="runs_hcp_400" ;;
  standard) LOGDIR="runs_standard" ;;
  enhanced) LOGDIR="runs_enhanced" ;;
  *) echo "[ERROR] 未知模式: $TRAIN_MODE"; exit 1 ;;
esac

for f in "$CFG" "$HYP" "$DATA"; do
  [ -f "$f" ] || { echo "[ERROR] 文件不存在: $f"; exit 1; }
done

if [ "$TRAIN_MODE" = "simple" ]; then
  echo "[INFO] 启动简化测试训练(20 epoch)..."
  $PYTHON /workspace/yolo/train_hcp_400.py \
      --cfg "$CFG" --hyp "$HYP" --data "$DATA" \
      --epochs 20 --imgsz "$IMG_SIZE" --batch "$BATCH_SIZE" \
      --device "$DEVICE" --close-p2-until 0 \
      2>&1 | tee -a "$LOGDIR/simple_test.log"

elif [ "$TRAIN_MODE" = "hcp400" ]; then
  echo "[INFO] 启动 HCP-400 训练..."
  $PYTHON /workspace/yolo/train_hcp_400.py \
      --cfg "$CFG" --hyp "$HYP" --data "$DATA" \
      --epochs "$EPOCHS" --imgsz "$IMG_SIZE" --batch "$BATCH_SIZE" \
      --device "$DEVICE" --close-p2-until "$CLOSE_P2_UNTIL" \
      2>&1 | tee -a "$LOGDIR/hcp400_train.log"

elif [ "$TRAIN_MODE" = "standard" ]; then
  echo "[INFO] 启动标准训练(单阶段)..."
  $PYTHON - <<'PYCODE'
from ultralytics import YOLO
import os
cfg=os.environ["CFG"]; hyp=os.environ["HYP"]; data=os.environ["DATA"]
imgsz=int(os.environ.get("IMG_SIZE","640")); batch=int(os.environ.get("BATCH_SIZE","16"))
device=os.environ.get("DEVICE","0"); epochs=int(os.environ.get("EPOCHS","100"))
m=YOLO(cfg)
m.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        project="runs_standard", name="standard", exist_ok=True, pretrained=False)
PYCODE

elif [ "$TRAIN_MODE" = "enhanced" ]; then
  echo "[INFO] 启动增强版训练(同 HCP-400，保留 P2 延迟等设置)..."
  $PYTHON /workspace/yolo/train_hcp_400.py \
      --cfg "$CFG" --hyp "$HYP" --data "$DATA" \
      --epochs "$EPOCHS" --imgsz "$IMG_SIZE" --batch "$BATCH_SIZE" \
      --device "$DEVICE" --close-p2-until "$CLOSE_P2_UNTIL" \
      2>&1 | tee -a "$LOGDIR/enhanced_train.log"
fi

echo "[INFO] 训练完成！日志目录: $LOGDIR"

# 自动评估（若存在最新 last.pt）  # 中文注释
LAST_PT=$(ls -t $LOGDIR/*/weights/last.pt 2>/dev/null | head -n1 || true)
if [ -f "$LAST_PT" ]; then
  echo "[INFO] 发现权重: $LAST_PT，开始评估..."
  $PYTHON - <<PYCODE
from ultralytics import YOLO
import os
pt=os.environ["LAST_PT"]; data=os.environ["DATA"]; device=os.environ.get("DEVICE","0")
YOLO(pt).val(data=data, device=device)
PYCODE
  echo "[INFO] 评估完成！"
else
  echo "[INFO] 未找到 last.pt，跳过评估"
fi
