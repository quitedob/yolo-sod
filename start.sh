#!/bin/bash
# /workspace/yolo/start_min.sh
# 中文注释：仅传通用参数，兼容老的 train.py；其它超参放到 hyp.yaml 里

DATA=${1:-/workspace/yolo/ultralytics/cfg/datasets/VisDrone.yaml}
EPOCHS=${2:-600}
BATCH=${3:-16}
SIZE=${4:-640}
DEVICE=${5:-0}
CLOSE_P2=${6:-30}
HYP=${7:-/workspace/yolo/ultralytics/cfg/new/hyp_500.yaml}

python /workspace/yolo/train.py \
  --cfg /workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-v5-all.yaml \
  --data "$DATA" \
  --epochs $EPOCHS \
  --imgsz $SIZE \
  --batch $BATCH \
  --device $DEVICE \
  --hyp "$HYP" \
  --close_p2_until $CLOSE_P2 \
  --use_boundary_loss \
  --use_detr_aux
