#!/bin/bash
# /workspace/yolo/start.sh  # 脚本文件路径
# 启动YOLOv12 VisDrone小目标训练脚本（更稳的抗过拟合设置）
# 使用 GPU 0

python /workspace/yolo/train_visdrone_smallobj.py \
  --model /workspace/yolo/ultralytics/cfg/models/new/yolov12-mambafusion-smallobj-640.yaml \
  --data  /workspace/yolo/ultralytics/cfg/datasets/VisDrone.yaml \
  --hyp   /workspace/yolo/ultralytics/cfg/models/new/hyp_smallobj_stable.yaml \
  --imgsz 640 \
  --epochs 300 \
  --batch -1 \
  --device 0 \
  --project runs/train \
  --name yolov12_mf_smallobj_visdrone640 \
  --exist_ok
