#!/bin/bash
# /workspace/yolo/start.sh  # 脚本文件路径
# 启动YOLOv12 VisDrone小目标训练脚本
# 使用 GPU 0

python /workspace/yolo/train_visdrone_smallobj.py \
  --model /workspace/yolo/ultralytics/cfg/models/new/yolov12-mambafusion-smallobj-640.yaml \
  --imgsz 640 \
  --device 0 \
  --project runs/train \
  --name yolov12_mf_smallobj_visdrone640
