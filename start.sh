#!/bin/bash
# /workspace/yolo/start.sh  # 文件路径：唯一训练入口（分级训练+稳定化），无交互可直接运行  # 中文注释

set -euo pipefail  # 严格模式：遇错即退，未定义变量即错，管道错误冒泡  # 中文注释

PY=${PYTHON:-python}  # 可通过环境变量PYTHON覆盖python解释器  # 中文注释

# 阶段配置：可通过环境变量覆盖，如 STAGE1=20 STAGE2=40  # 中文注释
STAGE1=${STAGE1:-20}  # 阶段1epoch（冻结P2/保守增广）  # 中文注释
STAGE2=${STAGE2:-40}  # 阶段2epoch（开启P2/InterpIoU）  # 中文注释
TOTAL=${TOTAL:-300}    # 总epoch  # 中文注释

MODEL_CFG="/workspace/yolo/ultralytics/cfg/models/new/yolov12-smallobj-stable.yaml"  # 模型YAML路径（稳定版）  # 中文注释
DATA_CFG="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml"                                    # 数据YAML路径  # 中文注释

echo "[start] 使用分级训练：STAGE1=${STAGE1}, STAGE2=${STAGE2}, TOTAL=${TOTAL}"  # 中文注释

# 直接调用分级训练脚本作为唯一入口  # 中文注释
exec "$PY" /workspace/yolo/train_staged_visdrone.py  # 以python进程执行脚本  # 中文注释
