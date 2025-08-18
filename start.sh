# /workspace/yolo/start.sh
#!/bin/bash
# 作用：统一“模型YAML 与 任务类型”，避免 backbone KeyError；并加载外部 hyp 超参  # 中文注释

# ===== 默认参数（可通过命令行覆盖） =====  # 中文注释
DATA=${1:-/workspace/yolo/ultralytics/cfg/datasets/VisDrone.yaml}   # 数据集YAML  # 中文注释
EPOCHS=${2:-600}                                                    # 训练轮数     # 中文注释
BATCH=${3:-16}                                                      # 批大小       # 中文注释
SIZE=${4:-640}                                                      # 输入尺寸     # 中文注释
DEVICE=${5:-0}                                                      # 设备ID       # 中文注释
CLOSE_P2=${6:-30}                                                   # 前N轮关闭P2  # 中文注释
HYP=${7:-/workspace/yolo/ultralytics/cfg/new/hyp_500.yaml}          # 超参文件     # 中文注释

# ===== 选择“带分割头”的全量模型YAML（含 DetectStable + Segment）=====  # 中文注释
MODEL=/workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-v5-all.yaml  # 中文注释

echo "=============================================="
echo "  YOLO-SOD-Fusion 训练"
echo "  data:  $DATA"
echo "  model: $MODEL"
echo "  epochs:$EPOCHS  batch:$BATCH  imgsz:$SIZE  device:$DEVICE"
echo "  close P2 until: $CLOSE_P2   hyp: $HYP"
echo "=============================================="

# 依赖自检（torch/ultralytics）
python - << 'PY' 2>/dev/null
import torch, ultralytics
print("PyTorch:", torch.__version__, "Ultralytics:", ultralytics.__version__)
PY
if [ $? -ne 0 ]; then
  echo "[ERR] 缺少依赖，请先: pip install -r /workspace/yolo/requirements.txt"
  exit 1
fi

# 启动训练（保留你常用的项目命名等）
python /workspace/yolo/train.py \
  --cfg "$MODEL" \
  --data "$DATA" \
  --epochs $EPOCHS \
  --imgsz $SIZE \
  --batch $BATCH \
  --device $DEVICE \
  --hyp "$HYP" \
  --close_p2_until $CLOSE_P2 \
  --use_boundary_loss \
  --use_detr_aux \
  --project runs_fusion \
  --name yolo_sod_fusion_exp

# 结果检查
if [ $? -eq 0 ]; then
  echo "=============================================="
  echo "训练完成，权重/日志在 runs_fusion/yolo_sod_fusion_exp/"
  echo "=============================================="
else
  echo "=============================================="
  echo "训练失败，请查看日志"
  echo "=============================================="
  exit 1
fi
