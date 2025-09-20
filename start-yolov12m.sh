#!/bin/bash
# 文件：/workspace/yolo/start.sh
# 作用：YOLOv12-SOD-Fusion-v5 训练启动脚本（支持选择 n/s/m/l/x 变体并自动选取对应 YAML；最小侵入，保证可跑）

# ---- 安全选项（使脚本在出错时立即退出） ----
set -Eeuo pipefail

# ---------- 标题 ----------
echo "=============================================="
echo "  YOLOv12-SOD-Fusion-v5 增强版训练系统"
echo "=============================================="
echo "集成模块："
echo "  ★ SE_Block / CBAM_Block / CA_Block / A2_Attn"
echo "  ★ SwinBlock / DETR-Aux / BoundaryAwareLoss / DetectStable"
echo "  ★ MambaBlock（自动回退到GLU门控卷积）"
echo "=============================================="

# ---------- 默认参数（保持与你原脚本一致） ----------
DEFAULT_DATA="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml"   # 数据集 YAML（默认）
DEFAULT_EPOCHS=500          # 训练轮数默认
DEFAULT_IMGSZ=640           # 输入尺寸默认
DEFAULT_BATCH=10           # 批大小默认
DEFAULT_DEVICE=0            # 设备默认
DEFAULT_CLOSE_P2_UNTIL=30   # P2 延迟激活默认
DEFAULT_VARIANT="x"         # 新增：模型变体（n/s/m/l/x），默认 m -> 12m

# ---------- 读取命令行参数（支持第7个参数指定变体） ----------
# 参数顺序：DATA EPOCHS BATCH IMG_SIZE DEVICE CLOSE_P2_UNTIL VARIANT
DATA_PATH=${1:-$DEFAULT_DATA}
EPOCHS=${2:-$DEFAULT_EPOCHS}
BATCH_SIZE=${3:-$DEFAULT_BATCH}
IMG_SIZE=${4:-$DEFAULT_IMGSZ}
DEVICE=${5:-$DEFAULT_DEVICE}
CLOSE_P2_UNTIL=${6:-$DEFAULT_CLOSE_P2_UNTIL}
MODEL_VARIANT=${7:-$DEFAULT_VARIANT}   # e.g. n / s / m / l / x

# ---------- 打印训练配置 ----------
echo "训练配置参数："
echo "  数据集配置: $DATA_PATH"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  图像尺寸: $IMG_SIZE"
echo "  训练设备: $DEVICE"
echo "  P2关闭轮数: $CLOSE_P2_UNTIL"
echo "  模型变体: $MODEL_VARIANT  (n/s/m/l/x)"
echo "=============================================="

# ---------- 基础检查 ----------
if [ ! -f "$DATA_PATH" ]; then
  echo "错误：数据集配置文件未找到: $DATA_PATH"
  echo "请修正 DEFAULT_DATA 或传入正确路径"
  exit 1
fi

if ! command -v python &> /dev/null; then
  echo "错误：未找到 Python 解释器"
  exit 1
fi

echo "检查依赖库..."
python -c "import importlib, sys; importlib.import_module('torch'); importlib.import_module('ultralytics')" 2>/dev/null || {
  echo "错误：缺少必要的Python库（torch, ultralytics），请先：pip install -r requirements.txt"
  exit 1
}
echo "依赖检查通过，开始选择模型配置..."
echo "=============================================="

# ---------- 自动定位最合适的 YOLOv12 变体 YAML ----------
# 设计：在常见模型目录下按 pattern 搜索第一个匹配的 YAML（大小写不敏感），找不到则回退到 simple 配置（保证可跑）
CANDIDATE_ROOTS=(
  "/workspace/yolo/ultralytics/cfg/models/new"
  "/workspace/yolo/ultralytics/cfg/models/v12"
  "/workspace/yolo/ultralytics/cfg/models"
  "/workspace/yolo/ultralytics/cfg/models/11"
)

# 备选 pattern（按优先级）
PATTERNS=(
  "yolov12-%VAR%*.yaml"
  "yolov12%VAR%*.yaml"
  "*yolov12*%VAR%*.yaml"
  "*yolov12*%VAR%*v5*.yaml"
  "*yolov12*%VAR%*sod*.yaml"
)

# 你现有的 simple 配置作为回退（保证不会因为找不到 YAML 而失败）：
FALLBACK_CFG="/workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-v5-simple.yaml"

find_cfg() {
  local variant="$1"
  local found=""
  for pat in "${PATTERNS[@]}"; do
    # 替换占位符
    local glob="${pat//%VAR%/$variant}"
    for root in "${CANDIDATE_ROOTS[@]}"; do
      if [ -d "$root" ]; then
        # 使用 find + -iname 以忽略大小写
        while IFS= read -r -d '' f; do
          found="$f"
          echo "$found"
          return 0
        done < <(find "$root" -type f -iname "$glob" -print0 2>/dev/null || true)
      fi
    done
  done
  return 1
}

CFG_PATH=""
if CFG_PATH="$(find_cfg "$MODEL_VARIANT")"; then
  echo "已为变体 [$MODEL_VARIANT] 选择模型配置：$CFG_PATH"
else
  echo "未找到匹配 [$MODEL_VARIANT] 的 YOLOv12 配置，回退到：$FALLBACK_CFG"
  CFG_PATH="$FALLBACK_CFG"
fi

# 最后再做一次存在性检查（防止回退路径也不存在）
if [ ! -f "$CFG_PATH" ]; then
  echo "错误：未找到可用的模型配置文件 (尝试路径: $CFG_PATH)"
  echo "请检查模型配置文件是否存在，或将正确的 YAML 放置在 /workspace/yolo/ultralytics/cfg/models/ 目录下"
  exit 1
fi

# ---------- 启动训练（沿用你原先的分段训练脚本） ----------
echo "开始训练，使用模型配置：$CFG_PATH"
python /workspace/yolo/train_yolov12_staged.py \
  --cfg "$CFG_PATH" \
  --data "$DATA_PATH" \
  --epochs $EPOCHS \
  --imgsz $IMG_SIZE \
  --batch $BATCH_SIZE \
  --device $DEVICE \
  --workers 8 \
  --close_p2_until $CLOSE_P2_UNTIL \
  --project runs_yolov12_staged \
  --name yolov12_sod_fusion_v5_exp

# ---------- 结果检查 ----------
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
