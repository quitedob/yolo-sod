# YOLO-SOD-Fusion 增强版使用说明

## 项目概述

本项目实现了一个集成多种先进模块的检测与分割模型，基于Ultralytics YOLO框架，融合了以下先进技术：

- **MambaBlock**: 长序列状态空间建模，线性复杂度处理高分辨率特征
- **SwinBlock**: 窗口注意力机制，局部-全局特征融合  
- **DETR-Aux**: 稀疏查询匹配辅助头，提升复杂场景召回率
- **BoundaryAwareLoss**: 边界感知损失，改善分割边界质量
- **MultiObjectTracker**: 时序后处理，集成卡尔曼滤波+匈牙利匹配+LSTM

## 架构设计图

```mermaid
flowchart TD
  A[数据/标签<br/>COCO/自定义+SOD mask] --> B[Backbone: CSP+MambaBlock<br/>长序列建模]
  B --> C[Neck: FPN/PAN+SwinBlock<br/>局部-全局注意力融合]
  C --> D[Head: DetectStable×4 尺度<br/>P2~P5]
  C --> E[Segment 原型头<br/>SOD 掩码预测]
  C --> F[DETR-Aux 辅助头<br/>100查询/集合匹配]
  E --> G[边界感知损失<br/>Sobel Edge+BCE(+IoU)]
  D --> H[推理检测框]
  E --> H
  H --> I[时序后处理 KF+匈牙利+LSTM<br/>稳定轨迹/ID保持]
  F --> |蒸馏/辅助监督| D
```

## 文件结构

```
/workspace/yolo/
├── ultralytics/
│   ├── nn/
│   │   └── modules/
│   │       ├── blocks_mamba.py          # Mamba状态空间模块
│   │       ├── blocks_transformer.py    # Swin Transformer模块
│   │       ├── heads_detr_aux.py       # DETR辅助头
│   │       ├── loss_boundary.py        # 边界感知损失
│   │       ├── tracker_kf_lstm.py      # 多目标跟踪器
│   │       └── detect_stable.py        # P2启停检测头
│   └── cfg/
│       └── models/
│           └── new/
│               └── yolov12-sod-fusion-v5-all.yaml  # 模型配置
├── train.py                            # 增强训练脚本  
├── start.sh                            # 训练启动脚本
├── inference_video_tracking.py         # 视频推理跟踪脚本
├── requirements.txt                     # 依赖库清单
└── README_FUSION_ENHANCED.md           # 本使用说明
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖库
pip install -r requirements.txt

# 确保start.sh有执行权限
chmod +x start.sh
```

### 2. 训练模型

#### 方式一：使用start.sh（推荐）

```bash
# 基本用法（需要先修改start.sh中的数据集路径）
./start.sh

# 自定义参数
./start.sh your_dataset.yaml 500 16 640 0 30
#         数据集配置    轮数 批次 尺寸 设备 P2关闭轮数
```

#### 方式二：直接调用Python脚本

```bash
python train.py \
    --data your_dataset.yaml \
    --epochs 500 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --use_boundary_loss \
    --close_p2_until 30 \
    --use_detr_aux
```

### 3. 视频推理与跟踪

```bash
python inference_video_tracking.py \
    --model runs_fusion/yolo_sod_fusion_exp/weights/best.pt \
    --video input_video.mp4 \
    --output output_video.mp4 \
    --confidence 0.5 \
    --iou_threshold 0.3
```

## 核心功能详解

### 1. Mamba长序列建模 (MambaBlock)

**动机**: 高分辨率特征图展平成序列后进行建模，弥补纯注意力在超长序列下的计算瓶颈

**特点**:
- 线性时间复杂度，处理长序列高效
- 选择性状态空间建模，保留重要信息
- 自动回退到GLU门控卷积（无mamba-ssm时）

**位置**: 骨干网络P5层，参数可调节序列缩减因子

### 2. Swin窗口注意力 (SwinBlock)  

**动机**: 局部窗口自注意力提供线性复杂度与多尺度特征表达，显著提升密集预测性能

**特点**:
- Shifted Windows策略建立局部-全局连接
- 多头自注意力增强特征表达
- 结合深度卷积保留局部特征

**位置**: 骨干网络P5层和颈部P2层，利于小目标检测

### 3. P2尺度启停控制 (DetectStable)

**动机**: 前期关闭P2尺度可以稳定训练，避免小目标噪声干扰

**策略**:
- 默认前30轮关闭P2，只使用P3-P5
- 第30轮后激活P2，提升小目标检测能力  
- 通过回调机制动态控制，不改动核心代码

### 4. 边界感知损失 (BoundaryAwareLoss)

**动机**: SOD研究表明边界质量对视觉效果至关重要，可减少锯齿和粘连

**实现**:
- Sobel算子提取边缘图
- 边缘BCE损失 + 基础BCE损失 + 可选IoU损失
- 权重可调，支持网格搜索优化

### 5. DETR辅助头 (DETRAuxHead)

**动机**: 稀疏查询匹配机制提升复杂场景召回率，辅助主检测头训练

**架构**:
- 100个可学习查询
- Transformer编码器处理特征  
- 输出分类logits和归一化边界框
- 支持匈牙利匹配损失（可扩展）

### 6. 多目标跟踪 (MultiObjectTracker)

**动机**: 视频场景中稳定目标ID，降低检测闪烁，提升用户体验

**算法**:
- 卡尔曼滤波预测目标状态
- 匈牙利算法最优匹配（回退贪心匹配）
- LSTM学习运动模式，平滑短期遮挡
- 自适应轨迹生命周期管理

## 训练参数说明

### 基础参数
- `--data`: 数据集配置文件路径 **(必需)**
- `--epochs`: 训练轮数 (默认: 500)
- `--batch`: 批次大小 (默认: 16) 
- `--imgsz`: 输入图像尺寸 (默认: 640)
- `--device`: 训练设备 (默认: '0')

### 增强功能开关
- `--use_boundary_loss`: 启用边界感知损失
- `--close_p2_until`: P2关闭轮数 (默认: 30)
- `--use_detr_aux`: 启用DETR辅助头

### 边界损失参数
- `--edge_weight`: 边缘损失权重 (默认: 1.0)
- `--bce_weight`: BCE损失权重 (默认: 1.0)  
- `--iou_weight`: IoU损失权重 (默认: 0.0)
- `--boundary_loss_weight`: 边界损失总权重 (默认: 0.2)

### 学习率参数
- `--lr0`: 初始学习率 (默认: 0.01)
- `--lrf`: 最终学习率比例 (默认: 0.01)
- `--optimizer`: 优化器类型 (默认: 'auto')

## 性能优化建议

### 1. 训练策略
- **前30轮关闭P2**: 稳定训练初期，避免小目标噪声
- **余弦退火学习率**: 配合warmup，提升收敛质量
- **边界损失权重**: 建议0.1-0.3，可通过验证集调优

### 2. 硬件配置
- **推荐GPU**: RTX 3080/4080以上，显存≥12GB
- **批次大小**: 根据显存调整，推荐16-32
- **数据加载**: 多线程加载，workers=8

### 3. 数据处理
- **图像尺寸**: 640x640平衡精度与速度
- **数据增强**: Mosaic + CopyPaste，配合边界损失效果更佳
- **标注质量**: SOD分支需要精确的掩码标注

## 故障排除

### 常见问题

1. **ImportError: No module named 'mamba_ssm'**
   - 解决：MambaBlock会自动回退到GLU卷积，不影响训练
   - 可选：`pip install mamba-ssm`（Linux推荐）

2. **FilterPy未安装，卡尔曼滤波功能禁用**
   - 解决：`pip install filterpy`
   - 或使用纯运动模型（功能简化但仍可用）

3. **P2启停回调执行失败**
   - 检查：DetectStable模块是否正确加载
   - 确认：YAML配置中使用DetectStable而非Detect

4. **边界损失计算异常** 
   - 确认：数据集包含masks标注
   - 检查：预测掩码维度是否正确

### 性能调优

1. **训练不稳定**
   - 减小学习率：`--lr0 0.005`
   - 延长P2关闭时间：`--close_p2_until 50`
   - 降低边界损失权重：`--boundary_loss_weight 0.1`

2. **显存不足**
   - 减小批次：`--batch 8`
   - 降低输入尺寸：`--imgsz 512`
   - 关闭DETR辅助头：移除`--use_detr_aux`

3. **小目标检测差**
   - 延后P2激活：`--close_p2_until 50`  
   - 使用更大输入：`--imgsz 800`
   - 调整anchor-free参数

## 模型部署

### 1. 导出ONNX模型

```bash
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs_fusion/yolo_sod_fusion_exp/weights/best.pt')

# 导出ONNX格式
model.export(format='onnx', dynamic=True, simplify=True)
```

### 2. TensorRT加速（可选）

```bash
# 导出TensorRT引擎  
model.export(format='engine', device=0)

# TensorRT推理
model = YOLO('runs_fusion/yolo_sod_fusion_exp/weights/best.engine')
results = model('image.jpg')
```

## 扩展开发

### 1. 添加新的注意力机制

在`ultralytics/nn/modules/`下创建新模块，参考`blocks_transformer.py`结构

### 2. 自定义损失函数

扩展`loss_boundary.py`，实现新的边界约束方法

### 3. 改进跟踪算法

在`tracker_kf_lstm.py`基础上集成ReID特征，提升遮挡场景表现

## 技术支持

如遇到问题，请检查：
1. 依赖库版本兼容性
2. CUDA环境配置  
3. 数据集格式正确性
4. 模型配置文件路径

本项目基于Ultralytics YOLO v8.3+开发，建议使用最新版本以获得最佳兼容性。
