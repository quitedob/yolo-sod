# YOLO-SOD: A Synergistic Paradigm of Heterogeneous Attention and Curriculum Learning for Small Object Detection

[![arXiv](https://img.shields.io/badge/arXiv-2408.12345-b31b1b.svg)](https://arxiv.org/abs/2408.12345)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **"A Synergistic Paradigm of Heterogeneous Attention and Curriculum Learning for Small Object Detection"** published in the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

## Authors

- Zhaoyang Zhang¹
- Qikun Shi¹* 
- Xiao Wang²
- Jiayi Lai³

¹ School of Computer Science and Technology, Wuhan University of Science and Technology, Wuhan 430081, China  
² Hubei Province Key Laboratory of Intelligent Information Processing and Real-time Industrial System, Wuhan University of Science and Technology, Wuhan 430081, China  
³ McGill University, 845 Rue Sherbrooke O, Montréal, QC H3A 0G4 Canada

*Corresponding author

## Abstract

Small Object Detection (SOD) remains a fundamental scientific challenge in computer vision, particularly in UAV imagery applications where objects occupy minimal pixels and exhibit ambiguous visual features. Despite significant progress in general object detection, SOD performance typically lags behind by 15-20% absolute points on challenging benchmarks, where state-of-the-art methods struggle to surpass 26% mAP@0.5:0.95.

This paper introduces a novel architecture-training co-design paradigm that fundamentally addresses these scientific challenges through two complementary innovations: (1) A Multi-Attention Fusion Neck (MAFN) that strategically deploys heterogeneous attention mechanisms across feature pyramid levels, enabling more effective feature extraction and representation learning for small objects; (2) A Staged Training Protocol using curriculum learning principles to systematically modulate training dynamics, ensuring stable convergence and optimal performance.

Comprehensive evaluation on the VisDrone benchmark demonstrates our model achieves **27.5% mAP@0.5:0.95** and **46.1% mAP@0.5**, surpassing recent YOLO variants and establishing a new paradigm for addressing the scientific challenges in SOD.

## Key Features

- **Multi-Attention Fusion Neck (MAFN)**: Heterogeneously deploys SE, CBAM, CA, A2, and Swin Transformer attention mechanisms across different feature pyramid levels
- **Staged Training Protocol**: Four-stage curriculum learning approach for stable optimization of complex architectures
- **State-of-the-Art Performance**: Achieves 27.5% mAP@0.5:0.95 on VisDrone2019, surpassing YOLOv8-L, YOLOv11-M, and RT-DETR baselines
- **Efficient Implementation**: 13.56M parameters, 41.5 GFLOPs, suitable for real-time applications
- **Comprehensive Evaluation**: Tested on VisDrone2019 and UAVVaste datasets

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+
- NVIDIA GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/quitedob/yolo-sod.git
cd yolo-sod
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the ultralytics package:
```bash
pip install -e .
```

## Usage

### Training

Train the model using the staged training protocol:

```bash
# Stage 1: Foundation training
python train_yolov12_staged.py --stage 1 --data visdrone.yaml --weights yolov12n.pt

# Stage 2: Fusion learning
python train_yolov12_staged.py --stage 2 --data visdrone.yaml --weights runs/stage1/weights/last.pt

# Stage 3: Small object refinement (delayed P2 activation)
python train_yolov12_staged.py --stage 3 --data visdrone.yaml --weights runs/stage2/weights/last.pt

# Stage 4: Final polishing
python train_yolov12_staged.py --stage 4 --data visdrone.yaml --weights runs/stage3/weights/last.pt
```

Or use the automated training script:
```bash
bash start.sh
```

### Inference

Run inference on images or videos:

```bash
python app.py --source path/to/image.jpg --weights runs/final/weights/best.pt
```

### Evaluation

Evaluate on VisDrone validation set:

```bash
python val.py --data visdrone.yaml --weights runs/final/weights/best.pt
```

## Results

### VisDrone2019 Validation Set

| Model | Params (M) | GFLOPs | mAP@0.5:0.95 | mAP@0.5 |
|-------|------------|--------|---------------|---------|
| YOLOv8-M | 25.9 | 78.9 | 24.6 | 40.7 |
| YOLOv8-L | 43.7 | 165.2 | 26.1 | 42.7 |
| YOLOv11-M | 20.0 | 67.7 | 25.9 | 43.1 |
| RT-DETR-R18 | 20.0 | 60.0 | 26.7 | 44.6 |
| HIC-YOLOv5 | 9.4 | 31.2 | 26.0 | 44.3 |
| **Ours** | **13.56** | **41.5** | **27.5** | **46.1** |

### UAVVaste Dataset

| Model | Params (M) | GFLOPs | AP | AP@50 |
|-------|------------|--------|----|-------|
| YOLOv11-S | 9.4 | 21.3 | 27.8 | 63.0 |
| HIC-YOLOv5 | 9.4 | 31.2 | 30.5 | 65.1 |
| RT-DETR-R18 | 20.0 | 57.3 | 36.3 | 72.6 |
| **Ours** | **13.56** | **41.4** | **46.0** | **79.3** |

## Ablation Study

Our ablation experiments show the progressive improvements:

- Baseline: 23.8% mAP@0.5:0.95, 39.6% mAP@0.5
- +P2 Head: 27.1% mAP@0.5:0.95, 44.5% mAP@0.5 (+4.9)
- +SE: 26.1% mAP@0.5:0.95, 43.2% mAP@0.5 (-1.3)
- +CBAM: 26.9% mAP@0.5:0.95, 44.2% mAP@0.5 (+1.0)
- +Swin: 27.1% mAP@0.5:0.95, 44.5% mAP@0.5 (+0.3)
- +CA+ST: 27.3% mAP@0.5:0.95, 44.8% mAP@0.5 (+0.3)
- **Complete**: **27.5% mAP@0.5:0.95, 46.1% mAP@0.5 (+1.3)**

## Architecture Details

### Multi-Attention Fusion Neck (MAFN)

The MAFN strategically deploys different attention mechanisms at various pyramid levels:

- **P2 (High-resolution)**: SE + A2 for lightweight channel recalibration and local context
- **P3 (Mid-resolution)**: CBAM + CA for coupled channel-spatial attention and position-sensitive localization
- **P4 (Low-resolution)**: Swin Transformer for long-range dependencies and semantic abstraction
- **P5 (Deep features)**: Enhanced fusion with heterogeneous attention

### Staged Training Protocol

Four-stage curriculum learning:
1. **Foundation Stage**: Stabilize basic representations
2. **Fusion Stage**: Emphasize feature fusion learning
3. **Refinement Stage**: Delay P2 activation for small object focus
4. **Polishing Stage**: Final convergence optimization

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2025synergistic,
  title={A Synergistic Paradigm of Heterogeneous Attention and Curriculum Learning for Small Object Detection},
  author={Zhang, Zhaoyang and Shi, Qikun and Wang, Xiao and Lai, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

## Conference Presentation

This work was presented at the **Second International Conference on Computer Vision, Image Processing and Computational Photography (CVIP 2025)**, held in Hangzhou, China, and published by IEEE.

## Acknowledgments

This work is supported by the National Nature Science Foundation of China (Grant No. 62302351).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please contact:
- Zhaoyang Zhang: dobqop999@gmail.com
