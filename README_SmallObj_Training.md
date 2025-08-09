## 小目标优化训练快速开始

- 模型：`ultralytics/cfg/models/new/yolov12-mambafusion-smallobj.yaml`
- 超参：`ultralytics/cfg/models/new/hyp_smallobj.yaml`
- 数据：`ultralytics/cfg/datasets/VisDrone.yaml`

示例命令：

```bash
python /workspace/yolo/train_visdrone_smallobj.py \
  --model ultralytics/cfg/models/new/yolov12-mambafusion-smallobj.yaml \
  --data ultralytics/cfg/datasets/VisDrone.yaml \
  --hyp ultralytics/cfg/models/new/hyp_smallobj.yaml \
  --imgsz 960 --epochs 300 --batch 16 --device 0
```

说明：`OmniKernelFusion` 模块会在检测到 `sageattention` 可用时自动启用注意力分支，无需额外改动。


