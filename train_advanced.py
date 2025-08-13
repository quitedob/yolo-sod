# /workspace/yolo/train_advanced.py
# 主要功能简介: 实现分阶段/分组微调训练协议, 配置差分冻结策略, 以稳定训练并提升小目标检测性能  

import os  # 导入os用于环境配置  
import argparse  # 解析命令行参数  
import torch  # 导入PyTorch  
from ultralytics import YOLO  # 导入Ultralytics训练接口  
from ultralytics.utils import yaml_load, DEFAULT_CFG_KEYS  # 导入YAML读取与有效键集合  


# 阶段划分配置  
STAGE1_EPOCHS = 20  # 第一阶段: 仅训练新模块与头部  
STAGE2_EPOCHS = 60  # 第二阶段: 训练整个颈部与头部  
STAGE3_EPOCHS = 220 # 第三阶段: 全局微调  
TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS  # 计算总轮数  

DEVICE = "0" if torch.cuda.is_available() else "cpu"  # 指定设备, 有GPU则用0号卡  
DEFAULT_MODEL_CFG = "/workspace/yolo/ultralytics/cfg/models/new/yolov12-smallobj-advanced.yaml"  # 模型配置默认路径  
DATA_CFG = "/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml"  # 数据集配置路径  
HYP_CFG = "/workspace/yolo/ultralytics/cfg/models/new/hyp_visdrone_advanced.yaml"  # 超参数配置路径  
HYP_OVERRIDES = {k: v for k, v in yaml_load(HYP_CFG).items() if k in DEFAULT_CFG_KEYS}  # 过滤仅保留框架认可键  


def parse_args():  # 解析命令行参数  
    parser = argparse.ArgumentParser()  # 构建解析器  
    parser.add_argument("--cfg", type=str, default=DEFAULT_MODEL_CFG, help="模型YAML配置路径")  # 模型配置  
    parser.add_argument("--use_sageattention2", type=str, default="0", help="是否尝试SageAttention2(1/0)")  # 注意力开关  
    return parser.parse_args()  # 返回解析结果  


def run_stage(model: YOLO, stage_index: int):  # 定义按阶段运行的辅助函数  
    # 根据不同阶段设置冻结策略与增强强度  
    if stage_index == 1:  # 第一阶段训练配置  
        kwargs = dict(data=DATA_CFG, epochs=STAGE1_EPOCHS, batch=8, imgsz=640, optimizer="AdamW", lr0=0.001, freeze=list(range(10)), mosaic=0.0, amp=True, device=DEVICE, project="runs_advanced", name="stage1_warmup_rafb")  
        kwargs.update(HYP_OVERRIDES)  # 注入超参覆盖  
        model.train(**kwargs)  
    elif stage_index == 2:  # 第二阶段训练配置  
        kwargs = dict(data=DATA_CFG, epochs=STAGE2_EPOCHS, batch=8,resume=True, freeze=list(range(10)), mosaic=1.0, amp=True, device=DEVICE, project="runs_advanced", name="stage2_train_neck_head")  
        kwargs.update(HYP_OVERRIDES)  
        model.train(**kwargs)  
    else:  # 第三阶段训练配置  
        kwargs = dict(data=DATA_CFG, epochs=STAGE3_EPOCHS, batch=8,resume=True, freeze=None, lr0=0.0001, device=DEVICE, project="runs_advanced", name="stage3_finetune_all")  
        kwargs.update(HYP_OVERRIDES)  
        model.train(**kwargs)  


if __name__ == "__main__":  # 入口  
    args = parse_args()  # 读取命令行参数  
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEVICE if DEVICE != "cpu" else "")  # 设置可见GPU  
    # 将是否启用SageAttention2写入环境变量, 供模块内部判断一次性尝试  
    os.environ["USE_SAGE_ATTENTION2"] = str(args.use_sageattention2)  # 设置注意力开关  
    model = YOLO(args.cfg)  # 按传入的模型YAML构建模型  

    print("\n" + "=" * 50)  # 打印阶段信息分隔线  
    print(f" HIERARCHICAL TRAINING: STAGE 1 (Epochs 1-{STAGE1_EPOCHS}) ")  # 提示第一阶段  
    print(" Training: New RAFB module and Head. Freezing: Backbone and standard Neck.")  # 阶段描述  
    print("=" * 50 + "\n")  # 结束分隔线  
    run_stage(model, stage_index=1)  # 运行第一阶段  

    print("\n" + "=" * 50)  # 第二阶段分隔线  
    print(f" HIERARCHICAL TRAINING: STAGE 2 (Epochs {STAGE1_EPOCHS+1}-{STAGE1_EPOCHS+STAGE2_EPOCHS}) ")  # 提示第二阶段  
    print(" Training: Full Neck and Head. Freezing: Backbone.")  # 阶段描述  
    print("=" * 50 + "\n")  # 结束分隔线  
    run_stage(model, stage_index=2)  # 运行第二阶段  

    print("\n" + "=" * 50)  # 第三阶段分隔线  
    print(f" HIERARCHICAL TRAINING: STAGE 3 (Epochs {STAGE1_EPOCHS+STAGE2_EPOCHS+1}-{TOTAL_EPOCHS}) ")  # 提示第三阶段  
    print(" Training: End-to-end fine-tuning of the entire model.")  # 阶段描述  
    print("=" * 50 + "\n")  # 结束分隔线  
    run_stage(model, stage_index=3)  # 运行第三阶段  


