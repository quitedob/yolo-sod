"""
/workspace/yolo/train_visdrone_smallobj.py  # 文件路径：训练入口脚本（含中文注释，每行注释简洁）
功能：一键准备VisDrone数据，分两阶段训练YOLOv12-MambaFusion（先冻骨干训练头部，再全量微调），并保持每次运行唯一名称
特性：
- 自动检查/下载VisDrone；
- 阶段1：冻结骨干若干层，仅训练Neck+Head；
- 阶段2：从阶段1权重恢复，解冻全网继续训练；
- 支持分阶段学习率、warmup与梯度裁剪；
"""

import argparse  # 引入argparse解析命令行参数
from pathlib import Path  # 引入Path用于文件路径操作
import os  # 引入os用于环境与路径处理
import sys  # 引入sys用于动态路径添加
import subprocess  # 引入subprocess用于调用yolo内置下载器

import torch  # 引入PyTorch库
from ultralytics import YOLO  # 引入Ultralytics YOLO训练入口
from ultralytics.utils import LOGGER, yaml_load  # 引入日志与YAML加载工具


def ensure_visdrone(data_yaml: Path) -> None:  # 确保VisDrone数据集存在
    cfg = yaml_load(data_yaml)  # 读取数据集YAML
    root = (Path(__file__).parent / cfg["path"]).resolve()  # 计算数据集根路径
    train_dir = root / cfg["train"]  # 训练图像目录
    val_dir = root / cfg["val"]  # 验证图像目录
    if train_dir.exists() and val_dir.exists():  # 若目录存在
        LOGGER.info(f"✓ VisDrone 已就绪: {root}")  # 打印已就绪
        return  # 直接返回

    LOGGER.warning("未检测到VisDrone数据，将尝试自动下载与转换...")  # 提示下载
    safe_yaml = data_yaml.as_posix()  # 将路径标准化为POSIX风格，避免反斜杠
    code_lines = [  # 按行构建代码，调用YAML中的download片段
        "from pathlib import Path",
        "from ultralytics.utils import yaml_load",
        f"y = yaml_load({safe_yaml!r})",
        "yaml = y",
        "exec(y.get('download', ''), globals(), locals())",
    ]
    code = "\n".join(code_lines)  # 合并为单个可执行字符串
    subprocess.run([sys.executable, "-c", code], check=True)  # 运行下载与转换


def parse_args() -> argparse.Namespace:  # 解析命令行参数
    p = argparse.ArgumentParser()  # 创建解析器
    p.add_argument("--model", type=str,  # 模型配置路径
                   default="ultralytics/cfg/models/new/yolov12-mambafusion-smallobj-640.yaml",
                   help="模型配置文件路径")
    p.add_argument("--data", type=str,  # 数据集配置路径
                   default="ultralytics/cfg/datasets/VisDrone.yaml",
                   help="数据集配置文件路径")
    p.add_argument("--hyp", type=str,  # 超参数配置路径
                   default="ultralytics/cfg/models/new/hyp_smallobj.yaml",
                   help="超参数配置文件路径")
    p.add_argument("--epochs", type=int, default=300, help="总训练轮数")  # 总轮数
    p.add_argument("--stage1_epochs", type=int, default=40, help="阶段1轮数：冻结骨干仅训头部")  # 阶段1轮数
    p.add_argument("--stage2_epochs", type=int, default=0, help="阶段2轮数：留空或0则自动=总-阶段1")  # 阶段2轮数
    p.add_argument("--freeze_backbone", type=int, default=14, help="阶段1冻结前N层（按模型层序号）")  # 冻结层数
    p.add_argument("--batch", type=int, default=16, help="批大小")  # 批大小
    p.add_argument("--imgsz", type=int, default=640, help="图像尺寸")  # 图像尺寸
    p.add_argument("--device", type=str, default="0", help="设备")  # 设备
    p.add_argument("--project", type=str, default="runs/train", help="输出根目录")  # 输出目录
    p.add_argument("--name", type=str, default="yolov12_mf_smallobj_visdrone640", help="实验名(基名)")  # 名称基
    p.add_argument("--exist_ok", action="store_true", help="覆盖已存在目录")  # 覆盖标志
    p.add_argument("--workers", type=int, default=8, help="Dataloader线程数")  # 线程数
    p.add_argument("--amp", action="store_true", default=True, help="开启AMP")  # AMP
    p.add_argument("--resume", type=str, default="", help="从权重恢复(留空自动按阶段恢复)")  # 恢复路径
    p.add_argument("--lr_s1", type=float, default=0.005, help="阶段1初始学习率")  # 阶段1学习率
    p.add_argument("--lr_s2", type=float, default=0.003, help="阶段2初始学习率")  # 阶段2学习率
    p.add_argument("--warmup_epochs", type=float, default=3.0, help="warmup轮数")  # warmup
    p.add_argument("--clip", type=float, default=1.0, help="梯度裁剪阈值(0禁用)")  # 裁剪
    return p.parse_args()  # 返回解析结果


def setup_env() -> None:  # 基础环境设置
    if torch.cuda.is_available():  # 若CUDA可用
        torch.backends.cudnn.benchmark = True  # 开启benchmark
        LOGGER.info(f"CUDA: {torch.cuda.get_device_name(0)}")  # 打印设备名
    else:  # 否则
        LOGGER.warning("CUDA不可用，使用CPU训练")  # 提示CPU
    torch.manual_seed(42)  # CPU种子
    if torch.cuda.is_available():  # CUDA可用时
        torch.cuda.manual_seed(42)  # CUDA种子


def main() -> None:  # 主流程
    args = parse_args()  # 解析参数
    setup_env()  # 设置环境

    data_yaml = Path(args.data)  # 数据集YAML路径
    ensure_visdrone(data_yaml)  # 确保数据就绪

    # 计算阶段轮数
    if args.stage2_epochs <= 0:  # 若未显式给出阶段2轮数
        args.stage2_epochs = max(0, args.epochs - args.stage1_epochs)  # 自动补齐

    # 创建模型
    LOGGER.info(f"创建模型: {args.model}")  # 打印模型YAML
    model = YOLO(args.model)  # 构造YOLO

    # 通用训练参数
# /workspace/yolo/train_visdrone_smallobj.py
# 替换 base_args 定义（删除 "clip": args.clip）
    base_args = {  # 打包通用参数  # 中文注释
        "data": str(data_yaml),   # 数据配置  # 中文注释
        "batch": args.batch,      # 批大小  # 中文注释
        "imgsz": args.imgsz,      # 图像尺寸  # 中文注释
        "device": args.device,    # 设备  # 中文注释
        "optimizer": "AdamW",     # 优化器  # 中文注释
        "project": args.project,  # 输出根目录  # 中文注释
        "exist_ok": args.exist_ok,# 覆盖开关  # 中文注释
        "val": True,              # 开启验证  # 中文注释
        "save_period": 50,        # 保存间隔  # 中文注释
        "workers": args.workers,  # 线程数  # 中文注释
        "amp": args.amp,          # AMP  # 中文注释
        "plots": True,            # 可视化  # 中文注释
        "verbose": True,          # 详细日志  # 中文注释
        "cfg": args.hyp,          # HYP配置（确保只含版本支持的键）  # 中文注释
        "warmup_epochs": args.warmup_epochs,  # warmup  # 中文注释
        # "clip": args.clip,       # 删除：当前版本不支持该键  # 中文注释
    }
    # 阶段1：冻结骨干，仅训练头部
    name_s1 = f"{args.name}_s1"  # 阶段1实验名
    s1_args = base_args | {  # 合并参数
        "epochs": args.stage1_epochs,  # 阶段1轮数
        "name": name_s1,  # 阶段1名称
        "freeze": args.freeze_backbone,  # 冻结前N层
        "lr0": args.lr_s1,  # 阶段1学习率
    }
    LOGGER.info(f"阶段1开始：冻结前 {args.freeze_backbone} 层，epochs={args.stage1_epochs}, lr0={args.lr_s1}")  # 日志
    results_s1 = model.train(**s1_args)  # 执行阶段1训练

    # 阶段2：全量微调，从阶段1权重继续
    name_s2 = f"{args.name}_s2"  # 阶段2实验名
    ckpt_s1 = Path(args.project) / name_s1 / "weights" / "last.pt"  # 阶段1权重路径
    resume_path = args.resume or ckpt_s1.as_posix()  # 优先使用用户传入resume
    s2_args = base_args | {  # 合并参数
        "epochs": args.stage2_epochs,  # 阶段2轮数
        "name": name_s2,  # 阶段2名称
        "freeze": None,  # 解冻全网
        "lr0": args.lr_s2,  # 阶段2学习率
        "resume": resume_path,  # 从阶段1继续
    }
    LOGGER.info(f"阶段2开始：全量微调，epochs={args.stage2_epochs}, lr0={args.lr_s2}, resume={resume_path}")  # 日志
    results_s2 = model.train(**s2_args)  # 执行阶段2训练

    # 训练完成日志
    LOGGER.info("两阶段训练完成！")  # 完成提示
    if hasattr(results_s2, "best_fitness"):  # 若包含最佳指标
        LOGGER.info(f"最佳mAP: {results_s2.best_fitness:.4f}")  # 打印最佳mAP


if __name__ == "__main__":  # 入口判断
    main()  # 执行主函数

