"""
/workspace/yolo/train_visdrone_smallobj.py
功能：一键下载并转换VisDrone数据集，使用小目标优化版YOLOv12-MambaFusion进行训练
特性：
- 自动检查/下载VisDrone并转YOLO标签
- 默认启用小目标友好HYP与新模型YAML
- 自动尝试启用SageAttention（若可用），已在模块内部适配
"""

import argparse  # 引入argparse解析命令行参数
from pathlib import Path  # 引入Path用于文件路径操作
import os  # 引入os用于环境与路径处理
import sys  # 引入sys用于动态路径添加
import subprocess  # 引入subprocess用于调用yolo内置下载器

import torch  # 引入PyTorch库
from ultralytics import YOLO  # 引入Ultralytics YOLO训练入口
from ultralytics.utils import LOGGER, yaml_load  # 引入日志与YAML加载工具


def ensure_visdrone(data_yaml: Path) -> None:
    """确保VisDrone数据集存在，若不存在则调用内置download脚本下载并转换"""
    cfg = yaml_load(data_yaml)  # 读取数据集YAML
    root = (Path(__file__).parent / cfg["path"]).resolve()  # 计算数据集根路径
    train_dir = root / cfg["train"]  # 训练图像目录
    val_dir = root / cfg["val"]  # 验证图像目录
    if train_dir.exists() and val_dir.exists():  # 若目录存在
        LOGGER.info(f"✓ VisDrone 已就绪: {root}")  # 打印已就绪
        return  # 直接返回

    LOGGER.warning("未检测到VisDrone数据，将尝试自动下载与转换...")  # 提示下载
    # 通过执行 Python 片段来使用 YAML 内的 download 脚本（避免 f-string 表达式中出现反斜杠）
    safe_yaml = data_yaml.as_posix()  # 将路径标准化为POSIX风格，避免反斜杠
    code_lines = [  # 按行构建可执行代码，避免在花括号中出现转义符
        "from pathlib import Path",
        "from ultralytics.utils import yaml_load",
        f"y = yaml_load({safe_yaml!r})",
        "yaml = y",
        "exec(y.get('download', ''), globals(), locals())",
    ]
    code = "\n".join(code_lines)  # 合并为单个可执行字符串
    subprocess.run([sys.executable, "-c", code], check=True)  # 运行下载与转换


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="ultralytics/cfg/models/new/yolov12-mambafusion-smallobj-640.yaml",
                   help="模型配置文件路径")
    p.add_argument("--data", type=str,
                   default="ultralytics/cfg/datasets/VisDrone.yaml",
                   help="数据集配置文件路径")
    p.add_argument("--hyp", type=str,
                   default="ultralytics/cfg/models/new/hyp_smallobj.yaml",
                   help="超参数配置文件路径")
    p.add_argument("--epochs", type=int, default=300, help="训练轮数")
    p.add_argument("--batch", type=int, default=16, help="批大小")
    p.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    p.add_argument("--device", type=str, default="0", help="设备")
    p.add_argument("--project", type=str, default="runs/train", help="输出根目录")
    p.add_argument("--name", type=str, default="yolov12_mf_smallobj_visdrone640", help="实验名")
    p.add_argument("--exist_ok", action="store_true", help="覆盖已存在目录")
    p.add_argument("--workers", type=int, default=8, help="Dataloader线程数")
    p.add_argument("--amp", action="store_true", default=True, help="开启AMP")
    p.add_argument("--resume", type=str, default="", help="恢复训练权重")
    return p.parse_args()


def setup_env() -> None:
    """基础环境设置：CUDA与随机种子"""
    if torch.cuda.is_available():  # 判断CUDA是否可用
        torch.backends.cudnn.benchmark = True  # 开启benchmark优化
        LOGGER.info(f"CUDA: {torch.cuda.get_device_name(0)}")  # 打印CUDA设备
    else:
        LOGGER.warning("CUDA不可用，使用CPU训练")  # 提示CPU训练
    torch.manual_seed(42)  # 设置CPU随机种子
    if torch.cuda.is_available():  # 若CUDA可用
        torch.cuda.manual_seed(42)  # 设置CUDA随机种子


def main() -> None:
    """主流程：准备数据与配置，创建模型并训练"""
    args = parse_args()  # 解析参数
    setup_env()  # 设置环境

    data_yaml = Path(args.data)  # 数据集YAML路径
    ensure_visdrone(data_yaml)  # 确保数据集准备就绪

    # 创建模型（从配置）
    LOGGER.info(f"创建模型: {args.model}")  # 打印模型配置
    model = YOLO(args.model)  # 构造YOLO模型

    # 训练参数打包
    train_args = {
        "data": str(data_yaml),  # 数据配置
        "epochs": args.epochs,  # 训练轮数
        "batch": args.batch,  # 批大小
        "imgsz": args.imgsz,  # 图像尺寸
        "device": args.device,  # 设备选择
        "optimizer": "AdamW",  # 使用AdamW优化器
        "project": args.project,  # 输出工程目录
        "name": args.name,  # 实验名称
        "exist_ok": args.exist_ok,  # 是否覆盖
        "val": True,  # 开启验证
        "save_period": 50,  # 保存间隔
        "workers": args.workers,  # 数据加载线程
        "amp": args.amp,  # AMP混合精度
        "plots": True,  # 训练过程可视化
        "verbose": True,  # 详细日志
        "cfg": args.hyp,  # 超参数配置（只包含Ultralytics支持字段）
    }

    # 触发训练
    LOGGER.info("开始训练小目标优化模型...")  # 日志提示开始
    results = model.train(**train_args)  # 执行训练
    LOGGER.info("训练完成！")  # 日志提示完成
    if hasattr(results, "best_fitness"):  # 若包含最佳mAP
        LOGGER.info(f"最佳mAP: {results.best_fitness:.4f}")  # 打印最佳mAP


if __name__ == "__main__":  # 入口判断
    main()  # 调用主函数


