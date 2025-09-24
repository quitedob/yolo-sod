# /workspace/yolo/train_all.py
# 作用：通过顺序加载 E1-E6 的专属 YAML 配置文件，来依次运行 v5-simple 模块级消融实验。
# 更新：重构了脚本逻辑，从动态修改模型改为加载独立的 YAML 文件。

import os
import argparse
import random
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import torch.nn as nn # 保留 nn 以便 cb_p2_gate 能找到 Identity, 虽然 apply_module_gates 已移除

# ---------- 1) 自定义模块注册：保证 YAML / 模型解析时自定义算子可用 ----------
# 功能保持不变，确保所有自定义层都能被 YOLO 正确解析
def register_custom_modules():
    """# 功能：把工程里的自定义算子注册到 ultralytics.nn.modules 命名空间，防止解析 YAML 失败"""
    import ultralytics.nn.modules as U
    try:
        from ultralytics.nn.modules.blocks_transformer import SwinBlock; U.SwinBlock = SwinBlock
    except Exception as e: print(f"[WARN] SwinBlock 导入失败: {e}")
    try:
        from ultralytics.nn.modules.a2_attn import A2_Attn; U.A2_Attn = A2_Attn
    except Exception as e: print(f"[WARN] A2_Attn 导入失败: {e}")
    try:
        from ultralytics.nn.modules.cbam_block import CBAM_Block; U.CBAM_Block = CBAM_Block
    except Exception as e: print(f"[WARN] CBAM_Block 导入失败: {e}")
    try:
        from ultralytics.nn.modules.ca_block import CA_Block; U.CA_Block = CA_Block
    except Exception as e: print(f"[WARN] CA_Block 导入失败: {e}")
    try:
        from ultralytics.nn.modules.smallobj_modules import SE_Block; U.SE_Block = SE_Block
    except Exception as e: print(f"[WARN] SE_Block 导入失败: {e}")
    try:
        from ultralytics.nn.modules.detect_stable import DetectStable; U.DetectStable = DetectStable
    except Exception as e: print(f"[WARN] DetectStable 导入失败：{e}")
    print("[INFO] 自定义模块注册完成（Swin/A2/CBAM/CA/SE/DetectStable）")

# ---------- 2) P2 延迟开启回调（保留） ----------
# 功能保持不变，对 E2-E6 仍然有效
def cb_p2_gate(close_p2_until:int):
    """# 功能：若模型头为 DetectStable，则在指定轮数前屏蔽P2尺度，之后自动开启"""
    def _cb(trainer):
        ep = trainer.epoch
        try:
            for m in trainer.model.modules():
                if hasattr(m, "set_active_mask") and hasattr(m, "nl"):
                    # E1的YAML中没有P2头，此代码不会执行
                    # E2-E6的YAML中有P2头，此代码会正常工作
                    mask = [ep >= close_p2_until] + [True] * (m.nl - 1)
                    m.set_active_mask(mask)
                    if ep == close_p2_until:
                        LOGGER.info(f"[P2] P2 检测层在 epoch {close_p2_until} 激活")
        except Exception:
            pass
    return _cb

# ---------- 3) 模块门控函数（已移除） ----------
# apply_module_gates 函数已不再需要，因为消融逻辑已经体现在各个独立的 YAML 文件中。

# ---------- 4) 单组训练封装（已简化） ----------
# 功能简化：不再需要 enable_modules 参数和 apply_module_gates 调用
def run_one(exp_name:str, cfg_path:str, data_path:str, args, p2_until:int, use_p2_gate=True):
    """# 功能：根据指定的 YAML 配置文件跑一组实验"""
    register_custom_modules()
    
    # 直接从指定的 cfg_path 加载模型，该 YAML 已定义好了消融结构
    model = YOLO(cfg_path)

    # 如果需要 P2 延迟开启（仅 DetectStable 有效），注册回调
    if use_p2_gate and p2_until > 0:
        model.add_callback("on_train_epoch_start", cb_p2_gate(p2_until))

    # 构造训练参数
    train_kwargs = dict(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/ablation-1",
        name=exp_name,
        workers=8,
        lr0=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        patience=7,
        save_period=10,
        seed=args.seed
    )

    LOGGER.info(f"--- [开始实验: {exp_name}] 使用配置='{cfg_path}', P2_delay={p2_until} ---")
    model.train(**train_kwargs)
    LOGGER.info(f"--- [完成实验: {exp_name}] ---")

# ---------- 5) 主流程：通过加载不同 YAML 调度 E1→E6 ----------
def main():
    """# 功能：解析参数并按顺序加载 E1-E6 的 YAML 文件执行消融实验"""
    parser = argparse.ArgumentParser(description="YOLOv12 Ablation Study Runner (YAML-based mode)")
    parser.add_argument("--data", default="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml", help="数据集配置文件路径")
    # 修改参数：从单个cfg文件改为cfg文件所在目录
    parser.add_argument("--cfg_dir", default="/workspace/yolo/ultralytics/cfg/models/new/", help="存放 E1-E6 YAML 文件的目录路径")
    parser.add_argument("--epochs", type=int, default=400, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=10, help="批处理大小")
    parser.add_argument("--device", default="0", help="使用的GPU设备ID")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lr", type=float, default=0.001, help="固定初始学习率")
    parser.add_argument("--close_p2_until", type=int, default=30, help="P2 延迟开启轮数 (仅 DetectStable 有效)")
    args = parser.parse_args()

    # 设置随机种子以保证可复现
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------- 消融序列（通过加载不同YAML实现）----------
    # 定义每个实验阶段的名称和对应的YAML文件名
    experiments = [
            ("E2_v5_P2", "E2.yaml"),
            ("E3_v5_P2_SE", "E3.yaml"),
            ("E4_v5_P2_SE_CBAM", "E4.yaml"),
            ("E5_v5_P2_SE_CBAM_Swin", "E5.yaml"),
            ("E6_v5_P2_SE_CBAM_Swin_A2", "E6.yaml"),
        ]
    # 循环执行所有实验
    for exp_name, yaml_filename in experiments:
        cfg_path = os.path.join(args.cfg_dir, yaml_filename)
        
        # 检查YAML文件是否存在，如果不存在则跳过
        if not os.path.exists(cfg_path):
            LOGGER.error(f"配置文件未找到，跳过实验 '{exp_name}': {cfg_path}")
            continue

        # 根据实验名称设置P2延迟开启逻辑
        # E1的YAML中没有P2头，即使设置了回调也不会产生影响
        p2_until = args.epochs + 1 if exp_name.startswith("E1") else args.close_p2_until
        
        # 运行单次实验
        run_one(exp_name, cfg_path, args.data, args, p2_until=p2_until, use_p2_gate=True)

    LOGGER.info("✅ 已完成 E1-E6 全部消融实验（YAML-based 模式），结果保存在 runs/ablation-1/ 目录下")

if __name__ == "__main__":
    main()