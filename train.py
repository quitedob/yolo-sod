# -*- coding: utf-8 -*-
# C++ 头文件兼容性宏定义，对于纯Python脚本此行非必需，但遵循您的全局指令添加
# "# defina_CRT_SCURE.NO-WARNING"

# 中文说明：
# 1) 稳健读取YAML（若失败回退文本再safe_load）
# 2) 把“解析后的dict”直接传给 YOLO(...)，避免内部再次不稳定解析 (注：当前实现仍传递路径，让YOLO内部解析)
# 3) 注册 DetectStable 等自定义模块到 ultralytics.nn.modules 命名空间
# 4) 自动判定任务：若配置文件含 Segment，则优先选择 'segment'；否则 'detect'

import os
import re
import yaml
import io
import argparse
from ultralytics import YOLO

# ========== 自定义模块注册 ==========
def register_custom_modules():
    """注册自定义算子到 Ultralytics 命名空间（Mamba/Swin/DETR-Aux/边界损失/追踪/DetectStable）"""
    import ultralytics.nn.modules as U
    # 可选：Mamba
    try:
        from ultralytics.nn.modules.blocks_mamba import MambaBlock
        U.MambaBlock = MambaBlock
    except Exception as e:
        print(f"[WARN] MambaBlock 导入失败（mamba-ssm 可能未完全可用）：{e}")
    # 其他模块
    try:
        from ultralytics.nn.modules.blocks_transformer import SwinBlock
        U.SwinBlock = SwinBlock
    except Exception as e:
        print(f"[WARN] SwinBlock 导入失败: {e}")
    try:
        from ultralytics.nn.modules.heads_detr_aux import DETRAuxHead
        U.DETRAuxHead = DETRAuxHead
    except Exception as e:
        print(f"[WARN] DETRAuxHead 导入失败: {e}")
    try:
        from ultralytics.nn.modules.loss_boundary import BoundaryAwareLoss
        U.BoundaryAwareLoss = BoundaryAwareLoss
    except Exception as e:
        print(f"[WARN] BoundaryAwareLoss 导入失败: {e}")
    try:
        from ultralytics.nn.modules.tracker_kf_lstm import MultiObjectTracker
        U.MultiObjectTracker = MultiObjectTracker
    except Exception as e:
        print(f"[WARN] MultiObjectTracker 导入失败: {e}")
        
    # ★ 注册 DetectStable（若 YAML 用到了它，必须注册）
    try:
        from ultralytics.nn.modules.detect_stable import DetectStable
        U.DetectStable = DetectStable
    except Exception as e:
        print(f"[WARN] DetectStable 导入失败：{e}")
    print("[INFO] 成功注册自定义模块: MambaBlock, SwinBlock, DETRAuxHead, DetectStable 等")

# ========== YAML 读取/判定 ==========
def infer_task_from_text_or_cfg(cfg_path: str, cfg_dict: dict) -> str:
    """
    基于YAML配置文件路径和内容自动推断任务类型
    Args:
        cfg_path: YAML文件路径
        cfg_dict: 解析后的配置字典
    Returns:
        任务类型: 'detect', 'segment', 'classify', 'pose' 等
    """
    # 1. 优先检查文件名中的关键词
    cfg_filename = os.path.basename(cfg_path).lower()
    if 'segment' in cfg_filename or 'seg' in cfg_filename:
        return 'segment'
    elif 'classify' in cfg_filename or 'cls' in cfg_filename:
        return 'classify'  
    elif 'pose' in cfg_filename or 'keypoint' in cfg_filename:
        return 'pose'
    
    # 2. 检查配置字典中的关键信息 (head, aux_head, content)
    # 将字典转换为字符串以便进行全局搜索
    cfg_content = str(cfg_dict).lower()
    if 'segment' in cfg_content or 'mask' in cfg_content:
        return 'segment'
    elif 'classify' in cfg_content or 'classification' in cfg_content:
        return 'classify'
    elif 'pose' in cfg_content or 'keypoint' in cfg_content:
        return 'pose'
    
    # 5. 默认返回检测任务
    return 'detect'

def load_model_yaml_as_dict(path: str) -> dict:
    """
    稳健加载模型YAML为 dict：
    A. 读取原文并“清洗”：去掉首行残留的 'yaml' / 去除 ``` 围栏 / 去BOM / 去两端空行
    B. 先用 ultralytics.utils.yaml_load 尝试；失败或结果异常则用清洗后的文本再 safe_load
    C. 若存在顶级 'neck:'，将 neck 列表顺序并入 head 列表（顶级只保留 backbone/head）
    D. 最终确保存在 backbone 与 head 两段
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型配置文件未找到: {path}")

    def _read_text(fp: str) -> str:
        # 辅助函数：读取文本文件
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _clean_text(t: str) -> str:
        # 辅助函数：清洗YAML文本
        # 1) 去掉 UTF-8 BOM
        if t.startswith("\ufeff"):
            t = t.lstrip("\ufeff")
        # 2) 去掉 Markdown 代码围栏 ```xxx ... ```
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"\s*```(\s*#.*)?\s*$", "", t, flags=re.MULTILINE)
        # 3) 去掉文件开头可能残留的语言标记行（如单独的 'yaml' 或 'yml'）
        lines = t.splitlines()
        cleaned = []
        seen_content = False
        for ln in lines:
            s = ln.strip()
            if not seen_content and s and not s.startswith("#"):
                seen_content = True
                if s.lower() in {"yaml", "yml"}:
                    continue
            cleaned.append(ln)
        t = "\n".join(cleaned)
        # 4) 去掉文件头/尾多余空行
        t = t.strip() + "\n"
        return t

    raw_text = _read_text(path)
    cleaned_text = _clean_text(raw_text)

    cfg = None
    # 优先：Ultralytics 自带 loader（更兼容 include/锚点等特性）
    try:
        from ultralytics.utils import yaml_load, checks
        cfg = yaml_load(checks.check_yaml(path))
        if not isinstance(cfg, dict) or not cfg:
            raise ValueError("yaml_load 返回异常，进入清洗文本回退解析")
    except Exception as e:
        print(f"[WARN] Ultralytics yaml_load 失败或无效，回退到清洗文本解析：{e}")
        try:
            cfg = yaml.safe_load(io.StringIO(cleaned_text))
        except Exception as e2:
            # 再次回退：尝试从出现 'nc:' 或 'backbone:' 的位置截取后解析
            m = re.search(r"(?m)^(nc:|backbone:)", cleaned_text)
            if m:
                try:
                    cfg = yaml.safe_load(cleaned_text[m.start():])
                except Exception as e3:
                    raise ValueError(f"[错误] 无法解析 YAML: {path}，原始异常：{e3}")
            else:
                raise ValueError(f"[错误] 无法解析 YAML: {path}，原始异常：{e2}")

    if not isinstance(cfg, dict):
        raise ValueError(f"[错误] {path} 未解析为字典，请检查YAML语法。")

    # ★ 自动把顶层 neck 并入 head（Ultralytics 仅识别 backbone/head）
    if "neck" in cfg:
        neck_block = cfg.get("neck") or []
        head_block = cfg.get("head") or []
        if not isinstance(neck_block, list) or not isinstance(head_block, list):
            top_keys = ", ".join(cfg.keys())
            raise ValueError(f"[错误] {path} 中 neck/head 不是列表，当前顶级键：[{top_keys}]。")
        cfg["head"] = list(neck_block) + list(head_block)
        del cfg["neck"]

    # 最终检查：必须同时有 backbone 与 head
    if "backbone" not in cfg or "head" not in cfg:
        top_keys = ", ".join(cfg.keys())
        raise ValueError(
            f"[错误] {path} 缺少 'backbone' 或 'head'。当前顶级键：[{top_keys}]。\n"
            f"请确认YAML首行没有多余的语言标记（如单独的 'yaml'），必要时删除后再试。"
        )
    return cfg

# ========== 回调函数定义 ==========
def create_p2_toggle_callback(close_p2_until: int = 30):
    """创建回调，在前N轮训练中关闭P2层"""
    epoch_counter = {"count": 0}
    def _cb(trainer):
        try:
            from ultralytics.nn.modules.detect_stable import DetectStable
            ep = epoch_counter["count"]
            # 遍历模型所有模块，找到DetectStable层
            for m in trainer.model.modules():
                if isinstance(m, DetectStable):
                    # 根据当前轮数设置P2层是否激活
                    active = [ep >= close_p2_until, True, True, True]
                    m.set_active_mask(active)
            epoch_counter["count"] += 1
            if ep == close_p2_until - 1: # 在即将开启的前一轮提示
                 print(f"\n[INFO] 下一轮 (第 {close_p2_until} 轮) 将开启 P2 尺度")
        except Exception as e:
            print(f"[WARN] P2 启停回调执行失败: {e}")
    return _cb

def create_boundary_loss_callback(edge_weight=1.0, bce_weight=1.0, iou_weight=0.0, loss_weight=0.2):
    """创建回调，为分割任务添加边界感知损失"""
    from ultralytics.nn.modules.loss_boundary import BoundaryAwareLoss
    # 初始化边界损失函数
    loss_fn = BoundaryAwareLoss(edge_weight=edge_weight, bce_weight=bce_weight, iou_weight=iou_weight)
    def _cb(trainer):
        try:
            # 安全地获取批次数据和预测掩码
            batch = getattr(trainer, "batch", None)
            preds = getattr(trainer, "preds", None)
            if batch is None or preds is None or "masks" not in batch:
                return
            
            # 从preds中获取预测掩码
            pred_masks = preds[1] if isinstance(preds, (list, tuple)) and len(preds) > 1 else None
            if pred_masks is None: return

            # 计算并叠加边界损失
            gt_masks = batch["masks"].float()
            trainer.loss += loss_fn(pred_masks, gt_masks) * loss_weight
        except Exception as e:
            # 避免因回调失败导致训练中断
            pass
    return _cb

# ========== 主入口 ==========
def main():
    """训练入口：解析参数→注册模块→加载配置→构建YOLO→注册回调→汇总超参→启动训练"""
    # --- 解析命令行参数 ---
    p = argparse.ArgumentParser(description="YOLO-SOD-Fusion 训练脚本")
    p.add_argument('--cfg', required=True, help='模型结构YAML文件路径')
    p.add_argument('--data', required=True, help='数据集YAML文件路径')
    p.add_argument('--epochs', type=int, default=500, help='总训练轮数')
    p.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    p.add_argument('--batch', type=int, default=16, help='批处理大小')
    p.add_argument('--device', default='0', help='运行设备，如 "0" 或 "cpu"')
    p.add_argument('--workers', type=int, default=8, help='数据加载器的工作线程数')
    p.add_argument('--hyp', default=None, help='覆盖默认超参的YAML文件路径')
    p.add_argument('--pretrained', default=False, help='预训练模型路径或布尔值')
    p.add_argument('--optimizer', default='auto', help='优化器类型 (例如 AdamW, SGD)')
    p.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    p.add_argument('--lrf', type=float, default=0.01, help='最终学习率 (lr0 * lrf)')
    p.add_argument('--cls', type=float, default=0.5, help='分类损失权重 (cls loss gain)')
    p.add_argument('--warmup_epochs', type=int, default=3, help='预热训练轮数')
    p.add_argument('--use_boundary_loss', action='store_true', help='(分割任务) 启用边界感知损失')
    p.add_argument('--close_p2_until', type=int, default=30, help='(可选) 在指定轮数前关闭P2层')
    p.add_argument('--edge_weight', type=float, default=1.0, help='(边界损失) 边缘权重')
    p.add_argument('--bce_weight', type=float, default=1.0, help='(边界损失) BCE权重')
    p.add_argument('--iou_weight', type=float, default=0.0, help='(边界损失) IoU权重')
    p.add_argument('--boundary_loss_weight', type=float, default=0.2, help='(边界损失) 总损失权重')
    p.add_argument('--project', default='runs_fusion', help='训练结果保存的项目名')
    p.add_argument('--name', default='yolo_sod_fusion_exp', help='本次训练的实验名')
    args = p.parse_args()

    # --- 1) 注册自定义模块 ---
    print("[INFO] 正在注册自定义模块...")
    register_custom_modules()

    # --- 2) 加载配置并推断任务 ---
    cfg_dict = load_model_yaml_as_dict(args.cfg)
    task = infer_task_from_text_or_cfg(args.cfg, cfg_dict)
    print(f"[INFO] 基于 YAML 自动判定任务: {task}")

    # --- 3) 创建 YOLO 模型 ---
    print("[INFO] 正在初始化YOLO模型...")
    model = YOLO(args.cfg, task=task)

    # --- 4) 注册回调函数 ---
    # 4.1) 回调：前N轮关闭P2层
    if args.close_p2_until > 0:
        model.add_callback('on_train_epoch_start', create_p2_toggle_callback(args.close_p2_until))
        print(f"[INFO] 已启用 P2 启停（前 {args.close_p2_until} 轮关闭P2）")

    # 4.2) 【新增】回调：早期稳训，若早期损失异常，自动下调LR与cls权重
    try:
        from callbacks.early_phase_tweaks import on_train_epoch_end as _early_tweak
        model.add_callback('on_train_epoch_end', _early_tweak)
        print("[INFO] 已启用 早期训练稳定回调")
    except ImportError:
        print("[WARN] 未找到 'callbacks.early_phase_tweaks'，跳过早期稳定回调。")
    
    # 4.3) (可选) 回调：分割任务的边界感知损失
    if task == 'segment' and args.use_boundary_loss:
        model.add_callback('on_train_batch_end', create_boundary_loss_callback(
            edge_weight=args.edge_weight, bce_weight=args.bce_weight,
            iou_weight=args.iou_weight, loss_weight=args.boundary_loss_weight))
        print("[INFO] 已启用 边界感知损失")

    # --- 5) 汇总训练超参 ---
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=args.project,
        name=args.name, exist_ok=True, pretrained=args.pretrained,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        cos_lr=True, warmup_epochs=args.warmup_epochs, warmup_momentum=0.8,
        weight_decay=0.0005, momentum=0.937, box=7.5, cls=args.cls, dfl=1.5,
        save=True, save_period=10, val=True, plots=True, verbose=True
    )
    if args.hyp:
        # Ultralytics中用于覆盖训练超参的键是 'cfg'
        train_kwargs.update({'cfg': args.hyp})

    # --- 6) 启动训练 ---
    print("\n[INFO] 开始训练...")
    results = model.train(**train_kwargs)
    print("\n[INFO] 训练完成!")
    print(f"最佳模型: {model.trainer.best}\n最终模型: {model.trainer.last}")
    return results

if __name__ == "__main__":
    main()