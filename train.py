# /workspace/yolo/train.py
# 中文说明：
# 1) 稳健读取YAML（若失败回退文本再safe_load）
# 2) 把“解析后的dict”直接传给 YOLO(...)，避免内部再次不稳定解析
# 3) 注册 DetectStable 到 ultralytics.nn.modules 命名空间
# 4) 自动判定任务：若文本含 Segment，则优先选择 'segment'；否则 'detect'

import os, re, yaml
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
    from ultralytics.nn.modules.blocks_transformer import SwinBlock
    from ultralytics.nn.modules.heads_detr_aux import DETRAuxHead
    from ultralytics.nn.modules.loss_boundary import BoundaryAwareLoss
    from ultralytics.nn.modules.tracker_kf_lstm import MultiObjectTracker
    U.SwinBlock = SwinBlock
    U.DETRAuxHead = DETRAuxHead
    U.BoundaryAwareLoss = BoundaryAwareLoss
    U.MultiObjectTracker = MultiObjectTracker
    # ★ 注册 DetectStable（若 YAML 用到了它，必须注册）
    try:
        from ultralytics.nn.modules.detect_stable import DetectStable
        U.DetectStable = DetectStable
    except Exception as e:
        print(f"[WARN] DetectStable 导入失败：{e}")
    print("[INFO] 成功注册自定义模块: MambaBlock, SwinBlock, DETRAuxHead")

# ========== YAML 读取/判定 ==========
def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def _ultra_yaml_load(path: str):
    """优先用 Ultralytics 的 yaml_load；失败则返回 Exception 让上层兜底"""
    from ultralytics.utils import yaml_load, checks
    return yaml_load(checks.check_yaml(path))

# /workspace/yolo/train.py  —— 替换此函数
def load_model_yaml_as_dict(path: str) -> dict:
    """
    稳健加载模型YAML为 dict：
    1) 先尝试 ultralytics.utils.yaml_load
    2) 失败则读原文再用 yaml.safe_load
    3) 若存在顶级 'neck:'，则把 neck 列表按顺序并入 head 列表（顶级只保留 backbone/head）
    4) 最终确保至少存在 backbone 与 head 两段
    """
    import yaml
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型配置文件未找到: {path}")

    # 优先：Ultralytics 自带 loader（更兼容）
    try:
        from ultralytics.utils import yaml_load, checks
        cfg = yaml_load(checks.check_yaml(path))
    except Exception as e:
        print(f"[WARN] Ultralytics yaml_load 失败，回退到 yaml.safe_load：{e}")
        # 回退：读取原文再 safe_load
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        try:
            cfg = yaml.safe_load(txt)
        except Exception as e2:
            raise ValueError(f"[错误] 无法解析 YAML: {path}，原始异常：{e2}")

    if not isinstance(cfg, dict):
        raise ValueError(f"[错误] {path} 未解析为字典，请检查YAML语法。")

    # ★ 关键修复：若存在顶级 neck，则并入 head，顶级只保留 backbone/head
    has_backbone = "backbone" in cfg
    has_head = "head" in cfg
    has_neck = "neck" in cfg

    if has_neck:
        neck_block = cfg.get("neck") or []
        head_block = cfg.get("head") or []
        if not isinstance(neck_block, list) or not isinstance(head_block, list):
            raise ValueError(f"[错误] {path} 中 neck/head 不是列表，请检查YAML。")
        # 先 neck，再原 head，保持执行顺序一致
        merged_head = list(neck_block) + list(head_block)
        cfg["head"] = merged_head
        # 删除顶级 neck，符合 Ultralytics 仅识别 backbone/head 的习惯
        del cfg["neck"]
        has_head = True  # 合并后必然有 head

    # 最终检查：至少 backbone 与 head 同时存在
    if not has_backbone or not has_head:
        # 这里给出具体提示，方便你检查
        top_keys = ", ".join(cfg.keys())
        raise ValueError(
            f"[错误] {path} 缺少 'backbone' 或 'head'。当前顶级键：[{top_keys}]。\n"
            f"若原本存在 'neck:'，请确认已正确合并到 'head:'。"
        )

    return cfg

# ========== 回调：P2 启停 & 边界损失（与之前相同，略） ==========
def create_p2_toggle_callback(close_p2_until: int = 30):
    epoch_counter = {"count": 0}
    def _cb(trainer):
        try:
            from ultralytics.nn.modules.detect_stable import DetectStable
            ep = epoch_counter["count"]
            for m in trainer.model.modules():
                if isinstance(m, DetectStable):
                    active = [ep >= close_p2_until, True, True, True]
                    m.set_active_mask(active)
            epoch_counter["count"] += 1
            if ep == close_p2_until:
                print(f"[INFO] 第 {close_p2_until} 轮：P2 尺度已开启")
        except Exception as e:
            print(f"[WARN] P2 回调失败: {e}")
    return _cb

def create_boundary_loss_callback(edge_weight=1.0, bce_weight=1.0, iou_weight=0.0, loss_weight=0.2):
    from ultralytics.nn.modules.loss_boundary import BoundaryAwareLoss
    fn = BoundaryAwareLoss(edge_weight=edge_weight, bce_weight=bce_weight, iou_weight=iou_weight)
    def _cb(trainer):
        try:
            batch = getattr(trainer, "batch", None)
            pred_masks = None
            for k in ("masks", "seg_masks", "pred_masks"):
                v = getattr(trainer, k, None)
                if v is not None:
                    pred_masks = v; break
            if pred_masks is None or batch is None or "masks" not in batch:
                return
            gt_masks = batch["masks"].float()
            pred_masks = pred_masks.float()
            if pred_masks.dim() == 4 and pred_masks.size(1) > 1:
                pred_masks = pred_masks[:, :1]
            if gt_masks.dim() == 4 and gt_masks.size(1) > 1:
                gt_masks = gt_masks[:, :1]
            trainer.loss += fn(pred_masks, gt_masks) * loss_weight
        except Exception:
            pass
    return _cb

# ========== 主入口 ==========
def main():
    import argparse
    p = argparse.ArgumentParser("YOLO-SOD-Fusion 训练脚本")
    p.add_argument('--cfg', required=True, help='模型结构YAML路径')
    p.add_argument('--data', required=True, help='数据集YAML路径')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--hyp', default=None, help='训练覆盖超参YAML')
    p.add_argument('--pretrained', default=False)
    p.add_argument('--optimizer', default='auto')
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--use_boundary_loss', action='store_true')
    p.add_argument('--close_p2_until', type=int, default=30)
    p.add_argument('--use_detr_aux', action='store_true')
    p.add_argument('--edge_weight', type=float, default=1.0)
    p.add_argument('--bce_weight', type=float, default=1.0)
    p.add_argument('--iou_weight', type=float, default=0.0)
    p.add_argument('--boundary_loss_weight', type=float, default=0.2)
    p.add_argument('--project', default='runs_fusion')
    p.add_argument('--name', default='yolo_sod_fusion_exp')
    args = p.parse_args()

    print("[INFO] 正在注册自定义模块...")
    register_custom_modules()

    # 读取并校验模型YAML -> dict
    cfg_dict = load_model_yaml_as_dict(args.cfg)
    # 基于文本/内容自动判定任务（包含 Segment 则优先选 segment）
    task = infer_task_from_text_or_cfg(args.cfg, cfg_dict)
    print(f"[INFO] 基于 YAML 自动判定任务: {task}")

    # ★ 关键：把 dict 直接传给 YOLO，避免内部再解析路径
    print("[INFO] 正在初始化YOLO模型...")
    model = YOLO(cfg_dict, task=task)

    # 回调
    if args.close_p2_until > 0:
        model.add_callback('on_train_epoch_start', create_p2_toggle_callback(args.close_p2_until))
        print(f"[INFO] 已启用 P2 启停（前 {args.close_p2_until} 轮关闭P2）")
    if task == 'segment' and args.use_boundary_loss:
        model.add_callback('on_train_batch_end', create_boundary_loss_callback(
            edge_weight=args.edge_weight, bce_weight=args.bce_weight,
            iou_weight=args.iou_weight, loss_weight=args.boundary_loss_weight))
        print("[INFO] 已启用 边界感知损失")

    # 训练参数
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=args.project,
        name=args.name, exist_ok=True, pretrained=args.pretrained,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        cos_lr=True, warmup_epochs=3, warmup_momentum=0.8,
        weight_decay=0.0005, momentum=0.937, box=7.5, cls=0.5, dfl=1.5,
        save=True, save_period=10, val=True, plots=True, verbose=True
    )
    if args.hyp:
        train_kwargs.update({'cfg': args.hyp})  # ulty 的训练覆盖键叫 cfg

    print("[INFO] 开始训练...")
    results = model.train(**train_kwargs)
    print("[INFO] 训练完成!")
    print(f"best: {model.trainer.best}\nlast: {model.trainer.last}")
    return results

if __name__ == "__main__":
    main()
