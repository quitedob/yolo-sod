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
    
    # 2. 检查配置字典中的关键信息
    # 如果有分割相关的head或模块，推断为分割任务
    head_info = cfg_dict.get('head', [])
    if isinstance(head_info, list):
        head_str = str(head_info).lower()
        if 'segment' in head_str or 'mask' in head_str:
            return 'segment'
    
    # 3. 检查辅助头信息
    aux_head_info = cfg_dict.get('aux_head', [])
    if isinstance(aux_head_info, list):
        aux_head_str = str(aux_head_info).lower()
        if 'segment' in aux_head_str or 'mask' in aux_head_str:
            return 'segment'
    
    # 4. 检查整个配置文件内容
    cfg_content = str(cfg_dict).lower()
    if 'segment' in cfg_content or 'mask' in cfg_content:
        return 'segment'
    elif 'classify' in cfg_content or 'classification' in cfg_content:
        return 'classify'
    elif 'pose' in cfg_content or 'keypoint' in cfg_content:
        return 'pose'
    
    # 5. 默认返回检测任务
    return 'detect'

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
    A. 读取原文并“清洗”：去掉首行残留的 'yaml' / 去除 ``` 围栏 / 去BOM / 去两端空行
    B. 先用 ultralytics.utils.yaml_load 尝试；失败或结果异常则用清洗后的文本再 safe_load
    C. 若存在顶级 'neck:'，将 neck 列表顺序并入 head 列表（顶级只保留 backbone/head）
    D. 最终确保存在 backbone 与 head 两段
    """
    import os, io, re, yaml
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型配置文件未找到: {path}")

    def _read_text(fp: str) -> str:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _clean_text(t: str) -> str:
        # 1) 去掉 UTF-8 BOM
        if t.startswith("\ufeff"):
            t = t.lstrip("\ufeff")
        # 2) 去掉 Markdown 代码围栏 ```xxx ... ```
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"\s*```(\s*#.*)?\s*$", "", t, flags=re.MULTILINE)
        # 3) 去掉文件开头可能残留的语言标记行（如单独的 'yaml' 或 'yml'）
        #    仅当它是“首个非空非注释行”时才剔除
        lines = t.splitlines()
        cleaned = []
        removed_lang = False
        seen_content = False
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not seen_content:
                if s == "" or s.startswith("#"):
                    cleaned.append(ln)
                    continue
                # 首个内容行：若就是 'yaml' / 'yml'，则跳过
                if s.lower() in {"yaml", "yml"}:
                    removed_lang = True
                    continue
                # 其他情况：正常内容
                seen_content = True
                cleaned.append(ln)
            else:
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
        # 若返回类型异常或返回空映射，尝试用清洗后的文本再解析一次
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
    has_backbone = "backbone" in cfg
    has_head = "head" in cfg
    if "neck" in cfg:
        neck_block = cfg.get("neck") or []
        head_block = cfg.get("head") or []
        if not isinstance(neck_block, list) or not isinstance(head_block, list):
            top_keys = ", ".join(cfg.keys())
            raise ValueError(f"[错误] {path} 中 neck/head 不是列表，当前顶级键：[{top_keys}]。")
        cfg["head"] = list(neck_block) + list(head_block)  # 先 neck 再 head，保持执行顺序
        del cfg["neck"]
        has_head = True

    # 最终检查：必须同时有 backbone 与 head
    if not has_backbone or not has_head:
        top_keys = ", ".join(cfg.keys())
        # 打印前若有“脏首行残留”的现象，可从归档中看到：首行是 'yaml'，随后才是结构键位（示例）  # 中文注释
        # 参考：yolo-main_All4.txt 的 YAML 片段，首行 'yaml' → 其后才是 nc/backbone（会导致解析异常）  # 中文注释
        raise ValueError(
            f"[错误] {path} 缺少 'backbone' 或 'head'。当前顶级键：[{top_keys}]。\n"
            f"请确认YAML首行没有多余的语言标记（如单独的 'yaml'），必要时删除后再试。"
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

    # ★ 关键：直接传递 YAML 文件路径给 YOLO，让其内部解析
    print("[INFO] 正在初始化YOLO模型...")
    model = YOLO(args.cfg, task=task)

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
