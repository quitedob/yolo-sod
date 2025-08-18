# /workspace/yolo/train.py
# 说明：
# 1) 新增 --task {auto,detect,segment}，默认 auto；
# 2) 自动从 YAML 判别是否包含 Segment 头；若 detect-only 则强制用 detect；
# 3) 边界损失与 P2 开关、DETR-Aux 回调仅在对应任务/模块存在时启用；
# 4) 中文注释，逻辑清晰。

import os, sys, yaml, argparse
from pathlib import Path
from ultralytics import YOLO

# ========== 工具：判断 YAML 是否为分割头 ==========
def yaml_has_segment_head(cfg_path: str) -> bool:
    # 简单扫描 'Segment' 关键字（大多数 fork 可用；亦可改为解析结构）
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        return 'Segment' in txt or 'segment' in txt
    except Exception:
        return False

# ========== 工具：注册自定义模块 ==========
def register_plugins():
    import importlib, ultralytics.nn.modules as U
    try:
        bm = importlib.import_module('ultralytics.nn.modules.blocks_mamba')
    except Exception:
        bm = importlib.import_module('plugins.blocks_mamba')  # 兼容你把插件放在 plugins 下的情况
    try:
        bt = importlib.import_module('ultralytics.nn.modules.blocks_transformer')
    except Exception:
        bt = importlib.import_module('plugins.blocks_transformer')
    try:
        da = importlib.import_module('ultralytics.nn.modules.heads_detr_aux')
    except Exception:
        da = importlib.import_module('plugins.heads_detr_aux')

    U.MambaBlock  = getattr(bm, 'MambaBlock')
    U.SwinBlock   = getattr(bt, 'SwinBlock')
    U.DETRAuxHead = getattr(da, 'DETRAuxHead')
    print('[INFO] 成功注册自定义模块: MambaBlock, SwinBlock, DETRAuxHead')

# ========== 回调：P2关闭/开启 ==========
def make_p2_toggle_cb(close_p2_until:int=30):
    global_epoch = {'n': 0}
    def _cb(trainer):
        try:
            from ultralytics.nn.modules.detect_stable import DetectStable
        except Exception:
            return
        n = global_epoch['n']
        for m in trainer.model.modules():
            if isinstance(m, DetectStable):
                # 前N轮关闭P2（仅 DetectStable 有 set_active_mask）
                try:
                    m.set_active_mask([n >= close_p2_until, True, True, True])
                except Exception:
                    pass
        global_epoch['n'] += 1
    return _cb

# ========== 回调：边界感知损失 ==========
def make_boundary_loss_cb(weight=0.2, edge_w=1.0, bce_w=1.0, iou_w=0.0):
    try:
        from plugins.loss_boundary import BoundaryAwareLoss
    except Exception:
        from ultralytics.nn.modules.loss_boundary import BoundaryAwareLoss
    bal = BoundaryAwareLoss(edge_weight=edge_w, bce_weight=bce_w, iou_weight=iou_w)
    def _cb(trainer):
        # 仅在 segment 任务下尝试取掩码
        try:
            batch = getattr(trainer, 'batch', None)
            pm    = getattr(trainer, 'masks', None)  # 不同版本命名不同
            if pm is None or batch is None or 'masks' not in batch:
                return
            gt = batch['masks'].float()
            pr = pm.float()
            if pr.ndim == 4 and pr.shape[1] != 1: pr = pr[:, :1]
            if gt.ndim == 4 and gt.shape[1] != 1: gt = gt[:, :1]
            trainer.loss += bal(pr, gt) * weight
        except Exception:
            pass
    return _cb

# ========== 回调：DETR-Aux（此处仅示例占位，实际蒸馏逻辑按需实现） ==========
def make_detr_aux_cb():
    def _cb(trainer):
        # 这里留给你后续接匈牙利匹配与 L1/GIoU 的蒸馏实现
        return
    return _cb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='模型yaml')
    parser.add_argument('--data', required=True, help='数据集yaml')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--hyp', default=None)
    parser.add_argument('--task', default='auto', choices=['auto','detect','segment'], help='训练任务类型')
    parser.add_argument('--close_p2_until', type=int, default=30)
    parser.add_argument('--use_boundary_loss', action='store_true')
    parser.add_argument('--use_detr_aux', action='store_true')
    args = parser.parse_args()

    # 任务类型判定：auto -> 基于 YAML 是否含 Segment 头
    if args.task == 'auto':
        inferred = 'segment' if yaml_has_segment_head(args.cfg) else 'detect'
        print(f'[INFO] 基于 YAML 自动判定任务: {inferred}')
        task = inferred
    else:
        task = args.task

    # 一致性强校验：detect-only YAML + segment 任务会直接报错，避免“过了构图、死在损失”
    has_segment = yaml_has_segment_head(args.cfg)
    if task == 'segment' and not has_segment:
        raise RuntimeError(f"[配置不一致] 选择了 task=segment，但 {args.cfg} 不含 Segment 头；请改为 task=detect 或更换含 Segment 头的 YAML。")
    if task == 'detect' and has_segment:
        print("[WARNING] 选择了 task=detect，但 YAML 含 Segment 头；将按 detect 任务训练，只是不计算掩码损失。")

    # 注册插件与构建模型
    sys.path.append(str(Path(__file__).parent))  # 保底
    register_plugins()
    print('[INFO] 正在初始化YOLO模型...')
    model = YOLO(args.cfg, task=task)

    # 注册通用回调
    if args.close_p2_until > 0:
        model.add_callback('on_train_epoch_start', make_p2_toggle_cb(args.close_p2_until))

    # 仅在 segment 任务下启用边界感知损失
    if task == 'segment' and args.use_boundary_loss:
        model.add_callback('on_train_batch_end', make_boundary_loss_cb())

    # DETR-Aux（与任务无强耦合；如你只想在 segment 下开，这里亦可加判断）
    if args.use_detr_aux:
        model.add_callback('on_train_batch_end', make_detr_aux_cb())

    # 训练参数
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project='runs_fusion', name='yolo_sod_fusion_exp', exist_ok=True, pretrained=False, cos_lr=True,
        deterministic=True, verbose=True
    )
    if args.hyp:  # ★ 把超参文件透传给 Ultralytics
        train_kwargs.update({'cfg': args.hyp})

    results = model.train(**train_kwargs)
    print('[INFO] 训练完成')
    return results

if __name__ == '__main__':
    main()
