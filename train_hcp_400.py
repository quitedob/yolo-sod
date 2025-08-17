#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /workspace/yolo/train_hcp_400.py
# 功能: HCP-400 四阶段训练；删除无效参数(iou_t/callbacks)；使用 add_callback 注册 P2 开关；跨阶段累积计数

import os
import sys
import argparse
from ultralytics import YOLO
from ultralytics.utils import yaml_load

# ====== 生成 P2 开关回调（DetectStable 存在时生效；按“全局累计轮数”判断） ======  # 中文注释
def make_p2_toggle_cb(close_p2_until: int):
    """返回 on_train_epoch_start 回调；仅当 DetectStable 可用时生效"""
    global_epoch = {"n": 0}  # 使用可变字典跨阶段累计  # 中文注释
    def _cb(trainer):
        try:
            from ultralytics.nn.modules.detect_stable import DetectStable
        except Exception:
            return  # 未集成 DetectStable 时静默  # 中文注释
        # 根据全局累计轮数，而不是 trainer.epoch（每阶段都会清零）  # 中文注释
        n = global_epoch["n"]
        for m in trainer.model.modules():
            if isinstance(m, DetectStable):
                active = [n >= close_p2_until, True, True, True]  # [P2,P3,P4,P5]
                # 仅在前期需要时打印一次提示  # 中文注释
                if n == 0 and close_p2_until > 0:
                    trainer.console.info(f"[P2] 前 {close_p2_until} epoch 关闭 P2 分支；当前全局epoch={n}")
                m.set_active_mask(active)
        global_epoch["n"] += 1
    return _cb

# ====== 四阶段配置 ======  # 中文注释
STAGES = [
    {"name":"stage1", "epochs":50,  "freeze":[0,1,2,3,4,5,6,7,8,9,10,11], "desc":"初始化新头/颈，冻结深层骨干"},
    {"name":"stage2", "epochs":100, "freeze":None, "desc":"解冻全网络"},
    {"name":"stage3", "epochs":200, "freeze":None, "desc":"全局微调"},
    {"name":"stage4", "epochs":50,  "freeze":None, "desc":"收官精炼"},
]

def merge_hyp(stage_hyp:dict, common_hyp:dict):
    """阶段超参与公共项合并（后者为默认）"""
    merged = dict(common_hyp or {})
    merged.update(stage_hyp or {})
    return merged

def main():
    parser = argparse.ArgumentParser(description="HCP-400 分层收敛训练")
    parser.add_argument("--cfg", type=str, required=True, help="模型配置文件路径")
    parser.add_argument("--hyp", type=str, required=True, help="超参数配置文件路径")
    parser.add_argument("--data", type=str, required=True, help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=400, help="总训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default="0", help="训练设备(字符串，如 '0' 或 '0,1')")
    parser.add_argument("--close-p2-until", type=int, default=0, help="前 N 个 epoch 关闭 P2(DetectStable 有效)")
    args = parser.parse_args()

    for fp in [args.cfg, args.hyp, args.data]:
        if not os.path.exists(fp):
            print(f"[ERROR] 文件不存在: {fp}")
            sys.exit(1)

    model = YOLO(args.cfg)
    hyp_cfg = yaml_load(args.hyp)

    # ----- 注册回调（仅 DetectStable 有效；兼容版 Detect 将被忽略） -----  # 中文注释
    if args.close_p2_until > 0:
        cb = make_p2_toggle_cb(args.close_p2_until)
        try:
            model.add_callback("on_train_epoch_start", cb)
            print(f"[INFO] 已注册 P2 开关回调：前 {args.close_p2_until} epoch 关闭 P2（若 DetectStable 存在）")
        except Exception as e:
            print(f"[WARN] 回调注册失败，将忽略 P2 延迟：{e}")

    remain = args.epochs
    total_done = 0

    print(f"\n[INFO] HCP-400 启动 | 目标总轮数: {args.epochs} | imgsz={args.imgsz} | batch={args.batch} | device={args.device}")
    for s in STAGES:
        if remain <= 0:
            break
        this_epochs = min(remain, s["epochs"])

        # 合并阶段/公共超参（过滤掉新版不支持的键）  # 中文注释
        stage_hyp  = hyp_cfg.get(s["name"], {})
        common_hyp = {k:v for k,v in hyp_cfg.items() if not k.startswith("stage")}
        ovr = merge_hyp(stage_hyp, common_hyp)

        print(f"\n{'='*64}")
        print(f"[STAGE] {s['name']} | {s['desc']} | epochs={this_epochs} | freeze={s['freeze'] or 'None'}")
        print(f"{'='*64}")

        # 注意：不要传入 iou_t / callbacks 等不存在的键  # 中文注释
        train_args = dict(
            data=args.data,
            epochs=this_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project="runs_hcp_400",
            name=s["name"],
            exist_ok=True,
            pretrained=False,   # 避免结构不匹配  # 中文注释
            workers=8,
            optimizer=ovr.get("optimizer","AdamW"),
            lr0=ovr.get("lr0", 0.001),
            lrf=ovr.get("lrf", 0.1),
            momentum=ovr.get("momentum", 0.9),
            weight_decay=ovr.get("weight_decay", 0.01),
            warmup_epochs=ovr.get("warmup_epochs", 10),
            warmup_momentum=ovr.get("warmup_momentum", 0.8),
            warmup_bias_lr=ovr.get("warmup_bias_lr", 0.1),
            box=ovr.get("box", 7.5),
            cls=ovr.get("cls", 0.5),
            dfl=ovr.get("dfl", 1.5),
            hsv_h=ovr.get("hsv_h", 0.015),
            hsv_s=ovr.get("hsv_s", 0.7),
            hsv_v=ovr.get("hsv_v", 0.4),
            degrees=ovr.get("degrees", 0.0),
            translate=ovr.get("translate", 0.1),
            scale=ovr.get("scale", 0.5),
            shear=ovr.get("shear", 0.0),
            perspective=ovr.get("perspective", 0.0),
            flipud=ovr.get("flipud", 0.0),
            fliplr=ovr.get("fliplr", 0.5),
            mosaic=ovr.get("mosaic", 1.0),
            mixup=ovr.get("mixup", 0.1),
            copy_paste=ovr.get("copy_paste", 0.0),
            label_smoothing=ovr.get("label_smoothing", 0.0),
            cos_lr=True,
            close_mosaic=ovr.get("close_mosaic", 0),
            freeze=s["freeze"] or [],
        )

        model.train(**train_args)

        total_done += this_epochs
        remain -= this_epochs
        print(f"[INFO] 完成 {s['name']}。累计 epochs={total_done} / {args.epochs}")

    print(f"\n[SUCCESS] HCP-400 完成，总轮数={total_done}")

if __name__ == "__main__":
    main()
