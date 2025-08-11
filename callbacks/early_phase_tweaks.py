# /workspace/yolo/callbacks/early_phase_tweaks.py  # 文件路径
# 作用：在早期若分类损失异常高，动态降低分类权重与学习率，进一步稳住训练  # 中文注释

def on_train_epoch_end(trainer):  # 每轮训练结束  # 中文注释
    try:
        # 从最近一次记录中估算分类损失（若可用），否则使用总损失  # 中文注释
        hist = getattr(trainer, "loss", None)
        tloss = getattr(trainer, "tloss", None)
        # 动态策略：在前10个epoch内，若检测到异常大则降低分类权重与学习率  # 中文注释
        if trainer.epoch < 10:
            est = 0.0
            if hist is not None:
                est = float(hist) if not hasattr(hist, "__len__") else float(sum(map(float, hist)) / max(len(hist), 1))
            elif tloss is not None:
                est = float(tloss) if not hasattr(tloss, "__len__") else float(sum(map(float, tloss)) / max(len(tloss), 1))
            if est > 1000:
                # 降低学习率  # 中文注释
                for g in trainer.optimizer.param_groups:
                    g["lr"] *= 0.5
                # 降低分类权重（若存在该参数）  # 中文注释
                if hasattr(trainer.args, "cls"):
                    trainer.args.cls = max(0.2, trainer.args.cls * 0.8)
    except Exception:
        pass


