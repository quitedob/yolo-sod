# 文件路径：/workspace/yolo/callbacks/toggle_p2.py  # 回调：按epoch分阶段启用P2尺度

def on_train_epoch_start(trainer):  # 训练每个epoch开始时触发  # 中文注释
    threshold = 30  # 固定前30个epoch关闭P2（延后P2介入以稳住CLS）  # 中文注释
    epoch = trainer.epoch  # 当前epoch编号  # 中文注释
    model = trainer.model  # 训练模型  # 中文注释
    # 延迟导入以避免循环依赖  # 中文注释
    from ultralytics.nn.modules.detect_stable import DetectStable  # 导入可控Detect  # 中文注释
    for m in model.modules():  # 遍历模型所有子模块  # 中文注释
        if isinstance(m, DetectStable):  # 找到DetectStable实例  # 中文注释
            if epoch < threshold:  # 若在阈值之前  # 中文注释
                m.set_active_mask([False, True, True, True])  # 关闭P2，开启P3/4/5  # 中文注释
            else:  # 否则  # 中文注释
                m.set_active_mask([True, True, True, True])  # 全部开启  # 中文注释


