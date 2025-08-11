# 文件路径：/workspace/yolo/train_stable_visdrone.py  # 一键稳定训练脚本（AMP/裁剪/Warmup/EMA/回调）
from ultralytics import YOLO  # 引入Ultralytics训练入口  # 中文注释
from callbacks.toggle_p2 import on_train_epoch_start  # 导入回调函数以阶段化启用P2  # 中文注释


def main():  # 主函数：配置并启动训练  # 中文注释
    model_cfg = "ultralytics/cfg/models/new/yolov12-mambafusion-smallobj-640.yaml"  # 模型配置路径  # 中文注释
    data_cfg = "ultralytics/cfg/datasets/visdrone.yaml"  # 数据集配置路径（请确认存在）  # 中文注释
    hyp_cfg = "ultralytics/cfg/hyp_visdrone_stable.yaml"  # 稳定化超参路径  # 中文注释

    model = YOLO(model_cfg)  # 构建模型  # 中文注释

    # 注册回调：训练早期关闭P2，避免梯度/损失爆炸  # 中文注释
    model.add_callback("on_train_epoch_start", on_train_epoch_start)  # 注册事件回调  # 中文注释

    results = model.train(  # 启动训练并返回结果  # 中文注释
        data=data_cfg,  # 数据集配置  # 中文注释
        epochs=300,  # 训练轮次  # 中文注释
        imgsz=640,  # 输入分辨率  # 中文注释
        device=0,  # 指定GPU设备  # 中文注释
        batch=16,  # 批大小（视显存调整）  # 中文注释
        workers=8,  # DataLoader工作线程  # 中文注释
        optimizer="AdamW",  # 优化器选择  # 中文注释
        lr0=0.002,  # 初始学习率再降低，缓解早期cls_loss峰值  # 中文注释
        lrf=0.05,  # 余弦退火最终比率  # 中文注释
        momentum=0.9,  # 动量项  # 中文注释
        weight_decay=0.05,  # 权重衰减  # 中文注释
        amp=True,  # 开启混合精度  # 中文注释
        cos_lr=True,  # 使用余弦学习率  # 中文注释
        warmup_epochs=10,  # 延长预热稳定早期  # 中文注释
        close_mosaic=10,  # 前期减少马赛克增强  # 中文注释
        # —— 直接传入被当前版本支持的超参键 ——  # 中文注释
        box=7.5,  # 边框损失权重  # 中文注释
        cls=2.0,  # 分类损失权重  # 中文注释
        dfl=1.0,  # DFL损失权重  # 中文注释
        hsv_h=0.015,  # 色相抖动  # 中文注释
        hsv_s=0.5,    # 饱和度抖动  # 中文注释
        hsv_v=0.5,    # 亮度抖动  # 中文注释
        degrees=0.0,  # 旋转  # 中文注释
        translate=0.08,  # 平移  # 中文注释
        scale=0.3,   # 缩放  # 中文注释
        shear=0.0,   # 切变  # 中文注释
        perspective=0.0,  # 透视  # 中文注释
        flipud=0.0,  # 垂直翻转  # 中文注释
        fliplr=0.5,  # 水平翻转  # 中文注释
        mosaic=0.7,  # 马赛克  # 中文注释
        mixup=0.05,  # 混合  # 中文注释
        copy_paste=0.1,  # 复制粘贴  # 中文注释
        project="runs_stable",  # 输出项目目录  # 中文注释
        name="mf_p2_stage_train",  # 运行名称  # 中文注释
        verbose=True,  # 打印详细日志  # 中文注释
        seed=42,  # 随机种子  # 中文注释
    )

    print(results)  # 打印训练结果统计  # 中文注释


if __name__ == "__main__":  # 启动入口  # 中文注释
    main()  # 调用主函数  # 中文注释


