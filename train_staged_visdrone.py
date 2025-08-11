# /workspace/yolo/train_staged_visdrone.py  # 文件路径
# 主要功能：分三阶段训练VisDrone（冻结P2→开启P2+InterpIoU→精修），集成梯度裁剪与NaN保险丝，稳定小目标训练  # 中文注释

import math  # 导入数学库以检测NaN与阈值判断  # 中文注释
import torch  # 导入PyTorch用于梯度裁剪与张量操作  # 中文注释
from ultralytics import YOLO  # 导入Ultralytics YOLO高层API  # 中文注释

# ===== 阶段超参（可与环境变量联动） =====  # 中文注释
STAGE1_EPOCHS = int(__import__("os").environ.get("STAGE1", 20))  # 阶段1：关闭P2、轻增广  # 中文注释
STAGE2_EPOCHS = int(__import__("os").environ.get("STAGE2", 40))  # 阶段2：开启P2、InterpIoU  # 中文注释


# ===== 梯度裁剪回调：每个batch后执行，抑制梯度爆炸 =====  # 中文注释
def on_train_batch_end(trainer):  # Ultralytics回调签名(trainer)  # 中文注释
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=5.0)  # 全局范数裁剪  # 中文注释


# ===== NaN保险丝：每个epoch结尾检查loss，异常则降LR并临时收紧增广 =====  # 中文注释
def on_fit_epoch_end(trainer):  # 训练每个epoch结束时触发  # 中文注释
    # 读取总损失，兼容张量/向量/列表情况，统一转标量  # 中文注释
    tloss = getattr(trainer, "tloss", float("inf"))
    try:
        if isinstance(tloss, (list, tuple)):
            parts = []
            for v in tloss:
                if torch.is_tensor(v):
                    parts.append(v.detach().mean().item())
                else:
                    parts.append(float(v))
            loss_val = float(sum(parts))
        elif torch.is_tensor(tloss):
            loss_val = tloss.detach().mean().item()
        else:
            loss_val = float(tloss)
    except Exception:
        loss_val = float("inf")
    if math.isnan(loss_val) or loss_val > 1e3:  # 检测NaN/异常大损失  # 中文注释
        for g in trainer.optimizer.param_groups:  # 遍历优化器参数组  # 中文注释
            g["lr"] *= 0.2  # 立刻降低学习率  # 中文注释
        # 收紧数据增强以稳定训练（若字段存在则生效）  # 中文注释
        args = trainer.args  # 便捷引用  # 中文注释
        if hasattr(args, "mosaic"):
            args.mosaic = min(getattr(args, "mosaic", 0.7), 0.5)  # 降低马赛克强度  # 中文注释
        if hasattr(args, "mixup"):
            args.mixup = 0.0  # 暂停mixup  # 中文注释
        if hasattr(args, "erasing"):
            args.erasing = 0.0  # 暂停erasing  # 中文注释


# ===== 分阶段启/停P2头：借助你已有的回调实现 =====  # 中文注释
from callbacks.toggle_p2 import on_train_epoch_start  # 按epoch动态屏蔽P2  # 中文注释
from callbacks.early_phase_tweaks import on_train_epoch_end as on_train_epoch_end_tweak  # 早期稳态微调  # 中文注释


# ===== 接入 InterpIoU：在训练开始时对box回归项打补丁 =====  # 中文注释
from ultralytics.nn.modules.losses.interpiou import interpiou_iou_xyxy  # 导入InterpIoU向量化IoU  # 中文注释
from ultralytics.utils.ops import xywh2xyxy  # 导入坐标转换工具  # 中文注释


def patch_bbox_loss(trainer):  # 在trainer构建完成后调用  # 中文注释
    # 替换v8DetectionLoss内的IoU项：将CIoU损失替换为InterpIoU近似  # 中文注释
    criterion = getattr(trainer.model, "criterion", None)  # 训练损失器  # 中文注释
    if criterion is None:  # 若未初始化则跳过，由Ultralytics内部稍后初始化  # 中文注释
        return  # 等到首次调用loss再由内部使用默认构造，但此处无法安全猴子补丁  # 中文注释

    # 早期稳定：减少正样本上限与TAL聚焦，降低分类梯度强度（若可用）  # 中文注释
    try:
        if hasattr(criterion, "assigner"):
            if hasattr(criterion.assigner, "topk"):
                criterion.assigner.topk = 1  # 进一步降到1，极限减压正样本  # 中文注释
            if hasattr(criterion.assigner, "beta"):
                criterion.assigner.beta = 3.0  # 再降低聚焦强度  # 中文注释
            if hasattr(criterion.assigner, "alpha"):
                criterion.assigner.alpha = 0.3  # 再降低分类放大系数  # 中文注释
    except Exception:
        pass  # 忽略不兼容情况  # 中文注释

    # 对 bbox_loss.forward 做包装：计算xyxy后用InterpIoU  # 中文注释
    bbox_loss = getattr(criterion, "bbox_loss", None)  # 读取BboxLoss实例  # 中文注释
    if bbox_loss is None:  # 若结构不同则跳过  # 中文注释
        return  # 安全退出  # 中文注释

    original_forward = bbox_loss.forward  # 备份原forward  # 中文注释

    def forward_with_interpiou(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # Ultralytics默认 pred_bboxes/target_bboxes 为xyxy或xywh依据调用；我们统一转xyxy后计算InterpIoU  # 中文注释
        pred_xyxy = pred_bboxes[fg_mask]  # 选出前景  # 中文注释
        tgt_xyxy = target_bboxes[fg_mask]  # 选出前景  # 中文注释
        # 有些路径传入的是xywh，若检测到x2< x1或y2< y1概率较高，做一次保守转换  # 中文注释
        if (pred_xyxy[..., 2:] < pred_xyxy[..., :2]).any():  # 简单鲁棒性判断  # 中文注释
            pred_xyxy = xywh2xyxy(pred_xyxy)  # 转换到xyxy  # 中文注释
        if (tgt_xyxy[..., 2:] < tgt_xyxy[..., :2]).any():  # 简单鲁棒性判断  # 中文注释
            tgt_xyxy = xywh2xyxy(tgt_xyxy)  # 转换到xyxy  # 中文注释

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # 与原实现一致的权重  # 中文注释
        iou_vec = interpiou_iou_xyxy(pred_xyxy, tgt_xyxy, samples=8)  # 逐样本插值IoU [N]  # 中文注释
        loss_iou = (((1.0 - iou_vec).unsqueeze(-1)) * weight).sum() / target_scores_sum  # 与原实现形式一致  # 中文注释

        # 其余DFL项保持不变：调用原forward拿到DFL并替换IoU  # 中文注释
        loss_iou_orig, loss_dfl = original_forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )  # 原计算  # 中文注释
        return loss_iou, loss_dfl  # 用InterpIoU替换IoU项，保留DFL  # 中文注释

    bbox_loss.forward = forward_with_interpiou  # 覆盖forward  # 中文注释


if __name__ == "__main__":  # 主入口  # 中文注释
    # 自动选择训练设备：有CUDA则用第0块GPU，否则CPU  # 中文注释
    DEVICE = "0" if torch.cuda.is_available() else "cpu"  # 设备字符串  # 中文注释
    # 加载改良版模型YAML  # 中文注释
    model = YOLO("/workspace/yolo/ultralytics/cfg/models/new/yolov12-smallobj-stable.yaml")  # 中文注释

    # 注册通用回调（阶段1/2/3通用）：P2渐进启用、梯度裁剪、NaN保险丝  # 中文注释
    model.add_callback("on_train_epoch_start", on_train_epoch_start)  # 分阶段开启P2  # 中文注释
    model.add_callback("on_train_batch_end", on_train_batch_end)      # 梯度裁剪  # 中文注释
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)          # NaN保险丝  # 中文注释
    model.add_callback("on_train_epoch_end", on_train_epoch_end_tweak)  # 早期动态降权与LR  # 中文注释

    # 阶段1：关闭P2、保守超参（imgsz=640 如你要求）  # 中文注释
    model.train(
        data="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml",  # 数据集YAML  # 中文注释
        epochs=STAGE1_EPOCHS,  # 轮数  # 中文注释
        batch=16,  # 批量大小  # 中文注释
        imgsz=640,  # 输入分辨率  # 中文注释
        optimizer="AdamW",  # 优化器  # 中文注释
        lr0=0.0010,  # 再降LR以抑制CLS  # 中文注释
        weight_decay=0.0005,  # 修复：由0.05降至5e-4  # 中文注释
        warmup_epochs=10,  # 再延长暖起  # 中文注释
        close_mosaic=0,  # 防误解，直接用mosaic概率控制  # 中文注释
        mosaic=0.0, mixup=0.0, erasing=0.0,  # 阶段1彻底停用强增广  # 中文注释
        amp=False,  # 关闭AMP以避免早期数值不稳  # 中文注释
        rect=True,  # 长宽比采样，稳定小目标分布  # 中文注释
        cls=0.03,  # 大幅降低分类损失权重，避免CLS爆炸  # 中文注释
        cos_lr=True,  # 余弦退火  # 中文注释
        device=DEVICE,  # 显式设置训练设备  # 中文注释
        project="runs_stable",  # 日志工程  # 中文注释
        name="stage1_freeze_p2",  # 运行名  # 中文注释
    )

    # 阶段2：开启P2 + 接入InterpIoU + 适度放开增广  # 中文注释
    # 在训练器启动回调中打补丁，确保criterion已构建后替换  # 中文注释
    model.add_callback("on_train_start", lambda tr: patch_bbox_loss(tr))  # 注册补丁回调（训练开始）  # 中文注释
    model.add_callback("on_train_epoch_start", lambda tr: patch_bbox_loss(tr))  # 每轮尝试补丁  # 中文注释
    model.train(
        data="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml",  # 数据集YAML  # 中文注释
        epochs=STAGE2_EPOCHS,  # 轮数  # 中文注释
        batch=16,  # 批量大小  # 中文注释
        imgsz=640,  # 输入分辨率  # 中文注释
        optimizer="AdamW",  # 优化器  # 中文注释
        lr0=0.0015,  # 略降学习率延续稳态  # 中文注释
        weight_decay=0.0005,  # 权重衰减维持稳定  # 中文注释
        warmup_epochs=4,  # 暖起略增  # 中文注释
        mosaic=0.4, mixup=0.02, erasing=0.10,  # 逐步放开但保守  # 中文注释
        amp=False,  # 第二阶段继续禁用AMP，待稳定后再开  # 中文注释
        cos_lr=True,  # 余弦退火  # 中文注释
        device=DEVICE,  # 显式设置训练设备  # 中文注释
        project="runs_stable",  # 日志工程  # 中文注释
        name="stage2_enable_p2_interpiou",  # 运行名  # 中文注释
    )

    # 阶段3：总轮数 - 前两阶段，精修结构（EMA/SAC等在YAML侧控制）  # 中文注释
    stage3_epochs = max(int(__import__("os").environ.get("TOTAL", 300)) - (STAGE1_EPOCHS + STAGE2_EPOCHS), 0)  # 计算余量  # 中文注释
    if stage3_epochs > 0:  # 仅当需继续训练时进入  # 中文注释
        model.train(
            data="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml",  # 数据集YAML  # 中文注释
            epochs=stage3_epochs,  # 轮数  # 中文注释
            batch=16,  # 批量大小  # 中文注释
            imgsz=640,  # 输入分辨率  # 中文注释
            optimizer="AdamW",  # 优化器  # 中文注释
            lr0=0.0012,  # 再小幅降低  # 中文注释
            weight_decay=0.0004,  # 轻微减弱正则  # 中文注释
            warmup_epochs=0,  # 无需额外暖起  # 中文注释
            mosaic=0.4, mixup=0.02, erasing=0.10,  # 稳态增广  # 中文注释
            amp=False,  # 视稳定性再考虑开启AMP  # 中文注释
            cos_lr=True,  # 余弦退火  # 中文注释
            device=DEVICE,  # 显式设置训练设备  # 中文注释
            project="runs_stable",  # 日志工程  # 中文注释
            name="stage3_refine",  # 运行名  # 中文注释
        )


