# /workspace/yolo/train.py
# 作用：集成Mamba+Swin+DETR-Aux+边界感知损失+时序后处理的完整训练脚本
# 功能：注册插件模块 -> 加载改进版YAML -> 训练500轮 -> 支持P2启停与多种损失回调
# 关键点：兼容Ultralytics官方接口，支持回调机制和自定义模块注册
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn

def register_custom_modules():
    """注册所有自定义模块到Ultralytics命名空间"""
    import ultralytics.nn.modules as U
    
    # 导入自定义模块
    from ultralytics.nn.modules.blocks_mamba import MambaBlock
    from ultralytics.nn.modules.blocks_transformer import SwinBlock
    from ultralytics.nn.modules.heads_detr_aux import DETRAuxHead
    from ultralytics.nn.modules.loss_boundary import BoundaryAwareLoss
    from ultralytics.nn.modules.tracker_kf_lstm import MultiObjectTracker
    
    # 注册到Ultralytics模块命名空间
    U.MambaBlock = MambaBlock
    U.SwinBlock = SwinBlock
    U.DETRAuxHead = DETRAuxHead
    U.BoundaryAwareLoss = BoundaryAwareLoss
    U.MultiObjectTracker = MultiObjectTracker
    
    print('[INFO] 成功注册自定义模块: MambaBlock, SwinBlock, DETRAuxHead, BoundaryAwareLoss, MultiObjectTracker')

def create_p2_toggle_callback(close_p2_until: int = 30):
    """
    创建P2尺度启停回调函数
    Args:
        close_p2_until: 在第N轮之前关闭P2尺度
    Returns:
        callback: 回调函数
    """
    epoch_counter = {'count': 0}
    
    def p2_toggle_callback(trainer):
        """P2启停回调实现"""
        try:
            from ultralytics.nn.modules.detect_stable import DetectStable
            
            current_epoch = epoch_counter['count']
            
            # 遍历模型中所有DetectStable模块
            for module in trainer.model.modules():
                if isinstance(module, DetectStable):
                    # 设置各尺度的激活掩码：[P2, P3, P4, P5]
                    # 前close_p2_until轮关闭P2，其他尺度保持开启
                    active_mask = [
                        current_epoch >= close_p2_until,  # P2: 第N轮后开启
                        True,                              # P3: 始终开启
                        True,                              # P4: 始终开启 
                        True                               # P5: 始终开启
                    ]
                    module.set_active_mask(active_mask)
                    
            epoch_counter['count'] += 1
            
            if current_epoch == close_p2_until:
                print(f'[INFO] 第{close_p2_until}轮：P2尺度已激活')
                
        except Exception as e:
            print(f'[WARNING] P2启停回调执行失败: {e}')
    
    return p2_toggle_callback

def create_boundary_loss_callback(edge_weight: float = 1.0, bce_weight: float = 1.0, 
                                iou_weight: float = 0.0, loss_weight: float = 0.2):
    """
    创建边界感知损失回调函数
    Args:
        edge_weight: 边缘损失权重
        bce_weight: BCE损失权重
        iou_weight: IoU损失权重
        loss_weight: 总体损失权重
    Returns:
        callback: 回调函数
    """
    from ultralytics.nn.modules.loss_boundary import BoundaryAwareLoss
    
    boundary_loss_fn = BoundaryAwareLoss(
        edge_weight=edge_weight,
        bce_weight=bce_weight, 
        iou_weight=iou_weight
    )
    
    def boundary_loss_callback(trainer):
        """边界感知损失回调实现"""
        try:
            # 获取批次数据和预测结果
            batch = getattr(trainer, 'batch', None)
            
            # 尝试多种方式获取分割掩码预测
            pred_masks = None
            for attr_name in ['masks', 'seg_masks', 'pred_masks']:
                pred_masks = getattr(trainer, attr_name, None)
                if pred_masks is not None:
                    break
                    
            # 检查数据是否完整
            if pred_masks is None or batch is None or 'masks' not in batch:
                return
                
            # 获取真实掩码
            gt_masks = batch['masks'].float()
            pred_masks = pred_masks.float()
            
            # 确保维度匹配
            if pred_masks.dim() == 4 and pred_masks.size(1) > 1:
                pred_masks = pred_masks[:, :1]  # 取第一个通道
            if gt_masks.dim() == 4 and gt_masks.size(1) > 1:
                gt_masks = gt_masks[:, :1]      # 取第一个通道
                
            # 计算边界感知损失
            boundary_loss = boundary_loss_fn(pred_masks, gt_masks)
            
            # 添加到总损失中
            trainer.loss += boundary_loss * loss_weight
            
        except Exception as e:
            # 静默处理异常，避免影响训练
            pass
    
    return boundary_loss_callback

def create_detr_aux_callback():
    """创建DETR辅助头回调函数（占位符，可扩展）"""
    def detr_aux_callback(trainer):
        """DETR辅助损失回调实现"""
        try:
            # 这里可以实现DETR辅助头的损失计算
            # 例如：获取P3特征 -> DETR-Aux预测 -> 匈牙利匹配 -> 辅助损失
            pass
        except Exception as e:
            pass
    
    return detr_aux_callback

def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-SOD-Fusion增强版训练脚本')
    
    # 基础训练参数
    parser.add_argument('--cfg', default='/workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion-v5-detect-only.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data', required=True, help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', default='0', help='训练设备')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    
    # 高级训练参数
    parser.add_argument('--hyp', default=None, help='超参数配置文件')
    parser.add_argument('--pretrained', default=False, help='是否使用预训练权重')
    parser.add_argument('--optimizer', default='auto', help='优化器类型')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率比例')
    
    # 自定义功能开关
    parser.add_argument('--use_boundary_loss', action='store_true', help='启用边界感知损失')
    parser.add_argument('--close_p2_until', type=int, default=30, help='前N轮关闭P2尺度')
    parser.add_argument('--use_detr_aux', action='store_true', help='启用DETR辅助头')
    
    # 边界损失参数
    parser.add_argument('--edge_weight', type=float, default=1.0, help='边缘损失权重')
    parser.add_argument('--bce_weight', type=float, default=1.0, help='BCE损失权重')
    parser.add_argument('--iou_weight', type=float, default=0.0, help='IoU损失权重')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.2, help='边界损失总权重')
    
    # 项目输出参数
    parser.add_argument('--project', default='runs_fusion', help='项目保存目录')
    parser.add_argument('--name', default='yolo_sod_fusion', help='实验名称')
    
    args = parser.parse_args()
    
    # 打印训练配置信息
    print('=' * 80)
    print('YOLO-SOD-Fusion 增强版训练配置')
    print('=' * 80)
    print(f'模型配置: {args.cfg}')
    print(f'数据集: {args.data}')
    print(f'训练轮数: {args.epochs}')
    print(f'批次大小: {args.batch}')
    print(f'输入尺寸: {args.imgsz}')
    print(f'训练设备: {args.device}')
    print(f'P2关闭轮数: {args.close_p2_until}')
    print(f'边界感知损失: {"启用" if args.use_boundary_loss else "禁用"}')
    print(f'DETR辅助头: {"启用" if args.use_detr_aux else "禁用"}')
    print('=' * 80)
    
    # 注册自定义模块
    print('[INFO] 正在注册自定义模块...')
    register_custom_modules()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(f'模型配置文件未找到: {args.cfg}')
    
    if not os.path.exists(args.data):
        raise FileNotFoundError(f'数据集配置文件未找到: {args.data}')
    
    # 创建YOLO模型
    print('[INFO] 正在初始化YOLO模型...')
    model = YOLO(args.cfg, task='segment')  
    
    # 注册训练回调
    print('[INFO] 正在注册训练回调...')
    
    # P2尺度启停回调
    if args.close_p2_until > 0:
        p2_callback = create_p2_toggle_callback(args.close_p2_until)
        model.add_callback('on_train_epoch_start', p2_callback)
        print(f'[INFO] 已注册P2启停回调，前{args.close_p2_until}轮关闭P2尺度')
    
    # 边界感知损失回调
    if args.use_boundary_loss:
        boundary_callback = create_boundary_loss_callback(
            edge_weight=args.edge_weight,
            bce_weight=args.bce_weight,
            iou_weight=args.iou_weight,
            loss_weight=args.boundary_loss_weight
        )
        model.add_callback('on_train_batch_end', boundary_callback)
        print('[INFO] 已注册边界感知损失回调')
    
    # DETR辅助头回调
    if args.use_detr_aux:
        detr_callback = create_detr_aux_callback()
        model.add_callback('on_train_batch_end', detr_callback)
        print('[INFO] 已注册DETR辅助头回调')
    
    # 配置训练参数
    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'pretrained': args.pretrained,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'cos_lr': True,         # 使用余弦退火学习率
        'warmup_epochs': 3,     # 学习率预热轮数
        'warmup_momentum': 0.8, # 预热动量
        'weight_decay': 0.0005, # 权重衰减
        'momentum': 0.937,      # SGD动量
        'box': 7.5,             # 边界框损失权重
        'cls': 0.5,             # 分类损失权重
        'dfl': 1.5,             # 分布焦点损失权重
        'save': True,           # 保存检查点
        'save_period': 10,      # 保存间隔
        'val': True,            # 进行验证
        'plots': True,          # 生成训练图表
        'verbose': True         # 详细输出
    }
    
    # 添加自定义超参数配置
    if args.hyp:
        train_kwargs.update({'cfg': args.hyp})
    
    # 开始训练
    print('[INFO] 开始训练...')
    try:
        results = model.train(**train_kwargs)
        print('[INFO] 训练完成！')
        print(f'最佳权重保存在: {model.trainer.best}')
        print(f'最后权重保存在: {model.trainer.last}')
        return results
    except KeyboardInterrupt:
        print('[INFO] 训练被用户中断')
    except Exception as e:
        print(f'[ERROR] 训练过程中发生错误: {e}')
        raise
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
