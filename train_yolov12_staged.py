#!/usr/bin/env python3
"""
YOLOv12-SOD-Fusion-v5 Staged Training Script
Implements the HCP-400 training protocol with 4 stages as described in plan.txt
"""

import argparse
import yaml
import os
import sys
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import LOGGER

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
    
    # ★ 注册新的注意力模块
    try:
        from ultralytics.nn.modules.ca_block import CA_Block
        U.CA_Block = CA_Block
    except Exception as e:
        print(f"[WARN] CA_Block 导入失败：{e}")
    
    try:
        from ultralytics.nn.modules.a2_attn import A2_Attn
        U.A2_Attn = A2_Attn
    except Exception as e:
        print(f"[WARN] A2_Attn 导入失败：{e}")
    
    try:
        from ultralytics.nn.modules.cbam_block import CBAM_Block
        U.CBAM_Block = CBAM_Block
    except Exception as e:
        print(f"[WARN] CBAM_Block 导入失败：{e}")
    
    # ★ 注册 SE_Block 别名
    try:
        from ultralytics.nn.modules.smallobj_modules import SE_Block
        U.SE_Block = SE_Block
    except Exception as e:
        print(f"[WARN] SE_Block 导入失败：{e}")
    
    print("[INFO] 成功注册自定义模块: MambaBlock, SwinBlock, DETRAuxHead, DetectStable, CA_Block, A2_Attn, CBAM_Block, SE_Block 等")

def create_hcp_400_config():
    """Create the HCP-400 staged training configuration"""
    hcp_config = {
        'stage1': {  # Epochs 1-50: Module preheating and stabilization
            'lr0': 0.002,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        },
        'stage2': {  # Epochs 51-150: Global unfreezing and collaborative fine-tuning
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        },
        'stage3': {  # Epochs 151-350: Deep optimization and refinement
            'lr0': 0.0006,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'mosaic': 0.8,  # Reduced mosaic intensity
            'mixup': 0.05,
            'copy_paste': 0.05,
            'degrees': 8.0,
            'translate': 0.08,
            'scale': 0.4,
            'shear': 1.5,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.01,
            'hsv_s': 0.6,
            'hsv_v': 0.3,
        },
        'stage4': {  # Epochs 351-400: Final polishing
            'lr0': 0.0003,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'mosaic': 0.0,  # Disable mosaic for clean data
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 5.0,
            'translate': 0.05,
            'scale': 0.3,
            'shear': 1.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.005,
            'hsv_s': 0.5,
            'hsv_v': 0.2,
        }
    }
    return hcp_config

def create_staged_training_callback(hcp_stages):
    """Create callback for staged training parameter updates"""
    def on_train_epoch_start(trainer):
        epoch = trainer.epoch
        stage_to_apply = None
        
        if epoch == 50:
            stage_to_apply = 'stage2'
        elif epoch == 150:
            stage_to_apply = 'stage3'
        elif epoch == 350:
            stage_to_apply = 'stage4'
        
        if stage_to_apply and stage_to_apply in hcp_stages:
            LOGGER.info(f"\n{'='*60}")
            LOGGER.info(f"🔄 Switching to training stage: {stage_to_apply.upper()}")
            LOGGER.info(f"{'='*60}")
            
            stage_hyp = hcp_stages[stage_to_apply]
            
            # Update trainer arguments
            for k, v in stage_hyp.items():
                if hasattr(trainer.args, k):
                    setattr(trainer.args, k, v)
                    LOGGER.info(f"  ✓ Updated {k}: {v}")
            
            # Update optimizer learning rate
            if 'lr0' in stage_hyp and trainer.optimizer:
                for pg in trainer.optimizer.param_groups:
                    pg['lr'] = stage_hyp['lr0']
                LOGGER.info(f"  ✓ Updated optimizer learning rate: {stage_hyp['lr0']}")
            
            LOGGER.info(f"{'='*60}")
    
    return on_train_epoch_start

def create_p2_toggle_callback(close_p2_until=30):
    """Create callback for P2 layer delayed activation"""
    def on_train_epoch_start(trainer):
        try:
            from ultralytics.nn.modules.detect_stable import DetectStable
            ep = trainer.epoch
            
            for m in trainer.model.modules():
                if isinstance(m, DetectStable):
                    # P2 layer is typically the first detection scale
                    active = [ep >= close_p2_until, True, True, True]  # [P2, P3, P4, P5]
                    m.set_active_mask(active)
            
            if ep == close_p2_until:
                LOGGER.info(f"\n🎯 P2 detection layer activated at epoch {close_p2_until}")
                
        except Exception as e:
            LOGGER.warning(f"⚠️  P2 toggle callback failed: {e}")
    
    return on_train_epoch_start

def main():
    parser = argparse.ArgumentParser(description="YOLOv12-SOD-Fusion-v5 Staged Training Script")
    parser.add_argument('--cfg', default='ultralytics/cfg/models/new/yolov12-sod-fusion-v5.yaml', 
                       help='Model configuration YAML file path')
    parser.add_argument('--data', default='ultralytics/cfg/datasets/visdrone.yaml',
                       help='Dataset configuration YAML file path')
    parser.add_argument('--epochs', type=int, default=400, help='Total training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--device', default='0', help='Training device (e.g., "0" or "cpu")')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--project', default='runs/yolov12_staged', help='Project name for saving results')
    parser.add_argument('--name', default='yolov12_sod_fusion_v5', help='Experiment name')
    parser.add_argument('--close_p2_until', type=int, default=30, help='Close P2 layer until this epoch')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--pretrained', default=None, help='Path to pretrained weights')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 YOLOv12-SOD-Fusion-v5 Staged Training System")
    print("=" * 80)
    print("📋 Integrated Modules:")
    print("   ✨ SE_Block - Squeeze-and-Excitation Attention")
    print("   ✨ CBAM_Block - Convolutional Block Attention Module")
    print("   ✨ CA_Block - Coordinate Attention")
    print("   ✨ A2_Attn - Area Attention")
    print("   ✨ SwinBlock - Swin Transformer")
    print("=" * 80)
    print("📊 Training Configuration:")
    print(f"   📁 Model Config: {args.cfg}")
    print(f"   📁 Dataset Config: {args.data}")
    print(f"   🔢 Total Epochs: {args.epochs}")
    print(f"   📦 Batch Size: {args.batch}")
    print(f"   📏 Image Size: {args.imgsz}")
    print(f"   💻 Device: {args.device}")
    print(f"   🔧 Workers: {args.workers}")
    print(f"   🎯 P2 Activation: Epoch {args.close_p2_until}")
    print("=" * 80)
    
    # Validate files exist
    if not os.path.exists(args.cfg):
        print(f"❌ Model config file not found: {args.cfg}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"❌ Dataset config file not found: {args.data}")
        sys.exit(1)
    
    # Register custom modules before YOLO model initialization
    print("🔧 Registering custom modules...")
    register_custom_modules()
    
    # Initialize YOLO model
    print("🔧 Initializing YOLO model...")
    if args.pretrained and os.path.exists(args.pretrained):
        model = YOLO(args.pretrained)
        print(f"✅ Loaded pretrained weights: {args.pretrained}")
    else:
        model = YOLO(args.cfg)
        print(f"✅ Loaded model configuration: {args.cfg}")
    
    # Create HCP-400 staged training configuration
    hcp_stages = create_hcp_400_config()
    print("✅ Created HCP-400 staged training configuration")
    
    # Register callbacks
    print("🔗 Registering training callbacks...")
    
    # 1. P2 layer delayed activation callback
    if args.close_p2_until > 0:
        model.add_callback("on_train_epoch_start", create_p2_toggle_callback(args.close_p2_until))
        print(f"   ✅ P2 delayed activation (until epoch {args.close_p2_until})")
    
    # 2. Staged training callback
    model.add_callback("on_train_epoch_start", create_staged_training_callback(hcp_stages))
    print("   ✅ HCP-400 staged training protocol")
    
    # Prepare training arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': 50,     # Early stopping patience
        'verbose': True,
        # Initial hyperparameters (Stage 1)
        **hcp_stages['stage1']
    }
    
    print("🚀 Starting staged training...")
    print("📈 Training Stages:")
    print("   🔥 Stage 1 (Epochs 1-50): Module preheating & stabilization")
    print("   🔄 Stage 2 (Epochs 51-150): Global unfreezing & collaborative fine-tuning")
    print("   🎯 Stage 3 (Epochs 151-350): Deep optimization & refinement")
    print("   ✨ Stage 4 (Epochs 351-400): Final polishing")
    print("=" * 80)
    
    try:
        # Start training
        results = model.train(**train_args)
        
        print("=" * 80)
        print("🎉 Training completed successfully!")
        print(f"📊 Results saved to: {args.project}/{args.name}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print("=" * 80)
        print(f"❌ Training failed with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()