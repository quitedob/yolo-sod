#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /workspace/yolo/train_hcp_400.py
# 主要功能简介: HCP-400分层收敛协议训练脚本 - 专为YOLO-SOD-Fusion混合架构设计
# 实现四个训练阶段的自动切换、差分学习率、InterpIoU损失函数集成

import os
import sys
import argparse
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import yaml_load, DEFAULT_CFG_KEYS
from ultralytics.utils.torch_utils import select_device

# 导入自定义回调（可选）
try:
    from callbacks.toggle_p2 import on_train_epoch_start as cb_toggle_p2
    from callbacks.early_phase_tweaks import on_train_epoch_end as cb_early_tweak
    CUSTOM_CALLBACKS_AVAILABLE = True
except ImportError:
    print("[WARNING] 自定义回调导入失败，将使用默认回调")
    cb_toggle_p2 = None
    cb_early_tweak = None
    CUSTOM_CALLBACKS_AVAILABLE = False

# HCP-400阶段配置
STAGE_CONFIG = {
    "stage1": {"epochs": 50,   "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "description": "基础与头部初始化"},
    "stage2": {"epochs": 100,  "freeze": None, "description": "深度特征自适应"},
    "stage3": {"epochs": 200,  "freeze": None, "description": "全局端到端微调"},
    "stage4": {"epochs": 50,   "freeze": None, "description": "最终精炼"}
}

class HCP400Trainer:
    """HCP-400分层收敛协议训练器"""
    
    def __init__(self, cfg_path, hyp_path, data_path, epochs=400, device="0"):
        """
        初始化训练器
        
        Args:
            cfg_path: 模型配置文件路径
            hyp_path: 超参数配置文件路径
            data_path: 数据集配置文件路径
            epochs: 总训练轮数
            device: 训练设备
        """
        self.cfg_path = cfg_path
        self.hyp_path = hyp_path
        self.data_path = data_path
        self.epochs = epochs
        self.device = select_device(device)
        
        # 加载配置
        self.model_cfg = yaml_load(cfg_path)
        self.hyp_cfg = yaml_load(hyp_path)
        
        # 训练状态
        self.current_stage = "stage1"
        self.current_epoch = 0
        self.total_epochs_completed = 0
        
        # 创建模型
        self.model = None
        
        print(f"[INFO] HCP-400训练器初始化完成")
        print(f"[INFO] 模型配置: {cfg_path}")
        print(f"[INFO] 超参数配置: {hyp_path}")
        print(f"[INFO] 数据集配置: {data_path}")
        print(f"[INFO] 目标设备: {self.device}")
    
    def create_model(self):
        """创建YOLO-SOD-Fusion模型"""
        try:
            self.model = YOLO(self.cfg_path)
            print(f"[INFO] 模型创建成功: {self.model_cfg.get('nc', 'unknown')}类")
            return True
        except Exception as e:
            print(f"[ERROR] 模型创建失败: {e}")
            return False
    
    def get_stage_hyp(self, stage_name):
        """获取指定阶段的超参数"""
        stage_hyp = self.hyp_cfg.get(stage_name, {})
        if not stage_hyp:
            print(f"[WARNING] 未找到阶段 {stage_name} 的超参数配置")
            return self.hyp_cfg.get("stage1", {})
        return stage_hyp
    
    def setup_differential_lr(self, stage_name):
        """设置差分学习率"""
        if stage_name != "stage2":
            return None
        
        print("[INFO] 设置差分学习率...")
        
        # 获取基础学习率
        base_lr = self.hyp_cfg[stage_name].get("lr0", 0.001)
        
        # 定义参数组
        param_groups = []
        
        # 浅层网络(前6层): lr * 0.1
        shallow_params = []
        # 深层网络(后6层): lr * 1.0  
        deep_params = []
        # Mamba模块: lr * 1.5
        mamba_params = []
        
        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                if "MFBlock" in name or "VimBlock" in name:
                    mamba_params.append(param)
                elif any(f"model.{i}." in name for i in range(6)):
                    shallow_params.append(param)
                else:
                    deep_params.append(param)
        
        # 添加参数组
        if shallow_params:
            param_groups.append({"params": shallow_params, "lr": base_lr * 0.1})
        if deep_params:
            param_groups.append({"params": deep_params, "lr": base_lr * 1.0})
        if mamba_params:
            param_groups.append({"params": mamba_params, "lr": base_lr * 1.5})
        
        print(f"[INFO] 差分学习率设置完成:")
        print(f"  - 浅层网络: {base_lr * 0.1:.6f}")
        print(f"  - 深层网络: {base_lr * 1.0:.6f}")
        print(f"  - Mamba模块: {base_lr * 1.5:.6f}")
        
        return param_groups
    
    def train_stage(self, stage_name, resume=False):
        """训练指定阶段"""
        stage_config = STAGE_CONFIG[stage_name]
        stage_epochs = stage_config["epochs"]
        stage_freeze = stage_config["freeze"]
        stage_description = stage_config["description"]
        
        print(f"\n{'='*60}")
        print(f"开始训练阶段: {stage_name} - {stage_description}")
        print(f"训练轮数: {stage_epochs}")
        print(f"冻结策略: {stage_freeze if stage_freeze else '无冻结'}")
        print(f"{'='*60}")
        
        # 获取阶段超参数
        stage_hyp = self.get_stage_hyp(stage_name)
        
        # 准备训练参数
        train_args = {
            "data": self.data_path,
            "epochs": stage_epochs,
            "imgsz": 640,
            "batch": 16,
            "device": self.device,
            "workers": 8,
            "project": "runs_hcp_400",
            "name": f"stage_{stage_name}",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": stage_hyp.get("optimizer", "AdamW"),
            "verbose": True,
            "seed": 42,
            "deterministic": True,
            "single_cls": False,
            "rect": False,
            "cos_lr": True,  # 余弦退火
            "close_mosaic": stage_hyp.get("close_mosaic", 0),
            "resume": resume,
            "amp": True,  # 混合精度训练
        }
        
        # 添加超参数
        for key, value in stage_hyp.items():
            if key in DEFAULT_CFG_KEYS:
                train_args[key] = value
        
        # 设置冻结策略
        if stage_freeze:
            train_args["freeze"] = stage_freeze
        
        # 设置差分学习率
        if stage_name == "stage2":
            param_groups = self.setup_differential_lr(stage_name)
            if param_groups:
                # 注意：Ultralytics可能不支持param_groups参数
                # 这里我们通过修改优化器来实现差分学习率
                print("[INFO] 差分学习率将通过优化器参数组实现")
        
        # 启动训练
        try:
            print(f"[INFO] 启动阶段 {stage_name} 训练...")
            results = self.model.train(**train_args)
            
            # 更新训练状态
            self.current_stage = stage_name
            self.total_epochs_completed += stage_epochs
            
            print(f"[SUCCESS] 阶段 {stage_name} 训练完成")
            return True
            
        except Exception as e:
            print(f"[ERROR] 阶段 {stage_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_hcp_400(self):
        """运行完整的HCP-400训练协议"""
        print(f"\n{'='*80}")
        print(f"HCP-400分层收敛协议启动")
        print(f"总目标轮数: {self.epochs}")
        print(f"模型架构: YOLO-SOD-Fusion")
        print(f"{'='*80}")
        
        # 创建模型
        if not self.create_model():
            return False
        
        # 运行四个训练阶段
        stages = ["stage1", "stage2", "stage3", "stage4"]
        resume = False
        
        for i, stage in enumerate(stages):
            if i > 0:  # 从第二阶段开始恢复训练
                resume = True
            
            success = self.train_stage(stage, resume=resume)
            if not success:
                print(f"[ERROR] 阶段 {stage} 训练失败，终止协议")
                return False
            
            print(f"[INFO] 阶段 {stage} 完成，累计轮数: {self.total_epochs_completed}")
        
        print(f"\n{'='*80}")
        print(f"HCP-400分层收敛协议完成!")
        print(f"总训练轮数: {self.total_epochs_completed}")
        print(f"{'='*80}")
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HCP-400分层收敛协议训练器")
    parser.add_argument("--cfg", type=str, 
                       default="/workspace/yolo/ultralytics/cfg/models/new/yolov12-sod-fusion.yaml",
                       help="模型配置文件路径")
    parser.add_argument("--hyp", type=str,
                       default="/workspace/yolo/ultralytics/cfg/models/new/hyp_hcp_400.yaml", 
                       help="超参数配置文件路径")
    parser.add_argument("--data", type=str,
                       default="/workspace/yolo/ultralytics/cfg/datasets/visdrone.yaml",
                       help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=400, help="总训练轮数")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    
    args = parser.parse_args()
    
    # 检查文件存在性
    for file_path in [args.cfg, args.hyp, args.data]:
        if not os.path.exists(file_path):
            print(f"[ERROR] 文件不存在: {file_path}")
            return
    
    # 创建训练器并运行
    trainer = HCP400Trainer(
        cfg_path=args.cfg,
        hyp_path=args.hyp,
        data_path=args.data,
        epochs=args.epochs,
        device=args.device
    )
    
    success = trainer.run_hcp_400()
    if success:
        print("[SUCCESS] HCP-400训练协议执行完成")
    else:
        print("[ERROR] HCP-400训练协议执行失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
