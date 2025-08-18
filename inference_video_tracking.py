# /workspace/yolo/inference_video_tracking.py
# 作用：视频推理脚本，集成时序后处理（卡尔曼滤波+匈牙利匹配+LSTM）
# 功能：稳定轨迹ID、降低闪烁、平滑目标运动预测
# 应用场景：视频目标检测与跟踪，适用于监控、自动驾驶等场景
import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
import time
from ultralytics import YOLO

def register_custom_modules():
    """注册自定义模块"""
    try:
        import ultralytics.nn.modules as U
        from ultralytics.nn.modules.blocks_mamba import MambaBlock
        from ultralytics.nn.modules.blocks_transformer import SwinBlock
        from ultralytics.nn.modules.heads_detr_aux import DETRAuxHead
        from ultralytics.nn.modules.tracker_kf_lstm import MultiObjectTracker
        
        U.MambaBlock = MambaBlock
        U.SwinBlock = SwinBlock
        U.DETRAuxHead = DETRAuxHead
        U.MultiObjectTracker = MultiObjectTracker
        
        print('[INFO] 自定义模块注册成功')
        return True
    except Exception as e:
        print(f'[WARNING] 自定义模块注册失败: {e}')
        return False

def draw_tracking_results(frame, tracks, class_names=None):
    """
    在帧上绘制跟踪结果
    Args:
        frame: 视频帧
        tracks: 跟踪结果列表
        class_names: 类别名称列表
    Returns:
        frame: 绘制后的帧
    """
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
    ]
    
    for track_result in tracks:
        track_id = track_result['track_id']
        bbox = track_result['bbox']
        score = track_result['score']
        class_id = track_result['class_id']
        
        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, bbox)
        
        # 选择颜色（基于轨迹ID）
        color = colors[track_id % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f'Class{class_id}'
        label = f'ID:{track_id} {class_name} {score:.2f}'
        
        # 计算标签背景尺寸
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # 绘制标签背景
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def process_video_with_tracking(model_path, video_path, output_path=None, 
                               max_age=30, iou_threshold=0.3, confidence_threshold=0.5):
    """
    处理视频并进行多目标跟踪
    Args:
        model_path: 模型权重路径
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        max_age: 轨迹最大存活帧数
        iou_threshold: IoU匹配阈值
        confidence_threshold: 检测置信度阈值
    """
    # 注册自定义模块
    register_custom_modules()
    
    # 加载模型
    print(f'[INFO] 正在加载模型: {model_path}')
    model = YOLO(model_path)
    
    # 获取类别名称
    class_names = model.names if hasattr(model, 'names') else None
    
    # 初始化多目标跟踪器
    from ultralytics.nn.modules.tracker_kf_lstm import MultiObjectTracker
    tracker = MultiObjectTracker(
        max_age=max_age,
        iou_threshold=iou_threshold,
        use_motion_lstm=True,
        lstm_sequence_length=5
    )
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'无法打开视频文件: {video_path}')
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'[INFO] 视频信息: {width}x{height}, {fps}FPS, {total_frames}帧')
    
    # 初始化视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f'[INFO] 输出视频将保存到: {output_path}')
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO检测
            results = model(frame, verbose=False)
            
            # 提取检测结果
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # 过滤低置信度检测
                    if conf >= confidence_threshold:
                        detections.append((bbox, float(conf), cls))
            
            # 多目标跟踪更新
            valid_tracks = tracker.update(detections)
            track_results = tracker.get_track_results()
            
            # 在帧上绘制跟踪结果
            output_frame = draw_tracking_results(frame.copy(), track_results, class_names)
            
            # 添加统计信息
            info_text = [
                f'Frame: {frame_count}/{total_frames}',
                f'Detections: {len(detections)}',
                f'Active Tracks: {len(track_results)}',
                f'FPS: {frame_count / (time.time() - start_time):.1f}'
            ]
            
            for i, text in enumerate(info_text):
                y_pos = 30 + i * 25
                cv2.rectangle(output_frame, (10, y_pos - 20), (300, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(output_frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 保存或显示帧
            if writer is not None:
                writer.write(output_frame)
            
            # 实时显示（可选，用于调试）
            if not output_path:
                cv2.imshow('YOLO Tracking', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 打印进度
            if frame_count % 30 == 0:
                progress = frame_count / total_frames * 100
                avg_fps = frame_count / (time.time() - start_time)
                print(f'[INFO] 处理进度: {progress:.1f}%, 平均FPS: {avg_fps:.1f}')
    
    finally:
        # 释放资源
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        # 输出统计信息
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f'[INFO] 处理完成')
        print(f'[INFO] 总帧数: {frame_count}, 总时间: {total_time:.2f}秒')
        print(f'[INFO] 平均处理速度: {avg_fps:.2f} FPS')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO视频跟踪推理脚本')
    
    # 必需参数
    parser.add_argument('--model', required=True, help='模型权重路径')
    parser.add_argument('--video', required=True, help='输入视频路径')
    
    # 可选参数
    parser.add_argument('--output', default=None, help='输出视频路径（不指定则实时显示）')
    parser.add_argument('--max_age', type=int, default=30, help='轨迹最大存活帧数')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='IoU匹配阈值')
    parser.add_argument('--confidence', type=float, default=0.5, help='检测置信度阈值')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.model).exists():
        raise FileNotFoundError(f'模型文件未找到: {args.model}')
    
    if not Path(args.video).exists():
        raise FileNotFoundError(f'视频文件未找到: {args.video}')
    
    # 执行视频跟踪处理
    try:
        process_video_with_tracking(
            model_path=args.model,
            video_path=args.video,
            output_path=args.output,
            max_age=args.max_age,
            iou_threshold=args.iou_threshold,
            confidence_threshold=args.confidence
        )
    except KeyboardInterrupt:
        print('\n[INFO] 用户中断处理')
    except Exception as e:
        print(f'[ERROR] 处理过程中发生错误: {e}')
        raise

if __name__ == '__main__':
    main()
