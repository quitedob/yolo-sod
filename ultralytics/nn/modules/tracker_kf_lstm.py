# /workspace/yolo/ultralytics/nn/modules/tracker_kf_lstm.py
# 作用：对每帧检测结果进行预测->匹配->更新，平滑抖动并降低ID Switch（DeepSORT思路）
# 参考：DeepSORT证明了"检测→跟踪"范式中结合卡尔曼滤波和匈牙利算法的鲁棒性
# 动机：在视频中稳定目标ID，抑制检测结果闪烁，并用LSTM预测位移来平滑短期遮挡
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

# 尝试导入卡尔曼滤波库
try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KalmanFilter = None
    KALMAN_AVAILABLE = False
    print("[WARNING] filterpy未安装，卡尔曼滤波功能将被禁用")

# 尝试导入scipy用于匈牙利算法
try:
    from scipy.optimize import linear_sum_assignment
    HUNGARIAN_AVAILABLE = True
except ImportError:
    linear_sum_assignment = None
    HUNGARIAN_AVAILABLE = False
    print("[WARNING] scipy未安装，将使用贪心匹配算法")

@dataclass
class Track:
    """跟踪轨迹数据结构"""
    id: int                    # 轨迹ID
    bbox: np.ndarray          # 边界框 [x1, y1, x2, y2]
    score: float              # 置信度得分
    class_id: int             # 类别ID
    kf: Optional[object]      # 卡尔曼滤波器
    age: int = 0              # 轨迹年龄（总帧数）
    hits: int = 0             # 命中次数（检测到的帧数）
    time_since_update: int = 0 # 自上次更新以来的帧数
    motion_history: List[np.ndarray] = None  # 运动历史用于LSTM

    def __post_init__(self):
        if self.motion_history is None:
            self.motion_history = []

class MotionLSTM(nn.Module):
    """LSTM运动预测模块，用于预测目标下一帧的位移"""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层用于序列建模
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # 输出层预测下一帧的位移
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
    def forward(self, sequence):
        """
        预测下一帧的边界框
        Args:
            sequence: (batch_size, seq_len, 4) 历史边界框序列
        Returns:
            prediction: (batch_size, 4) 下一帧边界框预测
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(sequence)
        
        # 取最后时刻的输出进行预测
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        prediction = self.output_layer(last_output)
        
        return prediction

class MultiObjectTracker:
    """多目标跟踪器，集成卡尔曼滤波、匈牙利匹配和LSTM运动预测"""
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3, 
                 use_motion_lstm: bool = True, lstm_sequence_length: int = 5):
        self.tracks: List[Track] = []         # 当前跟踪轨迹列表
        self.next_id = 1                      # 下一个轨迹ID
        self.max_age = max_age                # 轨迹最大存活帧数
        self.iou_threshold = iou_threshold    # IoU匹配阈值
        self.use_motion_lstm = use_motion_lstm
        self.lstm_sequence_length = lstm_sequence_length
        
        # 初始化LSTM运动预测模块
        if self.use_motion_lstm:
            self.motion_lstm = MotionLSTM()
            self.motion_lstm.eval()  # 设为评估模式
            
    def _create_kalman_filter(self, bbox: np.ndarray) -> Optional[KalmanFilter]:
        """
        为新轨迹创建卡尔曼滤波器
        状态向量：[cx, cy, w, h, vx, vy, vw, vh] (中心坐标, 尺寸, 速度)
        """
        if not KALMAN_AVAILABLE:
            return None
            
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 状态转移矩阵：位置 = 位置 + 速度 * 时间间隔
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i + 4] = 1.0
            
        # 观测矩阵：只能观测到位置和尺寸，不能直接观测速度
        kf.H = np.zeros((4, 8))
        for i in range(4):
            kf.H[i, i] = 1.0
            
        # 过程噪声协方差矩阵
        kf.Q *= 0.01
        
        # 观测噪声协方差矩阵
        kf.R *= 1.0
        
        # 初始状态不确定性
        kf.P *= 10.0
        
        # 初始化状态向量
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        kf.x[:4] = np.array([cx, cy, w, h]).reshape((4, 1))
        
        return kf
    
    def _bbox_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的IoU"""
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算并集
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _hungarian_match(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """使用匈牙利算法进行最优匹配"""
        if HUNGARIAN_AVAILABLE and cost_matrix.size > 0:
            # 使用scipy的匈牙利算法
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matches = list(zip(row_indices, col_indices))
            
            # 过滤低质量匹配
            filtered_matches = []
            for row, col in matches:
                if cost_matrix[row, col] < (1 - self.iou_threshold):
                    filtered_matches.append((row, col))
                    
            # 计算未匹配的轨迹和检测
            matched_tracks = [m[0] for m in filtered_matches]
            matched_detections = [m[1] for m in filtered_matches]
            unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matched_tracks]
            unmatched_detections = [i for i in range(cost_matrix.shape[1]) if i not in matched_detections]
            
            return filtered_matches, unmatched_tracks, unmatched_detections
        else:
            # 使用贪心匹配作为后备方案
            return self._greedy_match(cost_matrix)
    
    def _greedy_match(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """贪心匹配算法"""
        matches = []
        used_tracks = set()
        used_detections = set()
        
        if cost_matrix.size == 0:
            return matches, list(range(len(self.tracks))), []
        
        # 按IoU降序排列所有可能的匹配
        potential_matches = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                iou = 1 - cost_matrix[i, j]
                if iou >= self.iou_threshold:
                    potential_matches.append((i, j, iou))
        
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        # 贪心选择
        for track_idx, det_idx, iou in potential_matches:
            if track_idx not in used_tracks and det_idx not in used_detections:
                matches.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_detections.add(det_idx)
        
        # 计算未匹配项
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in used_tracks]
        unmatched_detections = [i for i in range(cost_matrix.shape[1]) if i not in used_detections]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _predict_with_lstm(self, track: Track) -> Optional[np.ndarray]:
        """使用LSTM预测轨迹下一帧位置"""
        if not self.use_motion_lstm or len(track.motion_history) < 2:
            return None
            
        # 准备序列数据
        sequence_length = min(self.lstm_sequence_length, len(track.motion_history))
        sequence = np.array(track.motion_history[-sequence_length:])
        
        # 转换为tensor并添加batch维度
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # LSTM预测
        with torch.no_grad():
            prediction = self.motion_lstm(sequence_tensor)
            predicted_bbox = prediction.squeeze(0).numpy()
            
        return predicted_bbox
    
    def predict(self):
        """预测所有轨迹的下一帧状态"""
        for track in self.tracks:
            # 使用卡尔曼滤波预测
            if track.kf is not None:
                track.kf.predict()
                
                # 从卡尔曼滤波状态提取边界框
                cx, cy, w, h = track.kf.x[:4, 0]
                predicted_bbox = np.array([
                    cx - w/2, cy - h/2,
                    cx + w/2, cy + h/2
                ])
                
                # 如果启用LSTM，结合LSTM预测结果
                if self.use_motion_lstm:
                    lstm_prediction = self._predict_with_lstm(track)
                    if lstm_prediction is not None:
                        # 加权融合卡尔曼和LSTM预测结果
                        predicted_bbox = 0.7 * predicted_bbox + 0.3 * lstm_prediction
                
                track.bbox = predicted_bbox
            
            # 更新轨迹状态
            track.age += 1
            track.time_since_update += 1
    
    def update(self, detections: List[Tuple[np.ndarray, float, int]]) -> List[Track]:
        """
        更新跟踪器状态
        Args:
            detections: [(bbox, score, class_id), ...] 检测结果列表
        Returns:
            valid_tracks: 有效轨迹列表
        """
        # 预测阶段
        self.predict()
        
        if len(detections) == 0:
            # 没有检测结果，只更新现有轨迹
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= 3 and t.time_since_update <= 3]
        
        # 构建代价矩阵（基于IoU）
        if len(self.tracks) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, (det_bbox, _, _) in enumerate(detections):
                    iou = self._bbox_iou(track.bbox, det_bbox)
                    cost_matrix[i, j] = 1 - iou  # 转换为代价（越小越好）
        else:
            cost_matrix = np.array([])
        
        # 匹配阶段
        matches, unmatched_tracks, unmatched_detections = self._hungarian_match(cost_matrix)
        
        # 更新匹配的轨迹
        for track_idx, det_idx in matches:
            det_bbox, det_score, det_class = detections[det_idx]
            track = self.tracks[track_idx]
            
            # 更新卡尔曼滤波器
            if track.kf is not None:
                cx = (det_bbox[0] + det_bbox[2]) / 2
                cy = (det_bbox[1] + det_bbox[3]) / 2
                w = det_bbox[2] - det_bbox[0]
                h = det_bbox[3] - det_bbox[1]
                track.kf.update(np.array([cx, cy, w, h]))
                
                # 从更新后的状态提取边界框
                cx, cy, w, h = track.kf.x[:4, 0]
                track.bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            else:
                track.bbox = det_bbox
            
            # 更新轨迹属性
            track.score = det_score
            track.class_id = det_class
            track.hits += 1
            track.time_since_update = 0
            
            # 更新运动历史
            track.motion_history.append(track.bbox.copy())
            if len(track.motion_history) > self.lstm_sequence_length + 5:
                track.motion_history.pop(0)
        
        # 为未匹配的检测创建新轨迹
        for det_idx in unmatched_detections:
            det_bbox, det_score, det_class = detections[det_idx]
            kf = self._create_kalman_filter(det_bbox)
            
            new_track = Track(
                id=self.next_id,
                bbox=det_bbox,
                score=det_score,
                class_id=det_class,
                kf=kf,
                motion_history=[det_bbox.copy()]
            )
            
            self.tracks.append(new_track)
            self.next_id += 1
        
        # 删除过期轨迹
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # 返回稳定轨迹（命中次数>=3且最近被更新）
        valid_tracks = [t for t in self.tracks if t.hits >= 3 and t.time_since_update <= 3]
        return valid_tracks
    
    def get_track_results(self) -> List[dict]:
        """获取当前所有轨迹的结果"""
        results = []
        for track in self.tracks:
            if track.hits >= 3 and track.time_since_update <= 3:
                results.append({
                    'track_id': track.id,
                    'bbox': track.bbox.tolist(),
                    'score': track.score,
                    'class_id': track.class_id
                })
        return results
