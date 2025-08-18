# /workspace/yolo/ultralytics/nn/modules/heads_detr_aux.py
# 作用：从单尺度特征（建议P3）产生N个查询的类别+框，供蒸馏/辅助监督使用
# 参考：Deformable-DETR以稀疏采样/多尺度注意力加速收敛，降低小目标检测难度
# 动机：作为辅助查询头用于蒸馏/辅助监督，有助于提升主检测头在拥挤和小目标场景下的召回率
import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    """2D位置编码模块，为特征图添加空间位置信息"""
    def __init__(self, channels: int, max_len: int = 256):
        super().__init__()
        self.max_len = max_len
        # 分别为行和列创建位置嵌入
        self.row_embed = nn.Embedding(max_len, channels // 2)
        self.col_embed = nn.Embedding(max_len, channels // 2)
        
        # 初始化位置编码权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化位置编码权重"""
        for embed in [self.row_embed, self.col_embed]:
            nn.init.uniform_(embed.weight, -1.0, 1.0)
    
    def forward(self, height: int, width: int, device: torch.device):
        """
        生成2D位置编码
        Args:
            height: 特征图高度
            width: 特征图宽度  
            device: 计算设备
        Returns:
            pos: (1, C, H, W) 位置编码张量
        """
        # 创建坐标索引
        i = torch.arange(width, device=device)
        j = torch.arange(height, device=device)
        
        # 生成列和行的位置嵌入
        x_emb = self.col_embed(i)  # (W, C//2)
        y_emb = self.row_embed(j)  # (H, C//2)
        
        # 扩展并连接为完整的2D位置编码
        x_emb = x_emb.unsqueeze(0).repeat(height, 1, 1)  # (H, W, C//2)
        y_emb = y_emb.unsqueeze(1).repeat(1, width, 1)   # (H, W, C//2)
        
        pos = torch.cat([x_emb, y_emb], dim=-1)  # (H, W, C)
        return pos.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

class DETRAuxHead(nn.Module):
    """DETR辅助检测头，用于辅助监督和蒸馏"""
    def __init__(self, input_channels: int, num_queries: int, num_classes: int, 
                 hidden_dim: int = 256, num_heads: int = 8, num_encoder_layers: int = 3):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 查询嵌入（可学习的目标查询）
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 输入特征投影到隐藏维度
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        
        # 2D位置编码
        self.pos_encoder = PositionalEncoding2D(hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 分类头：预测目标类别
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        
        # 回归头：预测边界框坐标(cx, cy, w, h)，归一化到[0,1]
        self.bbox_embed = nn.Linear(hidden_dim, 4)
        
        # 初始化边界框预测偏置为0.5，有助于训练稳定性
        nn.init.constant_(self.bbox_embed.bias, 0.5)
        
        # 初始化其他权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化查询嵌入
        nn.init.xavier_uniform_(self.query_embed.weight)
        
        # 初始化分类头
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0)
        
        # 初始化回归头
        nn.init.xavier_uniform_(self.bbox_embed.weight)
    
    def forward(self, x):
        """
        DETR辅助头前向传播
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            logits: (B, num_queries, num_classes) 分类预测
            boxes: (B, num_queries, 4) 边界框预测
        """
        B, C, H, W = x.shape
        
        # 特征投影并添加位置编码
        features = self.input_proj(x)  # (B, hidden_dim, H, W)
        pos_encoding = self.pos_encoder(H, W, x.device)  # (1, hidden_dim, H, W)
        features = features + pos_encoding
        
        # 将2D特征图展平为序列
        features_flat = features.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        
        # 准备查询嵌入
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, hidden_dim)
        
        # 将查询和特征拼接输入Transformer编码器
        encoder_input = torch.cat([queries, features_flat], dim=1)  # (B, num_queries+H*W, hidden_dim)
        
        # Transformer编码器处理
        encoded_features = self.encoder(encoder_input)
        
        # 只取查询部分的输出
        query_outputs = encoded_features[:, :self.num_queries, :]  # (B, num_queries, hidden_dim)
        
        # 预测分类和边界框
        logits = self.class_embed(query_outputs)  # (B, num_queries, num_classes)
        boxes = self.bbox_embed(query_outputs).sigmoid()  # (B, num_queries, 4) 归一化到[0,1]
        
        return logits, boxes
    
    def compute_auxiliary_loss(self, logits, boxes, targets, matcher=None):
        """
        计算辅助损失，用于训练时的额外监督
        Args:
            logits: (B, num_queries, num_classes) 预测分类
            boxes: (B, num_queries, 4) 预测边界框
            targets: 真实标签列表
            matcher: 匈牙利匹配器（可选）
        Returns:
            loss: 辅助损失值
        """
        # 这里是一个简化版本，实际使用时可以实现完整的匈牙利匹配
        # 目前返回一个基础的L1损失作为占位符
        if targets is None:
            return torch.tensor(0.0, device=logits.device)
        
        # 简单的L1损失（实际应用中应该使用匈牙利匹配）
        loss_cls = nn.CrossEntropyLoss()(logits.view(-1, self.num_classes), 
                                        torch.zeros(logits.size(0) * logits.size(1), 
                                        dtype=torch.long, device=logits.device))
        loss_bbox = nn.L1Loss()(boxes, torch.zeros_like(boxes))
        
        return loss_cls + loss_bbox
