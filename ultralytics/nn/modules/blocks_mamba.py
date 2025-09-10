# /workspace/yolo/ultralytics/nn/modules/blocks_mamba.py
# 作用：在骨干/颈部插入 Mamba 长序列状态空间建模模块，无 mamba-ssm 时回退到GLU门控卷积块
# 参考：Mamba 选择性SSM（线性时空复杂度）
# 动机：高分辨率特征图展平成序列后进行建模，弥补纯注意力在超长序列下的计算瓶颈
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_causal_conv1d_args(x, conv_w, conv_b=None):
    """
    中文注释：确保传给 causal_conv1d 的参数格式正确并返回预处理后的 x, weight, bias
      - 输入 x 期望：(batch, dim, seqlen)（channels-first）
      - conv_w 实际可能来自 nn.Conv1d，形状为 (dim, 1, width)（depthwise conv）或 (dim, width)
      - 本函数不会修改 Module 的真实参数，只会返回适合 native 调用的临时副本
    """
    # 1) 先把 x 统一为 (B, C, L)
    if x.ndim == 3:
        # 如果上游是 (B, L, C) -> permute 到 (B, C, L)
        if x.shape[-1] == conv_w.shape[0] and x.shape[1] != conv_w.shape[0]:
            x = x.permute(0, 2, 1).contiguous()
        # 另一个兜底判断：如果第1维等于 kernel dim，但第2维也是不一致则也尝试 permute
        elif x.shape[1] != conv_w.shape[0] and x.shape[-1] == conv_w.shape[0]:
            x = x.permute(0, 2, 1).contiguous()
    
    # 2) 处理 conv_w：把常见的 (dim,1,width) -> (dim,width)
    if conv_w is None:
        raise RuntimeError("conv1d weight required for native causal_conv1d call")
    
    # 如果 conv_w 是 Conv1d 的 weight，通常 shape 为 (out_ch, in_ch/groups, kernel)
    # 对于 depthwise (groups=out_ch) 常见形状为 (dim, 1, width) -> squeeze 中间维变为 (dim, width)
    if conv_w.dim() == 3 and conv_w.shape[1] == 1:
        conv_w2 = conv_w.squeeze(1).contiguous()  # -> (dim, width)
        print(f"[DEBUG] conv_w squeezed from {conv_w.shape} to {conv_w2.shape}")
        conv_w = conv_w2
    elif conv_w.dim() == 3 and conv_w.shape[1] != 1:
        # 若非标准情况：尝试将 (dim, in_ch, width) -> (dim, width) by merging in_ch if in_ch==1
        # 否则保留原样（因为我们不能随意丢信息）
        try:
            # 如果 in_ch * width can be reshaped to width (edge-case)，否则跳过
            conv_w = conv_w.reshape(conv_w.shape[0], -1).contiguous()
            print(f"[DEBUG] conv_w reshaped to {conv_w.shape}")
        except Exception:
            # 保持原有 conv_w（后面会有 shape 检查，可能导致回退）
            pass
    elif conv_w.dim() == 2:
        # 已是 (dim, width)，一切正常
        conv_w = conv_w.contiguous()
    else:
        # 兜底：尽量转为2D（如果可能）
        try:
            conv_w = conv_w.view(conv_w.shape[0], -1).contiguous()
            print(f"[DEBUG] conv_w viewed to {conv_w.shape}")
        except Exception:
            pass
    
    # 3) dtype/device/contiguous 对齐
    if x.dtype != conv_w.dtype:
        x = x.to(conv_w.dtype)
    if x.device != conv_w.device:
        x = x.to(conv_w.device)
    if not x.is_contiguous():
        x = x.contiguous()
    if not conv_w.is_contiguous():
        conv_w = conv_w.contiguous()
    
    if conv_b is not None:
        if conv_b.dtype != conv_w.dtype:
            conv_b = conv_b.to(conv_w.dtype)
        if conv_b.device != conv_w.device:
            conv_b = conv_b.to(conv_w.device)
    
    # 最后返回准备好的副本（不改 Module 本身）
    return x, conv_w, conv_b

class Conv1x1BN(nn.Sequential):
    """1x1卷积+BatchNorm+SiLU激活的组合模块"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, 1, bias=False),  # 1x1卷积用于通道变换
            nn.BatchNorm2d(c_out),  # 批归一化稳定训练
            nn.SiLU(inplace=True)  # SiLU激活函数
        )

class GLUBlock(nn.Module):
    """GLU门控卷积块，作为Mamba的回退方案"""
    def __init__(self, c: int, expansion: int = 2):
        super().__init__()
        hidden = c * expansion  # 扩展隐藏维度
        self.pw1 = nn.Conv2d(c, hidden * 2, 1, bias=False)  # 点卷积产生门控和内容
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)  # 深度卷积
        self.bn = nn.BatchNorm2d(hidden)  # 批归一化
        self.pw2 = nn.Conv2d(hidden, c, 1, bias=False)  # 输出投影
        self.act = nn.SiLU(inplace=True)  # 激活函数
    
    def forward(self, x):
        # GLU门控机制：将输入分为激活项和门控项
        a, g = self.pw1(x).chunk(2, dim=1)
        x = torch.sigmoid(g) * a  # 门控融合
        x = self.dw(x)  # 深度卷积特征提取
        x = self.bn(x)  # 批归一化
        x = self.act(x)  # 激活
        x = self.pw2(x)  # 输出投影
        return x

class MambaBlock(nn.Module):
    """Mamba状态空间模块，支持线性复杂度的长序列建模"""
    def __init__(self, c: int, c_hidden: int = 256, seq_reduction: int = 2):
        super().__init__()
        self.in_proj = Conv1x1BN(c, c_hidden)  # 输入投影
        self.out_proj = Conv1x1BN(c_hidden, c)  # 输出投影
        self.reduction = seq_reduction  # 序列长度缩减因子，降低计算复杂度
        
        # 智能MambaBlock回退机制 - 使用增强的参数准备函数测试兼容性
        print("Testing MambaBlock compatibility with enhanced argument preparation...")
        self.use_mamba = False  # 默认使用GLU回退
        
        try:
            # 检查版本兼容性
            import mamba_ssm
            import causal_conv1d
            
            print(f"[INFO] Found mamba-ssm version: {getattr(mamba_ssm, '__version__', 'unknown')}")
            print(f"[INFO] Found causal-conv1d version: {getattr(causal_conv1d, '__version__', 'unknown')}")
            
            # 尝试使用增强的参数准备函数进行兼容性测试
            from mamba_ssm import Mamba
            import torch
            
            seq_len = 16  # 使用16的倍数，更兼容内部计算
            # Mamba API 期望 (batch, seq_len, d_model) - sequence-first
            test_input = torch.randn(1, seq_len, c_hidden)  # [batch, seq_len, d_model]
            test_mamba = Mamba(d_model=c_hidden, d_state=16, d_conv=4, expand=1)
            
            print(f"[DEBUG] test_input.shape: {test_input.shape} (batch, seq_len, d_model)")
            
            with torch.no_grad():  # 避免梯度计算
                # 直接调用 Mamba，无需预处理（Mamba 内部会处理格式转换）
                if hasattr(test_mamba, 'conv1d') and hasattr(test_mamba.conv1d, 'weight'):
                    conv_w_raw = test_mamba.conv1d.weight
                    print(f"[DEBUG] conv_w.shape: {conv_w_raw.shape}")
                
                test_output = test_mamba(test_input)
                print(f"[DEBUG] test_output.shape: {test_output.shape}")
                print(f"[DEBUG] Mamba test successful!")
            
            self.use_mamba = True
            self.mamba = test_mamba
            print(f"[INFO] MambaBlock: 成功加载mamba-ssm，使用Mamba SSM进行长序列建模")
            
        except Exception as e:
            # 版本不兼容或其他问题，使用GLU回退
            error_msg = str(e)
            print(f"[DEBUG] Full error message: {error_msg}")
            
            if "causal_conv1d_fwd" in error_msg or "incompatible function arguments" in error_msg:
                print(f"[WARNING] MambaBlock: causal_conv1d版本不兼容，回退到GLU门控卷积")
            elif "mat1 and mat2 shapes" in error_msg:
                print(f"[WARNING] MambaBlock: 张量维度不匹配，回退到GLU门控卷积")
                print(f"[DEBUG] 具体错误: {error_msg}")
            else:
                print(f"[WARNING] MambaBlock: 初始化失败 ({error_msg[:200]}...)，回退到GLU门控卷积")
            
            self.use_mamba = False
            self.fallback = GLUBlock(c_hidden, expansion=2)
            print(f"[INFO] 使用GLU门控卷积作为MambaBlock的高效替代方案")
    
    def forward(self, x):
        B, C, H, W = x.shape
        # 输入投影到隐藏维度
        y = self.in_proj(x)
        
        # 如果设置了缩减因子，先进行下采样减少序列长度
        if self.reduction > 1:
            y = F.avg_pool2d(y, self.reduction, self.reduction)
        
        Bh, Ch, Hh, Wh = y.shape
        
        if self.use_mamba:
            # Mamba需要 (batch, seq_len, dim) 格式（注意：Mamba API 使用 seq-first）
            # 将2D特征图展平并转为 (B, L, C)
            y_mamba = y.flatten(2).permute(0, 2, 1).contiguous()  # [B, L, C]
            
            try:
                # 先尝试把内部的 conv1d weight/bias做预处理（不修改模块原参）
                if hasattr(self.mamba, 'conv1d') and hasattr(self.mamba.conv1d, 'weight'):
                    conv_w_raw = self.mamba.conv1d.weight
                    conv_b_raw = getattr(self.mamba.conv1d, 'bias', None)
                    try:
                        # 注意：_ensure_causal_conv1d_args 期望 x 是 (B, C, L) -> 但这里 x 是 (B, L, C)
                        # 所以对 Mamba 我们不把 y_mamba_prepped 传回 mamba（避免 double-permute）。
                        # 仅用于检查/打印以便调试
                        # 以下仅打印 shapes for debugging (safe)
                        print(f"[DEBUG] before calling Mamba: y_mamba.shape={y_mamba.shape}, conv_w.shape={conv_w_raw.shape}")
                    except Exception:
                        pass
                
                # 调用 Mamba（Mamba期望 (B, L, C)）
                y_mamba_out = self.mamba(y_mamba)  # 输出应该仍为 (B, L, C)
                # 把 Mamba 输出转回 (B, C, L) 以便 reshape 成 (B,C,H,W)
                y_mamba = y_mamba_out.permute(0, 2, 1).contiguous()  # -> (B, C, L)
                # 恢复二维尺寸
                y = y_mamba.reshape(Bh, Ch, Hh, Wh)
                
            except Exception as e:
                # 运行时回退到GLU（详细调试）
                import traceback
                print("===== MAMBA FORWARD ERROR DEBUG =====")
                try:
                    print(f"[DEBUG] y_mamba.shape: {y_mamba.shape}")
                    if hasattr(self.mamba, 'conv1d') and hasattr(self.mamba.conv1d, 'weight'):
                        print(f"[DEBUG] conv1d.weight.shape: {self.mamba.conv1d.weight.shape}")
                        print(f"[DEBUG] conv1d.weight.dtype: {self.mamba.conv1d.weight.dtype}")
                        print(f"[DEBUG] y_mamba.dtype: {y_mamba.dtype}")
                        print(f"[DEBUG] y_mamba.device: {y_mamba.device}")
                        print(f"[DEBUG] conv1d.weight.device: {self.mamba.conv1d.weight.device}")
                except:
                    pass
                print(f"[DEBUG] Exception: {repr(e)}")
                traceback.print_exc()
                print("=====================================")
                print(f"[WARNING] Mamba forward failed at runtime, falling back to GLU")
                
                if not hasattr(self, 'runtime_fallback'):
                    self.runtime_fallback = GLUBlock(Ch, expansion=2).to(y.device)
                y = self.runtime_fallback(y)
        else:
            # 使用GLU回退方案
            y = self.fallback(y)
        
        # 如果进行了下采样，需要上采样回原尺寸
        if self.reduction > 1:
            y = F.interpolate(y, size=(H, W), mode='nearest')
        
        # 输出投影并添加残差连接
        y = self.out_proj(y)
        return x + y  # 残差连接保持特征稳定性
