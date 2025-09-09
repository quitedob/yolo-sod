# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
    A2C2f,
    VimBlock,
    CA_FPN_Block,
    CompactInvertedBlock,
    SimAM,
    FusionLockTSS,
    GlobalDetail,
    DynamicPixelAttn,
    MFBlock,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect, DecoupledHead, SRAHead, MFDecHead
from .smallobj_modules import (
    Add,
    SE,
    SE_Block,
    MixedAttention,
    SpaceToDepth,
    OmniKernelFusion,
    HyperACEBlock,
    DecoupledHeadLite,
)
from .smallobj_modules_stable import (
    FusionLockTSS_Stable,
    HyperACEBlockStable,
)
from .stable_fuse import ChannelNorm, ScaleAdd  # 导入稳定融合/归一化  # 中文注释
from .detect_stable import DetectStable  # 导入可控Detect  # 中文注释
from .recurrent_attention_fusion_block import RecurrentAttentionFusionBlock  # 导入循环-注意力融合模块  # 中文注释
from .braf_block import BiLevelRoutingAttentionFusionBlock  # 导入BRAF模块  # 中文注释
# 导入自定义融合模块  # 中文注释
from .blocks_transformer import SwinBlock  # 导入Swin Transformer模块  # 中文注释
from .heads_detr_aux import DETRAuxHead  # 导入DETR辅助头  # 中文注释
from .ca_block import CA_Block  # 导入坐标注意力模块  # 中文注释
from .a2_attn import A2_Attn  # 导入区域注意力模块  # 中文注释
from .cbam_block import CBAM_Block  # 导入CBAM注意力模块  # 中文注释
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "TorchVision",
    "Index",
    "A2C2f",
    "VimBlock",
    "CA_FPN_Block",
    "CompactInvertedBlock",
    "SimAM",
    "FusionLockTSS",
    "GlobalDetail",
    "DynamicPixelAttn",
    "MFBlock",
    "DecoupledHead",
    "SRAHead",
    "MFDecHead",
    # small object enhancement modules
    "Add",
    "SE",
    "SE_Block",
    "MixedAttention",
    "SpaceToDepth",
    "OmniKernelFusion",
    "HyperACEBlock",
    "DecoupledHeadLite",
    # stable small-object modules
    "FusionLockTSS_Stable",
    "HyperACEBlockStable",
    # stable fuse & detect
    "ChannelNorm",
    "ScaleAdd",
    "DetectStable",
    # RAFB
    "RecurrentAttentionFusionBlock",
    # BRAF
    "BiLevelRoutingAttentionFusionBlock",
    # custom fusion modules
    "SwinBlock",
    "DETRAuxHead",
    # attention modules
    "CA_Block",
    "A2_Attn", 
    "CBAM_Block",
)
