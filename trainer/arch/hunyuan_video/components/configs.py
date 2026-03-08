"""HunyuanVideo model configuration.

The canonical model is HYVideo-T/2-cfgdistill:
  - 20 MMDoubleStreamBlocks
  - 40 MMSingleStreamBlocks
  - hidden_size=3072, heads_num=24
  - guidance_embed=True (cfg-distilled variant)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class HunyuanVideoConfig:
    """Architecture configuration for HunyuanVideo transformer."""

    # Transformer dimensions
    hidden_size: int = 3072
    heads_num: int = 24
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"

    # Block counts - 20 double + 40 single is the standard T/2 model
    mm_double_blocks_depth: int = 20
    mm_single_blocks_depth: int = 40

    # RoPE dimension list: [time_dim, h_dim, w_dim], must sum to head_dim
    # head_dim = hidden_size // heads_num = 3072 // 24 = 128
    # 16 + 56 + 56 = 128
    rope_dim_list: List[int] = field(default_factory=lambda: [16, 56, 56])

    # Patch size: (time, height, width) - spatial-only patching
    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])

    # Input/output channels - 16 for the 884-16c-hy VAE
    in_channels: int = 16
    out_channels: int = 16

    # Attention & normalization
    qkv_bias: bool = True
    qk_norm: bool = True
    qk_norm_type: str = "rms"

    # Text encoder dimensions
    # LLM (Llama) hidden state dim
    text_states_dim: int = 4096
    # CLIP pooled projection dim
    text_states_dim_2: int = 768

    # Text projection type: "linear" or "single_refiner"
    text_projection: str = "single_refiner"
    use_attention_mask: bool = True

    # Guidance embedding (CFG-distilled variant)
    guidance_embed: bool = True

    # RoPE theta
    rope_theta: float = 256.0


# The single supported config variant
HUNYUAN_VIDEO_CONFIG = HunyuanVideoConfig()
