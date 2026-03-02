"""HunyuanVideo 1.5 architecture configuration.

Key differences from HunyuanVideo (original):
- 54 double-stream blocks (vs 20 in HV)
- 0 single-stream blocks (HV has 40)
- patch_size = [1, 1, 1] — no spatial/temporal patching
- guidance_embed = False — no guidance embedding
- Text encoders: Qwen2.5-VL + ByT5 (vs LLM + CLIP-L in HV)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HV15ModelConfig:
    """Immutable configuration for HunyuanVideo 1.5 transformer.

    All values are fixed architecture constants — not user-tunable hyperparams.
    """
    # Block counts
    num_double_blocks: int = 54
    num_single_blocks: int = 0       # HV 1.5 has no single-stream blocks

    # Patch size: [temporal, height, width] — no patching at all
    patch_size: list[int] = field(default_factory=lambda: [1, 1, 1])

    # Latent channels for I/O (out_channels = in_channels - 1 effectively; model sees concat)
    in_channels: int = 32            # 16 latent + 16 cond_latent + 1 mask = 33 total input, but model patchifies; actual in_channels=32 (see PatchEmbed doubling)
    out_channels: int = 32           # output VAE latent channels (= VAE_LATENT_CHANNELS=16 * 2 unpatchify)
    latent_channels: int = 16        # raw VAE latent channel count

    # Model dimension
    hidden_size: int = 2048
    heads_num: int = 16
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"

    # QK-norm settings
    qk_norm: bool = True
    qk_norm_type: str = "rms"
    qkv_bias: bool = True

    # Guidance embedding — DISABLED for HV 1.5
    guidance_embed: bool = False

    # Text encoder dimensions
    text_states_dim: int = 3584      # Qwen2.5-VL output dim
    byt5_dim: int = 1472             # ByT5 output dim

    # Vision states (SigLIP, for I2V)
    vision_states_dim: int = 1152

    # RoPE parameters — 3D positional encoding
    rope_dim_list: list[int] = field(default_factory=lambda: [16, 56, 56])
    rope_theta: float = 256.0

    # Attention
    use_attention_mask: bool = True

    # SingleTokenRefiner depth for text
    text_refiner_depth: int = 2


# The single canonical config for HV 1.5.
HV15_CONFIG = HV15ModelConfig()
