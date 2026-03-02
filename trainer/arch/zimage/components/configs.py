"""Z-Image model configuration constants and presets.

Ported from Musubi_Tuner zimage/zimage_config.py.
All inference-only constants are omitted; only training-relevant constants are kept.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Architecture constants (cached by strategy, not accessed per-step)
# ---------------------------------------------------------------------------

# VAE normalization: latent_model = (vae_latent - shift) * scale
ZIMAGE_VAE_SHIFT_FACTOR: float = 0.1159
ZIMAGE_VAE_SCALING_FACTOR: float = 0.3611

# VAE spatial downscale factor (stride = 8 in each spatial dim)
ZIMAGE_VAE_SCALE_FACTOR: int = 8

# Latent channels produced by the VAE encoder
ZIMAGE_VAE_LATENT_CHANNELS: int = 16

# Sequence padding multiple: total seq len must be a multiple of SEQ_MULTI_OF
SEQ_MULTI_OF: int = 32

# Default transformer patch sizes
DEFAULT_TRANSFORMER_PATCH_SIZE: tuple[int, ...] = (2,)
DEFAULT_TRANSFORMER_F_PATCH_SIZE: tuple[int, ...] = (1,)

# Default transformer architecture dimensions
DEFAULT_TRANSFORMER_IN_CHANNELS: int = 16
DEFAULT_TRANSFORMER_DIM: int = 3840
DEFAULT_TRANSFORMER_N_LAYERS: int = 30
DEFAULT_TRANSFORMER_N_REFINER_LAYERS: int = 2
DEFAULT_TRANSFORMER_N_HEADS: int = 30
DEFAULT_TRANSFORMER_N_KV_HEADS: int = 30
DEFAULT_TRANSFORMER_NORM_EPS: float = 1e-5
DEFAULT_TRANSFORMER_QK_NORM: bool = True
DEFAULT_TRANSFORMER_CAP_FEAT_DIM: int = 2560   # Qwen3-4B hidden dim
DEFAULT_TRANSFORMER_T_SCALE: float = 1000.0

# RoPE configuration
ROPE_THETA: float = 256.0
ROPE_AXES_DIMS: list[int] = [32, 48, 48]
ROPE_AXES_LENS: list[int] = [1536, 512, 512]

# Timestep embedding
ADALN_EMBED_DIM: int = 256
FREQUENCY_EMBEDDING_SIZE: int = 256
MAX_PERIOD: int = 10000

# Default scheduler
DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS: int = 1000
DEFAULT_SCHEDULER_SHIFT: float = 3.0


# ---------------------------------------------------------------------------
# ZImageConfig dataclass — used by strategy.setup() to hold model metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZImageConfig:
    """Immutable configuration for the Z-Image transformer.

    Strategy caches an instance of this in setup() and reads it in training_step()
    via self._zimage_config.
    """
    # Spatial patch size tuple, e.g. (2,)
    patch_size: tuple[int, ...]
    # Temporal patch size tuple, e.g. (1,)
    f_patch_size: tuple[int, ...]
    # Input latent channels
    in_channels: int
    # Transformer hidden dim
    dim: int
    # Number of main transformer layers
    n_layers: int
    # Number of refiner layers (noise + context)
    n_refiner_layers: int
    # Attention heads
    n_heads: int
    n_kv_heads: int
    # RMS norm epsilon
    norm_eps: float
    # Whether to apply QK norm
    qk_norm: bool
    # Caption feature dim (from Qwen3-4B)
    cap_feat_dim: int
    # RoPE theta
    rope_theta: float
    # Timestep scale (model receives t * t_scale)
    t_scale: float
    # RoPE axes dims
    axes_dims: tuple[int, ...]
    # RoPE axes lens
    axes_lens: tuple[int, ...]
    # VAE spatial stride used to compute latent h/w from pixel h/w
    vae_scale_factor: int


# ---------------------------------------------------------------------------
# Default config (only one variant exists for Z-Image)
# ---------------------------------------------------------------------------

ZIMAGE_DEFAULT_CONFIG = ZImageConfig(
    patch_size=DEFAULT_TRANSFORMER_PATCH_SIZE,
    f_patch_size=DEFAULT_TRANSFORMER_F_PATCH_SIZE,
    in_channels=DEFAULT_TRANSFORMER_IN_CHANNELS,
    dim=DEFAULT_TRANSFORMER_DIM,
    n_layers=DEFAULT_TRANSFORMER_N_LAYERS,
    n_refiner_layers=DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
    n_heads=DEFAULT_TRANSFORMER_N_HEADS,
    n_kv_heads=DEFAULT_TRANSFORMER_N_KV_HEADS,
    norm_eps=DEFAULT_TRANSFORMER_NORM_EPS,
    qk_norm=DEFAULT_TRANSFORMER_QK_NORM,
    cap_feat_dim=DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
    rope_theta=ROPE_THETA,
    t_scale=DEFAULT_TRANSFORMER_T_SCALE,
    axes_dims=tuple(ROPE_AXES_DIMS),
    axes_lens=tuple(ROPE_AXES_LENS),
    vae_scale_factor=ZIMAGE_VAE_SCALE_FACTOR,
)

# Registry-style dict for uniform access (only one variant, but mirrors Wan's pattern)
ZIMAGE_CONFIGS: dict[str, ZImageConfig] = {
    "zimage-base": ZIMAGE_DEFAULT_CONFIG,
}
