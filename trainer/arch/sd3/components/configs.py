"""SD3 variant configurations.

Each SD3Config specifies the transformer architecture dimensions for a particular
model variant (SD3-medium, SD3.5-medium, SD3.5-large).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SD3Config:
    """SD3 variant configuration."""
    name: str
    num_layers: int                        # joint transformer blocks
    num_single_layers: int                 # single (image-only) blocks (0 for SD3.0)
    hidden_size: int                       # 1536 for medium, 2048 for large
    num_attention_heads: int
    patch_size: int                        # 2
    latent_channels: int                   # 16
    vae_scaling_factor: float              # 1.5305
    vae_shift_factor: float                # 0.0609
    pooled_projection_dim: int             # 2048 (CLIP-L 768 + CLIP-G 1280)
    caption_projection_dim: int            # 4096 (T5-XXL)
    joint_attention_dim: int               # 4096 (T5-XXL hidden dim)
    dual_attention_layers: tuple[int, ...] | None  # which layers have dual attention (SD3.5)


SD3_CONFIGS: dict[str, SD3Config] = {
    "sd3-medium": SD3Config(
        name="sd3-medium",
        num_layers=24,
        num_single_layers=0,
        hidden_size=1536,
        num_attention_heads=24,
        patch_size=2,
        latent_channels=16,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        pooled_projection_dim=2048,
        caption_projection_dim=4096,
        joint_attention_dim=4096,
        dual_attention_layers=None,
    ),
    "sd3.5-medium": SD3Config(
        name="sd3.5-medium",
        num_layers=24,
        num_single_layers=12,
        hidden_size=1536,
        num_attention_heads=24,
        patch_size=2,
        latent_channels=16,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        pooled_projection_dim=2048,
        caption_projection_dim=4096,
        joint_attention_dim=4096,
        dual_attention_layers=tuple(range(24)),
    ),
    "sd3.5-large": SD3Config(
        name="sd3.5-large",
        num_layers=38,
        num_single_layers=12,
        hidden_size=2048,
        num_attention_heads=32,
        patch_size=2,
        latent_channels=16,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        pooled_projection_dim=2048,
        caption_projection_dim=4096,
        joint_attention_dim=4096,
        dual_attention_layers=tuple(range(38)),
    ),
}
