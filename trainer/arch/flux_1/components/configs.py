"""Flux 1 variant configurations.

Flux 1 uses 16-channel latents packed 2x2 to 64-channel, T5+CLIP text encoders,
3D RoPE (16, 56, 56), GEGLU activation, and per-block AdaLayerNorm modulation.
19 double-stream blocks + 38 single-stream blocks, hidden size 3072 (24 heads x 128 dim).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flux1Config:
    """Flux 1 variant configuration."""
    name: str
    num_double_blocks: int    # 19
    num_single_blocks: int    # 38
    hidden_size: int          # 3072
    num_attention_heads: int  # 24
    head_dim: int             # 128
    mlp_ratio: float          # 4.0
    latent_channels: int      # 16
    patch_size: int           # 2
    in_channels: int          # 64  (16 * 2 * 2, after packing)
    context_dim: int          # 4096  (T5-XXL)
    pooled_dim: int           # 768   (CLIP-L)
    rope_axes: tuple[int, int, int]  # (16, 56, 56)
    use_guidance_embed: bool  # True for dev, False for schnell
    activation: str           # "geglu"


FLUX1_CONFIGS: dict[str, Flux1Config] = {
    "dev": Flux1Config(
        name="flux-1-dev",
        num_double_blocks=19,
        num_single_blocks=38,
        hidden_size=3072,
        num_attention_heads=24,
        head_dim=128,
        mlp_ratio=4.0,
        latent_channels=16,
        patch_size=2,
        in_channels=64,
        context_dim=4096,
        pooled_dim=768,
        rope_axes=(16, 56, 56),
        use_guidance_embed=True,
        activation="geglu",
    ),
    "schnell": Flux1Config(
        name="flux-1-schnell",
        num_double_blocks=19,
        num_single_blocks=38,
        hidden_size=3072,
        num_attention_heads=24,
        head_dim=128,
        mlp_ratio=4.0,
        latent_channels=16,
        patch_size=2,
        in_channels=64,
        context_dim=4096,
        pooled_dim=768,
        rope_axes=(16, 56, 56),
        use_guidance_embed=False,
        activation="geglu",
    ),
}
