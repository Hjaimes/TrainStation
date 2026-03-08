"""Flux Kontext variant configurations.

Flux.1-Kontext-dev is the only known variant. Architecture dimensions are taken
from the ``configs_flux_dev_context`` spec in Musubi_Tuner's flux_models.py.

Key differences from Flux 2:
- 64 in_channels (16-channel latents packed 2×2 = 64)
- 3-axis position IDs (not 4-axis)
- axes_dim sums to 128 (16+56+56) - head_dim = hidden_size // num_heads = 128
- qkv_bias = True (original Flux uses biased QKV)
- vec_in_dim = 768 (CLIP-L pooled embed, not Mistral3/Qwen3)
- context_in_dim = 4096 (T5-XXL hidden dim)
- guidance_embed = True (distilled, always use guidance_vec = 1.0 for training)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FluxKontextVariantConfig:
    """Architecture dimensions for a Flux Kontext transformer variant.

    All variants share the original Flux attention mechanism (3-axis RoPE,
    QKV bias, CLIP vector conditioning) and differ only in block counts.
    """

    # Variant identity
    variant: str                          # e.g. "dev"

    # Packed latent channels: 16-ch latents packed 2×2 → 64 channels
    in_channels: int                      # 64

    # Text encoder dims
    context_in_dim: int                   # T5-XXL hidden dim (4096)
    vec_in_dim: int                       # CLIP-L pooled dim (768)

    # Transformer dims
    hidden_size: int                      # transformer hidden dim
    num_heads: int                        # number of attention heads
    depth: int                            # number of DoubleStreamBlocks
    depth_single_blocks: int              # number of SingleStreamBlocks
    axes_dim: tuple[int, ...]             # per-axis RoPE dims (must sum to hidden_size // num_heads)
    theta: int                            # RoPE base frequency
    mlp_ratio: float                      # MLP expansion ratio

    # Architecture flags
    qkv_bias: bool                        # True for original Flux (unlike Flux 2)
    guidance_embed: bool                  # True for dev (guidance-distilled)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

# Flux.1-Kontext-dev:
#   hidden 3072, 24 heads → head_dim 128 → axes_dim must sum to 128.
#   Original Flux axes_dim = [16, 56, 56] → sum = 128.
FLUX_KONTEXT_DEV = FluxKontextVariantConfig(
    variant="dev",
    in_channels=64,
    context_in_dim=4096,
    vec_in_dim=768,
    hidden_size=3072,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=(16, 56, 56),
    theta=10_000,
    mlp_ratio=4.0,
    qkv_bias=True,
    guidance_embed=True,
)


# ---------------------------------------------------------------------------
# Lookup table: model_version -> FluxKontextVariantConfig
# ---------------------------------------------------------------------------

FLUX_KONTEXT_CONFIGS: dict[str, FluxKontextVariantConfig] = {
    "dev": FLUX_KONTEXT_DEV,
}
