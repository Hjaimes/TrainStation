"""Flux 2 variant configurations.

Ported from Musubi_Tuner flux_2/flux2_models.py and flux_2/flux2_utils.py.
Each variant is a dataclass specifying the transformer architecture dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Flux2VariantConfig:
    """Architecture dimensions for a Flux 2 transformer variant.

    Fields mirror Flux2Params / Klein*Params from Musubi_Tuner, but expressed as
    an immutable config object for clean lookup at training time.
    """

    # Variant identity
    variant: str                          # e.g. "dev", "klein-4b"

    # Transformer dims
    in_channels: int                      # packed latent channels (always 128)
    context_in_dim: int                   # text-encoder hidden dim (15360 Mistral / 12288 or 7680 Qwen3)
    hidden_size: int                      # transformer hidden dim
    num_heads: int                        # number of attention heads
    depth: int                            # number of DoubleStreamBlocks
    depth_single_blocks: int              # number of SingleStreamBlocks
    axes_dim: tuple[int, ...]             # per-axis RoPE dims (must sum to hidden_size // num_heads)
    theta: int                            # RoPE base frequency
    mlp_ratio: float                      # MLP expansion ratio

    # Guidance
    use_guidance_embed: bool              # True for dev (guidance-distilled)

    # Text encoder selector - None = Mistral3, "4B" / "8B" = Qwen3 variant
    qwen_variant: str | None              # None for Mistral3, "4B" or "8B" for Qwen3


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

# Dev (FLUX.2-dev): large guidance-distilled model, uses Mistral3 text encoder.
# Hidden 6144, 48 heads → head_dim 128 → axes_dim sums to 128.
FLUX2_DEV = Flux2VariantConfig(
    variant="dev",
    in_channels=128,
    context_in_dim=15360,
    hidden_size=6144,
    num_heads=48,
    depth=8,
    depth_single_blocks=48,
    axes_dim=(32, 32, 32, 32),
    theta=2000,
    mlp_ratio=3.0,
    use_guidance_embed=True,
    qwen_variant=None,
)

# Klein 9B (distilled, guidance-embedded=False): uses Qwen3-8B text encoder.
FLUX2_KLEIN_9B = Flux2VariantConfig(
    variant="klein-9b",
    in_channels=128,
    context_in_dim=12288,
    hidden_size=4096,
    num_heads=32,
    depth=8,
    depth_single_blocks=24,
    axes_dim=(32, 32, 32, 32),
    theta=2000,
    mlp_ratio=3.0,
    use_guidance_embed=False,
    qwen_variant="8B",
)

# Klein 9B base (not distilled, CFG at inference): same arch as klein-9b.
FLUX2_KLEIN_BASE_9B = Flux2VariantConfig(
    variant="klein-base-9b",
    in_channels=128,
    context_in_dim=12288,
    hidden_size=4096,
    num_heads=32,
    depth=8,
    depth_single_blocks=24,
    axes_dim=(32, 32, 32, 32),
    theta=2000,
    mlp_ratio=3.0,
    use_guidance_embed=False,
    qwen_variant="8B",
)

# Klein 4B (distilled): uses Qwen3-4B text encoder.
FLUX2_KLEIN_4B = Flux2VariantConfig(
    variant="klein-4b",
    in_channels=128,
    context_in_dim=7680,
    hidden_size=3072,
    num_heads=24,
    depth=5,
    depth_single_blocks=20,
    axes_dim=(32, 32, 32, 32),
    theta=2000,
    mlp_ratio=3.0,
    use_guidance_embed=False,
    qwen_variant="4B",
)

# Klein 4B base (not distilled): same arch as klein-4b.
FLUX2_KLEIN_BASE_4B = Flux2VariantConfig(
    variant="klein-base-4b",
    in_channels=128,
    context_in_dim=7680,
    hidden_size=3072,
    num_heads=24,
    depth=5,
    depth_single_blocks=20,
    axes_dim=(32, 32, 32, 32),
    theta=2000,
    mlp_ratio=3.0,
    use_guidance_embed=False,
    qwen_variant="4B",
)


# ---------------------------------------------------------------------------
# Lookup table: model_version -> Flux2VariantConfig
# Matches FLUX2_MODEL_INFO keys in Musubi_Tuner flux2_utils.py.
# ---------------------------------------------------------------------------

FLUX2_CONFIGS: dict[str, Flux2VariantConfig] = {
    "dev": FLUX2_DEV,
    "klein-9b": FLUX2_KLEIN_9B,
    "klein-base-9b": FLUX2_KLEIN_BASE_9B,
    "klein-4b": FLUX2_KLEIN_4B,
    "klein-base-4b": FLUX2_KLEIN_BASE_4B,
}
