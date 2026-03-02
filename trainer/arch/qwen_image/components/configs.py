"""QwenImage model configuration presets.

Three modes are supported:
    t2i     — text-to-image (default)
    edit    — image editing with control image(s)
    layered — layered image generation (multi-layer)

Model architecture is fixed: 60 transformer blocks, 24 heads × 128 dim = 3072 inner_dim,
joint_attention_dim = 3584 (Qwen2.5-VL embedding dimension), in_channels = 64 (patchified).
"""
from __future__ import annotations

from types import SimpleNamespace


def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_shared = _ns(
    # Transformer architecture (fixed for all modes)
    patch_size=2,
    in_channels=64,         # 16 latent channels × 2×2 patch = 64 after pixel-unshuffle
    out_channels=16,
    num_layers=60,
    attention_head_dim=128,
    num_attention_heads=24,
    joint_attention_dim=3584,  # Qwen2.5-VL hidden size
    axes_dims_rope=(16, 56, 56),
    guidance_embeds=False,
    # VAE
    vae_scale_factor=8,
    latent_channels=16,
    # Training defaults
    train_dtype="bf16",
    sample_steps=20,
    discrete_flow_shift=3.0,
)


def _make_config(name: str, **overrides) -> SimpleNamespace:
    import copy
    cfg = copy.deepcopy(_shared)
    cfg.__name__ = name
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Mode configs
# ---------------------------------------------------------------------------

# Standard text-to-image
t2i = _make_config(
    "Config: QwenImage T2I",
    mode="t2i",
    zero_cond_t=False,
    use_additional_t_cond=False,
    use_layer3d_rope=False,
)

# Image editing — uses control images concatenated in sequence dimension
edit = _make_config(
    "Config: QwenImage Edit",
    mode="edit",
    zero_cond_t=False,
    use_additional_t_cond=False,
    use_layer3d_rope=False,
)

# Edit-2511 variant — uses zero conditioning for time (zero_cond_t=True)
edit_2511 = _make_config(
    "Config: QwenImage Edit-2511",
    mode="edit",
    zero_cond_t=True,
    use_additional_t_cond=False,
    use_layer3d_rope=False,
)

# Layered generation — uses 3D Layer RoPE and additional time conditioning
layered = _make_config(
    "Config: QwenImage Layered",
    mode="layered",
    zero_cond_t=False,
    use_additional_t_cond=True,   # is_rgb embedding for layer type
    use_layer3d_rope=True,
)

# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

QWEN_IMAGE_CONFIGS: dict[str, SimpleNamespace] = {
    "t2i": t2i,
    "edit": edit,
    "edit-2511": edit_2511,
    "layered": layered,
}
