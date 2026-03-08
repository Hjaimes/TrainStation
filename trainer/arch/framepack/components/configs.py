"""FramePack architecture configuration.

FramePack uses HunyuanVideo as the base architecture, always in I2V mode.
The packed format uses multi-scale temporal context (1x/2x/4x clean latents).
"""
from __future__ import annotations

from types import SimpleNamespace


def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# FramePack model configuration (single variant - always I2V HunyuanVideo)
# ---------------------------------------------------------------------------

# Architecture dimensions from load_packed_model in Musubi_Tuner:
#   attention_head_dim=128, num_attention_heads=24, num_layers=20,
#   num_single_layers=40, num_refiner_layers=2, patch_size=2, patch_size_t=1
#   rope_axes_dim=(16, 56, 56), rope_theta=256.0
#   text_embed_dim=4096, pooled_projection_dim=768
#   image_proj_dim=1152 (SigLIP image encoder)

FRAMEPACK_CONFIG = _ns(
    # Architecture identity
    architecture="framepack",
    # DiT dimensions
    num_attention_heads=24,
    attention_head_dim=128,
    inner_dim=24 * 128,           # 3072
    # Transformer depth
    num_layers=20,                 # double-stream blocks
    num_single_layers=40,          # single-stream blocks
    num_refiner_layers=2,          # text token refiner
    mlp_ratio=4.0,
    # Patch embedding
    patch_size=2,                  # spatial patch size (height/width)
    patch_size_t=1,                # temporal patch size
    # VAE latent channels
    in_channels=16,
    out_channels=16,
    # Text encoder
    text_embed_dim=4096,           # LLaMA hidden size
    pooled_projection_dim=768,     # CLIP-L pooler size
    # Image encoder (SigLIP)
    image_proj_dim=1152,
    # RoPE
    rope_axes_dim=(16, 56, 56),
    rope_theta=256.0,
    # Normalization
    qk_norm="rms_norm",
    # Mixed precision (fixed for FramePack)
    vae_dtype="float16",
    dit_dtype="bfloat16",
    # Temporal context packing: 1x, 2x, 4x clean latents are packed with noise
    clean_latents_1x_count=1,      # one 1x context frame (the image frame)
    clean_latents_2x_count=2,      # two 2x context frames
    clean_latents_4x_count=16,     # sixteen 4x context frames
    # Default latent window for auto-regressive generation
    latent_window_size=9,
    # VAE spatial stride (for H/W compression)
    vae_stride=(4, 8, 8),          # (temporal, height, width)
    # Guidance embedding
    guidance_embeds=True,
    default_guidance_scale=10.0,
)


FRAMEPACK_CONFIGS: dict[str, SimpleNamespace] = {
    "framepack": FRAMEPACK_CONFIG,
}
