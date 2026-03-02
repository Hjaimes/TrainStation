"""Z-Image transformer model loader.

Wraps musubi_tuner.zimage.zimage_model for use in the training app.

The Z-Image model forward signature is:
    model(x, t, cap_feats, cap_mask) -> Tensor [B, C, F, H, W]

Where:
    x:         [B, C, F, H, W] — noisy latents (F=1 for images, with dummy frame dim)
    t:         [B] — reversed timesteps in [0, 1], computed as (1000 - raw_t) / 1000
    cap_feats: [B, L, D] — Qwen3-4B text embeddings (cached, not encoded on-the-fly)
    cap_mask:  [B, L] bool — True for valid tokens, or None if split_attn

Unlike Wan (which returns List[Tensor]), ZImageTransformer2DModel returns a single
batched Tensor [B, C, F, H, W].
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from .configs import ZImageConfig

logger = logging.getLogger(__name__)


def detect_zimage_sd_dtype(dit_path: str) -> torch.dtype:
    """Detect the dtype of the DiT checkpoint without loading the full model.

    Peeks at the first tensor in the safetensors file.
    Returns torch.bfloat16 as default if detection fails.
    """
    try:
        from safetensors import safe_open
        with safe_open(dit_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if keys:
                first = f.get_tensor(keys[0])
                return first.dtype
    except Exception as exc:
        logger.warning(
            "Could not detect dtype from '%s': %s. Defaulting to bfloat16.", dit_path, exc
        )
    return torch.bfloat16


def load_zimage_model(
    config: ZImageConfig,
    device: torch.device,
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: torch.device,
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    use_16bit_for_attention: bool = True,
    disable_numpy_memmap: bool = False,
) -> torch.nn.Module:
    """Load a ZImageTransformer2DModel from a safetensors checkpoint.

    Args:
        config:                  ZImageConfig with architecture parameters.
        device:                  Target computation device.
        dit_path:                Path to the .safetensors checkpoint.
        attn_mode:               Attention backend ('torch', 'flash', 'xformers', etc.).
        split_attn:              Use split attention (no attention mask needed).
        loading_device:          Device to load weights to (cpu when using block swap).
        dit_weight_dtype:        Weight dtype; None when fp8_scaled=True.
        fp8_scaled:              Use FP8 scaled weight optimisation.
        use_16bit_for_attention: Use 16-bit precision in attention ops (reduces VRAM).
        disable_numpy_memmap:    Disable numpy memmap when reading safetensors.

    Returns:
        ZImageTransformer2DModel ready for training.
    """
    from musubi_tuner.zimage.zimage_model import load_zimage_model as _load

    logger.info(
        "Loading ZImage model from '%s' (attn=%s, split_attn=%s, dtype=%s, fp8=%s)",
        dit_path, attn_mode, split_attn, dit_weight_dtype, fp8_scaled,
    )

    model = _load(
        device=device,
        dit_path=dit_path,
        attn_mode=attn_mode,
        split_attn=split_attn,
        loading_device=loading_device,
        dit_weight_dtype=dit_weight_dtype,
        fp8_scaled=fp8_scaled,
        use_16bit_for_attention=use_16bit_for_attention,
        disable_numpy_memmap=disable_numpy_memmap,
    )
    return model
