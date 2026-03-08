"""Z-Image VAE - encode-side only for latent caching during training.

The Z-Image VAE uses a non-standard normalization:
    model_latent = (vae_latent - shift) * scale
    shift = 0.1159, scale = 0.3611

Unlike Wan (which uses HunyuanVideo's 3D VAE), Z-Image uses a standard 2D VAE
(AutoencoderKL) with 16 latent channels.

Only the encode path is ported here; decode is inference-only.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import torch
import torch.nn as nn

from .configs import ZIMAGE_VAE_SHIFT_FACTOR, ZIMAGE_VAE_SCALING_FACTOR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization helpers - applied AFTER encoding with the raw VAE
# ---------------------------------------------------------------------------

def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
    """Apply Z-Image latent normalization: (latents - shift) * scale.

    This converts raw VAE encoder outputs to the distribution the DiT expects.
    Always applied at training time after encode().

    Args:
        latents: Raw VAE latent tensor, any shape.

    Returns:
        Normalized latent tensor, same shape.
    """
    return (latents - ZIMAGE_VAE_SHIFT_FACTOR) * ZIMAGE_VAE_SCALING_FACTOR


def denormalize_latents(latents: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize_latents: latents / scale + shift.

    Used when passing latents to the VAE decoder (inference-only).

    Args:
        latents: Normalized latent tensor.

    Returns:
        Raw VAE latent tensor.
    """
    return latents / ZIMAGE_VAE_SCALING_FACTOR + ZIMAGE_VAE_SHIFT_FACTOR


# ---------------------------------------------------------------------------
# VAE loader
# ---------------------------------------------------------------------------

def load_zimage_vae(
    vae_path: str,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Load the Z-Image AutoencoderKL from a checkpoint.

    The VAE is loaded in float32 for numerical precision, placed on the
    specified device. Caller is responsible for moving to GPU if needed.

    Args:
        vae_path: Path to the .safetensors VAE checkpoint.
        device:   Device to place the loaded VAE on.

    Returns:
        AutoencoderKL in eval mode, float32.
    """
    from musubi_tuner.zimage.zimage_autoencoder import load_autoencoder_kl

    logger.info("Loading Z-Image VAE from '%s'", vae_path)
    vae = load_autoencoder_kl(vae_path, device=device, disable_mmap=False)
    vae.eval()
    return vae


# ---------------------------------------------------------------------------
# Encode helper - used by latent caching pipeline
# ---------------------------------------------------------------------------

def encode_pixels_to_latents(
    vae: nn.Module,
    pixel_values: torch.Tensor,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """Encode pixel images to normalized Z-Image latents.

    The pipeline:
      1. Pass pixels through VAE encoder -> DiagonalGaussianDistribution
      2. Sample (or take mode) from the distribution
      3. Apply Z-Image normalization: (latent - shift) * scale

    Args:
        vae:          Loaded AutoencoderKL in float32.
        pixel_values: [B, 3, H, W] float tensor in [-1, 1].
        sample_mode:  "sample" to draw from distribution, "mode" for deterministic.

    Returns:
        Normalized latents [B, 16, H//8, W//8] in float32.
    """
    with torch.no_grad():
        posterior = vae.encode(pixel_values.to(dtype=torch.float32))
        if sample_mode == "sample":
            latents = posterior.sample()
        else:
            latents = posterior.mode()

    return normalize_latents(latents)
