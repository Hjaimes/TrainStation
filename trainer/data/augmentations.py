"""Spatial augmentations applied to cached latent tensors.

All functions operate on tensors that are already in latent space, so pixel
dimensions are divided by the VAE spatial compression factor before computing
shift magnitudes. Both 4-D image [B, C, H, W] and 5-D video [B, C, F, H, W]
tensors are supported.
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def apply_crop_jitter(
    latents: Tensor,
    jitter_pixels: int,
    vae_scale_factor: int = 8,
) -> Tensor:
    """Random spatial roll of latents to simulate crop augmentation.

    Uses ``torch.roll`` so there is no padding artefact — latents wrap around.
    This matches the Musubi_Tuner approach while being safe for all bucket sizes.

    Args:
        latents: Float tensor of shape [B, C, H, W] or [B, C, F, H, W].
        jitter_pixels: Maximum pixel-space shift in each spatial direction.
            The equivalent shift in latent space is ``jitter_pixels // vae_scale_factor``.
        vae_scale_factor: Spatial compression ratio of the VAE (default 8).

    Returns:
        Tensor of the same shape and dtype as ``latents``.
    """
    max_shift = jitter_pixels // vae_scale_factor
    if max_shift <= 0:
        return latents

    shift_h = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    shift_w = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())

    # dims=-2 is H, dims=-1 is W — valid for both 4-D and 5-D tensors.
    return torch.roll(latents, shifts=(shift_h, shift_w), dims=(-2, -1))


def apply_random_flip(
    latents: Tensor,
    probability: float = 0.5,
) -> Tensor:
    """Horizontal flip on the last spatial dimension.

    The flip decision is made independently for each item in the batch so that
    augmentation diversity is maximised even when batch_size > 1.

    Args:
        latents: Float tensor of shape [B, C, H, W] or [B, C, F, H, W].
        probability: Probability of flipping each sample.

    Returns:
        Tensor of the same shape and dtype as ``latents`` with a random subset
        of samples flipped along the width axis (dim=-1).
    """
    if probability <= 0.0:
        return latents

    batch_size = latents.shape[0]

    # Build a boolean mask for which samples to flip — one draw per sample.
    flip_mask = torch.rand(batch_size) < probability  # shape [B]

    if not flip_mask.any():
        return latents

    # Clone only the samples that will be flipped to avoid in-place aliasing.
    result = latents.clone()
    indices = flip_mask.nonzero(as_tuple=False).squeeze(1)  # [N]
    result[indices] = torch.flip(result[indices], dims=[-1])
    return result
