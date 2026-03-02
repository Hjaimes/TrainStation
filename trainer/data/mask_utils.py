"""Mask loading and normalization utilities for masked training.

Masks are stored as safetensors alongside latent caches:
  {name}_mask.safetensors containing key "mask" with shape (1, H_lat, W_lat)
  or (1, F, H_lat, W_lat) for video.

Values in [0, 1] where 1 = train on this region, 0 = ignore this region.
"""
from __future__ import annotations

import torch
from torch import Tensor


def load_mask(path: str) -> Tensor:
    """Load a mask from a safetensors file.

    Args:
        path: Path to {name}_mask.safetensors file.

    Returns:
        Mask tensor, shape (1, H, W) or (1, F, H, W), values in [0, 1].
    """
    from safetensors.torch import load_file
    data = load_file(path)
    mask = data["mask"]
    return mask.clamp(0.0, 1.0)


def normalize_mask(mask: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    """Resize mask to match latent spatial dimensions.

    Uses nearest-neighbor interpolation to avoid blurring mask edges.

    Args:
        mask: Input mask, shape (1, H, W) or (1, F, H, W).
        target_shape: Target spatial shape (H_target, W_target) for images
                      or (F_target, H_target, W_target) for video.

    Returns:
        Resized mask matching target_shape.
    """
    if mask.shape[1:] == target_shape:
        return mask

    if mask.ndim == 3:
        # Image: (1, H, W) -> (1, 1, H, W) for interpolate
        m = mask.unsqueeze(0)
        m = torch.nn.functional.interpolate(m, size=target_shape, mode="nearest")
        return m.squeeze(0)
    elif mask.ndim == 4:
        # Video: (1, F, H, W) -> interpolate spatial dims only.
        # Reshape to (F, 1, H, W) for interpolation.
        m = mask.squeeze(0).unsqueeze(1)  # (F, 1, H, W)
        m = torch.nn.functional.interpolate(m, size=target_shape[-2:], mode="nearest")
        m = m.squeeze(1).unsqueeze(0)  # back to (1, F, H_new, W_new)
        return m
    else:
        raise ValueError(f"Unsupported mask ndim: {mask.ndim}. Expected 3 (image) or 4 (video).")
