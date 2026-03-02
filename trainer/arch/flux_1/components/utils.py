"""Flux 1 packing/unpacking and position-ID utilities.

Key differences from Flux 2:
- Pack 16ch latents with 2x2 spatial patches -> 64ch (not 128ch from 128ch latents)
- 3D position IDs (channel, y, x) — not 4D (t, h, w, l) as in Flux 2
- No temporal dimension (image-only model)
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor


def pack_latents(x: Tensor) -> Tensor:
    """Pack 2x2 spatial patches: (B, 16, H, W) -> (B, (H/2)*(W/2), 64).

    Packs 2x2 spatial patches into the channel dimension. Input must have
    spatial dims divisible by 2. The packed channel dim is 16*2*2=64.
    """
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_latents(x: Tensor, h: int, w: int) -> Tensor:
    """Unpack: (B, HW, 64) -> (B, 16, H, W) where h, w are packed spatial dims (H/2, W/2).

    Args:
        x: Packed sequence of shape (B, h*w, 64).
        h: Packed spatial height (latent_H / 2).
        w: Packed spatial width  (latent_W / 2).

    Returns:
        (B, 16, H*2, W*2) latent tensor.
    """
    return rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)


def prepare_img_ids(h: int, w: int) -> Tensor:
    """Create 3D position IDs for image tokens: (1, h*w, 3).

    Dimensions: [channel_idx (always 0), y_pos, x_pos].
    h, w are the packed spatial dimensions (latent_H/2, latent_W/2).

    Returns:
        Float tensor of shape (1, h*w, 3).
    """
    ys = torch.arange(h).unsqueeze(1).expand(h, w).reshape(-1)
    xs = torch.arange(w).unsqueeze(0).expand(h, w).reshape(-1)
    ids = torch.stack([torch.zeros_like(ys), ys, xs], dim=-1)  # (h*w, 3)
    return ids.unsqueeze(0).float()  # (1, h*w, 3)


def prepare_txt_ids(seq_len: int) -> Tensor:
    """Create 3D position IDs for text tokens: (1, seq_len, 3). All zeros.

    Text tokens do not have spatial positions — all zeros signals to RoPE
    that they share the same "origin" position.

    Returns:
        Float tensor of shape (1, seq_len, 3).
    """
    return torch.zeros(1, seq_len, 3)
