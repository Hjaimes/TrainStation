"""Flux 2 packing/unpacking and position-ID utilities.

Ported from Musubi_Tuner flux_2/flux2_utils.py with the following improvements:
- print() replaced with logger calls
- logging.basicConfig() removed
- torch.concat → torch.cat
- Dead/commented-out code removed
- Use set() for O(1) checks where appropriate
"""
from __future__ import annotations

import logging

import torch
from einops import rearrange
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image packing (spatial H×W → sequence)
# ---------------------------------------------------------------------------

def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Pack a 4-D latent tensor into the sequence format expected by Flux 2.

    Args:
        x: Shape ``(C, H, W)`` or ``(B, C, H, W)`` - 128-channel latents.
        t_coord: Optional 1-D int tensor of time-coordinate values (length 1 for
            a single frame). Defaults to ``torch.arange(1)``.

    Returns:
        Tuple of:
        - ``x_packed``: ``(HW, C)`` or ``(B, HW, C)`` - flattened spatial tokens.
        - ``x_ids``:    ``(HW, 4)`` or ``(B, HW, 4)`` - per-token 4-D position IDs
          with layout ``(t, h, w, l)``.
    """
    h = x.shape[-2]
    w = x.shape[-1]

    t_coords = torch.arange(1, device=x.device) if t_coord is None else t_coord
    x_ids = torch.cartesian_prod(
        t_coords,
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        torch.arange(1, device=x.device),
    )  # (H*W, 4)

    if x.ndim == 3:  # single sample (C, H, W)
        x_packed = rearrange(x, "c h w -> (h w) c")
    else:  # batched (B, C, H, W)
        x_packed = rearrange(x, "b c h w -> b (h w) c")
        x_ids = x_ids.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, HW, 4)

    return x_packed, x_ids.to(x.device)


# ---------------------------------------------------------------------------
# Text packing (sequence-dim already flat, just generate IDs)
# ---------------------------------------------------------------------------

def prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Generate 4-D position IDs for text tokens.

    Args:
        x: Shape ``(L, D)`` or ``(B, L, D)`` - text embeddings.
        t_coord: Optional 1-D int tensor for the time coordinate. Defaults to
            ``torch.arange(1)``.

    Returns:
        Tuple of (x, x_ids) where x_ids has shape ``(L, 4)`` or ``(B, L, 4)``.
    """
    seq_len = x.shape[-2]

    t_coords = torch.arange(1, device=x.device) if t_coord is None else t_coord
    x_ids = torch.cartesian_prod(
        t_coords,
        torch.arange(1, device=x.device),   # dummy h
        torch.arange(1, device=x.device),   # dummy w
        torch.arange(seq_len, device=x.device),
    )  # (L, 4)

    if x.ndim == 3:  # batched (B, L, D)
        x_ids = x_ids.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, L, 4)

    return x, x_ids.to(x.device)


# ---------------------------------------------------------------------------
# Unpack: sequence tokens → 4-D latent (for post-model decode)
# ---------------------------------------------------------------------------

def unpack_latents(x: Tensor, h: int, w: int) -> Tensor:
    """Reshape packed token sequence back to spatial latent grid.

    Args:
        x: ``(B, HW, C)`` - model output tokens.
        h: Latent height (H = image_height // 16).
        w: Latent width  (W = image_width // 16).

    Returns:
        ``(B, C, H, W)`` latent tensor.
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


# ---------------------------------------------------------------------------
# Scatter / reorder utilities (used at inference, not training)
# ---------------------------------------------------------------------------

def _compress_time(t_ids: Tensor) -> Tensor:
    """Remap sparse time IDs to a dense 0-based index."""
    t_ids_max = t_ids.max()
    t_remap = torch.zeros(t_ids_max + 1, device=t_ids.device, dtype=t_ids.dtype)
    t_unique = torch.unique(t_ids, sorted=True)
    t_remap[t_unique] = torch.arange(len(t_unique), device=t_ids.device, dtype=t_ids.dtype)
    return t_remap[t_ids]


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """Scatter packed token sequence back to 5-D ``(1, C, T, H, W)`` tensors.

    Used at inference to convert model outputs back to latent grids.

    Args:
        x:      ``(B, HW, C)``
        x_ids:  ``(B, HW, 4)`` - position IDs ``(t, h, w, l)``

    Returns:
        List of ``(1, C, T, H, W)`` tensors, one per batch item.
    """
    result: list[Tensor] = []
    for data, pos in zip(x, x_ids):
        _, channels = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_compressed = _compress_time(t_ids)

        t_dim = int(t_compressed.max().item()) + 1
        h_dim = int(h_ids.max().item()) + 1
        w_dim = int(w_ids.max().item()) + 1

        flat_ids = t_compressed * w_dim * h_dim + h_ids * w_dim + w_ids

        out = torch.zeros(
            t_dim * h_dim * w_dim, channels,
            device=data.device, dtype=data.dtype,
        )
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        result.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t_dim, h=h_dim, w=w_dim))

    return result
