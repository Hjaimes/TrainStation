"""Flux Kontext packing/unpacking and position-ID utilities.

Ported from Musubi_Tuner flux/flux_utils.py (prepare_img_ids) with:
- print() removed
- logging.basicConfig() removed
- torch.concat → torch.cat
- Dead/commented-out code removed
- Self-contained (no imports from flux_2 utils)

Key differences from Flux 2 utils:
- Position IDs are 3-axis: (row, col, is_ctrl_flag) not 4-axis (t, h, w, l)
- Control image IDs set their first axis to 1 (is_ctrl=True) to distinguish them
  from noise latent IDs (is_ctrl=False, first axis = 0)
- Latents are 16-channel (not 128), packed 2×2 → 64 channels per token
"""
from __future__ import annotations

import logging

import torch
from einops import rearrange
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position ID generation
# ---------------------------------------------------------------------------

def prepare_img_ids(
    batch_size: int,
    packed_latent_height: int,
    packed_latent_width: int,
    is_ctrl: bool = False,
) -> Tensor:
    """Build 3-axis position IDs for image (noise or control) tokens.

    The Flux Kontext model uses 3-axis RoPE:
      axis 0: reserved / control flag (0 = noise token, 1 = control token)
      axis 1: row index
      axis 2: column index

    Args:
        batch_size:           Number of samples in the batch.
        packed_latent_height: H after 2×2 packing (original_h // 2).
        packed_latent_width:  W after 2×2 packing (original_w // 2).
        is_ctrl:              If True, set axis 0 = 1 (control token marker).

    Returns:
        ``(B, H*W, 3)`` integer position ID tensor.
    """
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width)[None, :]
    if is_ctrl:
        img_ids[..., 0] = 1
    # Repeat for each batch item: (H, W, 3) → (B, H*W, 3)
    img_ids = rearrange(img_ids, "h w c -> 1 (h w) c").expand(batch_size, -1, -1).contiguous()
    return img_ids


def prepare_txt_ids(batch_size: int, seq_len: int) -> Tensor:
    """Build 3-axis position IDs for text tokens (all zeros).

    Args:
        batch_size: Number of samples in the batch.
        seq_len:    Number of text tokens (T5 sequence length).

    Returns:
        ``(B, L, 3)`` zero-filled position ID tensor.
    """
    return torch.zeros(batch_size, seq_len, 3)


# ---------------------------------------------------------------------------
# Latent packing / unpacking
# ---------------------------------------------------------------------------

def pack_latents(x: Tensor) -> Tensor:
    """Pack 4-D latents from spatial format to token sequence.

    Flux Kontext uses 16-channel latents with a 2×2 spatial patch:
    ``(B, 16, H, W)`` → ``(B, H//2 * W//2, 64)``

    Args:
        x: ``(B, 16, H, W)`` latent tensor.

    Returns:
        ``(B, H*W//4, 64)`` packed token sequence.
    """
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_latents(x: Tensor, packed_h: int, packed_w: int) -> Tensor:
    """Unpack token sequence back to 4-D latent spatial format.

    Inverse of ``pack_latents``.

    Args:
        x:        ``(B, H*W, 64)`` packed token sequence.
        packed_h: H after packing (original_h // 2).
        packed_w: W after packing (original_w // 2).

    Returns:
        ``(B, 16, H*2, W*2)`` latent tensor.
    """
    return rearrange(
        x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed_h, w=packed_w, ph=2, pw=2,
    )
