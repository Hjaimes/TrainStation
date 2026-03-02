"""Kandinsky 5 utility functions.

Ported from Musubi_Tuner's kandinsky5/models/utils.py.
Improvements over source:
- `get_freqs` uses in-place float-cast to avoid extra allocation.
- `fast_sta_nabla` operates on pre-allocated arange, no redundant `Tensor()` call.
- `nablaT_v2` uses in-place `cumsum_` and avoids intermediate named tensors.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import BlockMask
    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    _FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    """Compute RoPE base frequencies (float32, CPU)."""
    return torch.exp(
        -math.log(max_period)
        * torch.arange(0, dim, dtype=torch.float32)
        / dim
    )


# ---------------------------------------------------------------------------
# Fractal flatten / unflatten (used for NABLA attention block structure)
# ---------------------------------------------------------------------------

def _local_patching(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    duration, height, width = shape
    g1, g2, g3 = group_size
    ndim_prefix = len(x.shape[:dim])
    x = x.reshape(
        *x.shape[:dim],
        duration // g1, g1,
        height // g2, g2,
        width // g3, g3,
        *x.shape[dim + 3:],
    )
    # Permute to interleave group tiles with spatial positions
    x = x.permute(
        *range(ndim_prefix),
        dim, dim + 2, dim + 4,
        dim + 1, dim + 3, dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def _local_merge(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    duration, height, width = shape
    g1, g2, g3 = group_size
    ndim_prefix = len(x.shape[:dim])
    x = x.reshape(
        *x.shape[:dim],
        duration // g1, height // g2, width // g3,
        g1, g2, g3,
        *x.shape[dim + 2:],
    )
    x = x.permute(
        *range(ndim_prefix),
        dim, dim + 3,
        dim + 1, dim + 4,
        dim + 2, dim + 5,
        *range(dim + 6, len(x.shape)),
    )
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x


def fractal_flatten(
    x: Tensor,
    rope: Tensor,
    shape: tuple[int, int, int],
    block_mask: bool = False,
) -> tuple[Tensor, Tensor]:
    """Flatten spatial dimensions; use fractal (tile-based) order when block_mask=True."""
    if block_mask:
        pixel_size = 8
        x = _local_patching(x, shape, (1, pixel_size, pixel_size), dim=0)
        rope = _local_patching(rope, shape, (1, pixel_size, pixel_size), dim=0)
        x = x.flatten(0, 1)
        rope = rope.flatten(0, 1)
    else:
        x = x.flatten(0, 2)
        rope = rope.flatten(0, 2)
    return x, rope


def fractal_unflatten(x: Tensor, shape: tuple[int, int, int], block_mask: bool = False) -> Tensor:
    """Inverse of fractal_flatten."""
    if block_mask:
        pixel_size = 8
        x = x.reshape(-1, pixel_size ** 2, *x.shape[1:])
        x = _local_merge(x, shape, (1, pixel_size, pixel_size), dim=0)
    else:
        x = x.reshape(*shape, *x.shape[1:])
    return x


# ---------------------------------------------------------------------------
# Sparse attention mask helpers
# ---------------------------------------------------------------------------

def fast_sta_nabla(
    T: int,
    H: int,
    W: int,
    wT: int = 3,
    wH: int = 3,
    wW: int = 3,
    device: str | torch.device = "cuda",
) -> Tensor:
    """Build the STA (Spatio-Temporal Attention) binary mask for NABLA attention.

    Returns a bool tensor of shape [T*H*W, T*H*W].
    Indices within the sliding window (wT/2, wH/2, wW/2) are True.
    """
    l = max(T, H, W)
    r = torch.arange(l, dtype=torch.int16, device=device)
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()

    sta_t = mat[:T, :T].flatten() <= (wT // 2)
    sta_h = mat[:H, :H].flatten() <= (wH // 2)
    sta_w = mat[:W, :W].flatten() <= (wW // 2)

    # sta_hw[h1*W+w1, h2*W+w2] = sta_h[h1,h2] & sta_w[w1,w2]
    sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
    # sta[t1*H*W+s1, t2*H*W+s2] = sta_t[t1,t2] & sta_hw[s1,s2]
    sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
    return sta.reshape(T * H * W, T * H * W)


def nablaT_v2(
    q: Tensor,
    k: Tensor,
    sta: Tensor,
    thr: float = 0.9,
    add_sta: bool = True,
    method: str = "topcdf",
) -> "BlockMask":
    """Compute a NABLA block-sparse attention mask from query/key coarse statistics.

    Args:
        q: Query tensor [B, h, S, D].
        k: Key tensor [B, h, S, D].
        sta: STA mask tensor (pre-computed by fast_sta_nabla).
        thr: CDF threshold (topcdf) or top-k ratio (topk).
        add_sta: Whether to OR the STA prior onto the computed mask.
        method: "topcdf" or "topk".

    Returns:
        A torch.nn.attention.flex_attention.BlockMask.
    """
    if not _FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "flex_attention is required for nablaT_v2 but torch.nn.attention.flex_attention "
            "is not available in this PyTorch version."
        )

    B, h, S, D = q.shape
    s1 = S // 64
    # Coarse attention map over 64-token super-blocks
    qa = q.reshape(B, h, s1, 64, D).mean(-2)
    ka = k.reshape(B, h, s1, 64, D).mean(-2).transpose(-2, -1)
    attn_map = torch.softmax((qa @ ka) / math.sqrt(D), dim=-1)

    # Binarise
    vals, inds = attn_map.sort(-1)
    if method == "topk":
        k_top = int(thr * vals.shape[-1]) if 0 < thr < 1 else int(thr)
        k_top = max(1, min(k_top, vals.shape[-1]))
        mask = torch.zeros_like(vals, dtype=torch.int)
        mask.scatter_(-1, inds[..., -k_top:], 1)
    else:
        # topcdf: keep tokens whose cumulative probability reaches (1 - thr)
        cvals = vals.cumsum_(-1)
        mask = (cvals >= (1 - thr)).int()
        mask = mask.gather(-1, inds.argsort(-1))

    if add_sta:
        mask = torch.logical_or(mask, sta)

    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        torch.zeros_like(kv_nb),
        kv_inds,
        kv_nb,
        kv_inds,
        BLOCK_SIZE=64,
        mask_mod=None,
    )
