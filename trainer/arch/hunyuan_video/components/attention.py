"""Attention implementation for HunyuanVideo.

Ported from Musubi_Tuner's hunyuan_model/attention.py.
Improvements:
  - Removed print() / logging.basicConfig()
  - Removed dead/commented-out code
  - Kept multi-backend support: flash, flash_fixlen, sageattn, torch, xformers, vanilla
  - Added proper logger for optional import warnings
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------------

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward,
        flash_attn_varlen_func,
        flash_attn_func,
    )
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

# ---------------------------------------------------------------------------
# Memory layout transformations per backend
# ---------------------------------------------------------------------------

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "flash_fixlen": (
        lambda x: x,
        lambda x: x,
    ),
    "sageattn": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "sageattn_fixlen": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "xformers": (
        lambda x: x,
        lambda x: x,
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def get_cu_seqlens(text_mask: torch.Tensor, img_len: int) -> torch.Tensor:
    """Compute cumulative sequence lengths for varlen flash attention.

    Args:
        text_mask: [B, S_text] bool tensor, True for real tokens.
        img_len: number of image/video tokens.

    Returns:
        cu_seqlens: [2*B+1] int32 tensor for flash_attn_varlen_func.
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        s = text_len[i] + img_len
        cu_seqlens[2 * i + 1] = i * max_len + s
        cu_seqlens[2 * i + 2] = (i + 1) * max_len

    return cu_seqlens


def attention(
    q_or_qkv_list,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    mode: str = "flash",
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    total_len: Optional[torch.Tensor] = None,
    causal: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    batch_size: int = 1,
) -> torch.Tensor:
    """Multi-backend QKV attention.

    Supports: flash (varlen), flash_fixlen, sageattn (varlen), sageattn_fixlen,
              torch (SDPA), xformers, vanilla.

    Input q/k/v shape: [B, S, H, D] (batch, seq, heads, head_dim).
    Output shape: [B, S, H*D].
    """
    # Accept either a list [q, k, v] or separate tensors
    if isinstance(q_or_qkv_list, list):
        q, k, v = q_or_qkv_list
        q_or_qkv_list.clear()
    else:
        q = q_or_qkv_list

    split_attn = total_len is not None

    # Auto-upgrade mode when varlen is not applicable
    if (split_attn or cu_seqlens_q is None) and mode == "sageattn":
        mode = "sageattn_fixlen"
    elif (split_attn or cu_seqlens_q is None) and mode == "flash":
        mode = "flash_fixlen"

    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    if split_attn:
        trimmed_len = q.shape[1] - total_len
        q = [pre_attn_layout(q[i: i + 1, : total_len[i]]) for i in range(len(q))]
        k = [pre_attn_layout(k[i: i + 1, : total_len[i]]) for i in range(len(k))]
        v = [pre_attn_layout(v[i: i + 1, : total_len[i]]) for i in range(len(v))]
    else:
        q = pre_attn_layout(q)
        k = pre_attn_layout(k)
        v = pre_attn_layout(v)

    # --- torch (SDPA) ---
    if mode == "torch":
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = F.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate, is_causal=causal)
                q[i] = k[i] = v[i] = None
                x.append(x_i)
        else:
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(q.dtype)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)

    # --- xformers ---
    elif mode == "xformers":
        assert xops is not None, "xformers not installed"
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)
                q[i] = k[i] = v[i] = None
                x.append(x_i)
        else:
            x = xops.memory_efficient_attention(q, k, v, p=drop_rate)

    # --- flash (varlen) ---
    elif mode == "flash":
        assert flash_attn_varlen_func is not None, "flash_attn not installed"
        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])

    # --- flash_fixlen ---
    elif mode == "flash_fixlen":
        assert flash_attn_func is not None, "flash_attn not installed"
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = flash_attn_func(q[i], k[i], v[i], dropout_p=drop_rate, causal=causal)
                q[i] = k[i] = v[i] = None
                x.append(x_i)
        else:
            x = flash_attn_func(q, k, v, dropout_p=drop_rate, causal=causal)

    # --- sageattn (varlen) ---
    elif mode == "sageattn":
        assert sageattn_varlen is not None, "sageattention not installed"
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])

    # --- sageattn_fixlen ---
    elif mode == "sageattn_fixlen":
        assert sageattn is not None, "sageattention not installed"
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = sageattn(q[i], k[i], v[i])
                q[i] = k[i] = v[i] = None
                x.append(x_i)
        else:
            x = sageattn(q, k, v)

    # --- vanilla (manual scaled dot-product) ---
    elif mode == "vanilla":
        assert not split_attn, "Vanilla attention does not support split_attn"
        scale = 1.0 / math.sqrt(q.size(-1))
        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_bias + attn_mask
        scores = (q @ k.transpose(-2, -1)) * scale + attn_bias
        scores = scores.softmax(dim=-1)
        scores = torch.dropout(scores, p=drop_rate, train=True)
        x = scores @ v

    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode!r}")

    # Reassemble split batches
    if split_attn:
        x = [post_attn_layout(x_i) for x_i in x]
        x = [F.pad(x_i, (0, 0, 0, 0, 0, trimmed_len[i])) for i, x_i in enumerate(x)]
        x = torch.cat(x, dim=0)
    else:
        x = post_attn_layout(x)

    b, s, a, d = x.shape
    return x.reshape(b, s, a * d)
