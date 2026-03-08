"""Attention implementation for HunyuanVideo 1.5.

Self-contained AttentionParams + attention() function. Supports:
- torch (SDPA, default)
- flash (flash-attn v2)
- sageattn
- xformers

Porting improvements over Musubi_Tuner source:
- Removed print() calls
- Removed logging.basicConfig()
- Consistent imports
- Docstrings trimmed to non-obvious logic only
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional fast-attention backends - gracefully degrade to SDPA
try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_func = None
    flash_attn_func = None

try:
    from sageattention import sageattn_varlen, sageattn

    SAGE_ATTN_AVAILABLE = True
except ImportError:
    SAGE_ATTN_AVAILABLE = False
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    xops = None


# ---------------------------------------------------------------------------
# AttentionParams
# ---------------------------------------------------------------------------

@dataclass
class AttentionParams:
    """Parameters controlling how attention is computed for a given batch."""

    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode=attn_mode, split_attn=split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str],
        split_attn: bool,
        img_len: Optional[int],
        attention_mask: Optional[torch.Tensor],
    ) -> "AttentionParams":
        if attention_mask is None:
            return AttentionParams(attn_mode=attn_mode, split_attn=split_attn)

        # attention_mask covers text tokens only; add img tokens to seqlen
        seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len
        max_seqlen = attention_mask.shape[1] + img_len

        if split_attn:
            return AttentionParams(
                attn_mode=attn_mode, split_attn=split_attn, img_len=img_len,
                attention_mask=attention_mask, seqlens=seqlens,
                cu_seqlens=None, max_seqlen=max_seqlen,
            )

        # Build cumulative seqlens for flash/varlen attention
        batch_size = attention_mask.shape[0]
        cu_seqlens = torch.zeros(2 * batch_size + 1, dtype=torch.int32, device=attention_mask.device)
        for i in range(batch_size):
            cu_seqlens[2 * i + 1] = i * max_seqlen + seqlens[i]
            cu_seqlens[2 * i + 2] = (i + 1) * max_seqlen

        # Expand mask to include image tokens
        attention_mask = F.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]

        if attn_mode == "xformers" and XFORMERS_AVAILABLE:
            seqlens_list = seqlens.cpu().tolist()
            attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                seqlens_list, seqlens_list, device=attention_mask.device
            )
        elif attn_mode == "torch":
            attention_mask = attention_mask[:, None, None, :].to(torch.bool)

        return AttentionParams(
            attn_mode=attn_mode, split_attn=split_attn, img_len=img_len,
            attention_mask=attention_mask, seqlens=seqlens,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )


# ---------------------------------------------------------------------------
# attention()
# ---------------------------------------------------------------------------

def attention(
    qkv_or_q: Union[torch.Tensor, list[torch.Tensor]],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """Scaled dot-product attention supporting multiple backends.

    Args:
        qkv_or_q: Either a [q, k, v] list (consumed in-place) or just q [B, L, H, D].
        k: Key tensor when qkv_or_q is a plain query.
        v: Value tensor when qkv_or_q is a plain query.
        attn_params: Attention mode, mask, and sequence length metadata.
        drop_rate: Attention dropout probability.

    Returns:
        Attention output [B, L, H*D].
    """
    # Unpack list form to free references early
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        qkv_or_q.clear()
    else:
        q = qkv_or_q
        assert k is not None and v is not None

    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # Sequence trimming: when all seqlens are equal, trim before SDPA/xformers
    seqlen_trimmed = False
    if (
        not attn_params.split_attn
        and attn_params.attention_mask is not None
        and attn_params.seqlens is not None
        and attn_params.attn_mode not in ("flash", "sageattn")
    ):
        if torch.all(attn_params.seqlens == attn_params.seqlens[0]):
            sl = int(attn_params.seqlens[0].item())
            q = q[:, :sl]
            k = k[:, :sl]
            v = v[:, :sl]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)
            attn_params.max_seqlen = max_seqlen
            seqlen_trimmed = True

    # Layout helpers differ per backend
    # SDPA/sageattn-noCu expect [B, H, L, D]; others [B, L, H, D]
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        def transpose_fn(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(1, 2)

        def pad_fn(x: torch.Tensor, pad_to: int) -> torch.Tensor:
            return F.pad(x, (0, 0, 0, pad_to - x.shape[-2]))
    else:
        def transpose_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        def pad_fn(x: torch.Tensor, pad_to: int) -> torch.Tensor:
            return F.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]))

    # ------------------------------------------------------------------
    # Split-attention path: process each batch element independently
    # ------------------------------------------------------------------
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, True)
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]

        qs = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(q.shape[0])]
        ks = [transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(k.shape[0])]
        vs = [transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(v.shape[0])]

        outputs = []
        for i in range(len(qs)):
            if attn_params.attn_mode == "torch":
                xi = F.scaled_dot_product_attention(qs[i], ks[i], vs[i], dropout_p=drop_rate)
            elif attn_params.attn_mode == "xformers" and XFORMERS_AVAILABLE:
                xi = xops.memory_efficient_attention(qs[i], ks[i], vs[i], p=drop_rate)
            elif attn_params.attn_mode == "sageattn" and SAGE_ATTN_AVAILABLE:
                xi = sageattn(qs[i], ks[i], vs[i])
            elif attn_params.attn_mode == "flash" and FLASH_ATTN_AVAILABLE:
                xi = flash_attn_func(qs[i], ks[i], vs[i], drop_rate)
            else:
                xi = F.scaled_dot_product_attention(qs[i], ks[i], vs[i], dropout_p=drop_rate)
            qs[i] = ks[i] = vs[i] = None
            outputs.append(pad_fn(xi, attn_params.max_seqlen))

        x = torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # Batched attention path
    # ------------------------------------------------------------------
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

        mode = attn_params.attn_mode

        if mode == "torch":
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate
            )

        elif mode == "xformers" and XFORMERS_AVAILABLE:
            x = xops.memory_efficient_attention(
                q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate
            )

        elif mode == "sageattn" and SAGE_ATTN_AVAILABLE:
            if attn_params.cu_seqlens is None:
                x = sageattn(q, k, v)
            else:
                B, L, H, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
                q = q.reshape(B * L, H, D)
                k = k.reshape(B * L, H, D)
                v = v.reshape(B * L, H, D)
                x = sageattn_varlen(
                    q, k, v,
                    attn_params.cu_seqlens, attn_params.cu_seqlens,
                    attn_params.max_seqlen, attn_params.max_seqlen,
                )
                x = x.view(B, L, x.shape[-2], x.shape[-1])

        elif mode == "flash" and FLASH_ATTN_AVAILABLE:
            if attn_params.cu_seqlens is None:
                x = flash_attn_func(q, k, v, drop_rate)
            else:
                B, L, H, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
                q = q.reshape(B * L, H, D)
                k = k.reshape(B * L, H, D)
                v = v.reshape(B * L, H, D)
                x = flash_attn_varlen_func(
                    q, k, v,
                    attn_params.cu_seqlens, attn_params.cu_seqlens,
                    attn_params.max_seqlen, attn_params.max_seqlen,
                    drop_rate,
                )
                x = x.view(B, L, x.shape[-2], x.shape[-1])

        else:
            # Fallback to SDPA
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate
            )

    # Merge heads: [B, H, L, D] or [B, L, H, D] → [B, L, H*D]
    x = transpose_fn(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)

    if seqlen_trimmed and attn_params.max_seqlen is not None:
        x = F.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]))

    return x
