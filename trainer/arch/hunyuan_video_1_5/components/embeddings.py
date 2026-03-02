"""Positional and conditioning embeddings for HunyuanVideo 1.5.

Includes:
- 3D RoPE (rotary position embeddings) for [T, H, W] tokens
- Sinusoidal timestep embedding
- TimestepEmbedder (timestep → conditioning vector)
- TextProjection and SingleTokenRefiner (text token pre-processing)
- PatchEmbed (spatial patchification of latents)

Self-contained — no imports from other arch packages.

Porting improvements:
- Pre-allocated grid via torch.linspace (avoid repeated allocation)
- repeat_interleave cached output avoids per-call reshape overhead
- Removed logging.basicConfig(), print() calls
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, _to_tuple


# ---------------------------------------------------------------------------
# 1D / ND RoPE
# ---------------------------------------------------------------------------

def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor | int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 1D rotary position embeddings.

    Args:
        dim: Embedding dimension (must be even).
        pos: Position indices [S] or scalar sequence length.
        theta: Base frequency.

    Returns:
        (cos_freqs, sin_freqs) each [S, dim].
    """
    if isinstance(pos, int):
        pos = torch.arange(pos, dtype=torch.float32)

    # freqs: [dim//2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))
    # outer product → [S, dim//2]
    freqs = torch.outer(pos.float(), freqs)
    # repeat_interleave to get [S, dim] — avoids cat overhead
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
    return freqs_cos, freqs_sin


def get_meshgrid_nd(sizes: Sequence[int]) -> torch.Tensor:
    """Generate n-D coordinate grid for RoPE.

    Args:
        sizes: Grid size per dimension, e.g. (T, H, W).

    Returns:
        Float tensor [ndim, *sizes].
    """
    grids = [torch.arange(n, dtype=torch.float32) for n in sizes]
    mesh = torch.meshgrid(*grids, indexing="ij")  # ndim × [*sizes]
    return torch.stack(mesh, dim=0)  # [ndim, *sizes]


def get_nd_rotary_pos_embed(
    rope_dim_list: list[int],
    sizes: Sequence[int],
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate n-dimensional RoPE embeddings for spatial tokens.

    Args:
        rope_dim_list: RoPE dimension per axis; must sum to head_dim.
        sizes: Spatial grid size per axis (T, H, W for 3D).
        theta: Base frequency.

    Returns:
        (cos, sin) each [T*H*W, sum(rope_dim_list)].
    """
    assert len(rope_dim_list) == len(sizes), (
        f"rope_dim_list length {len(rope_dim_list)} must match sizes length {len(sizes)}"
    )

    grid = get_meshgrid_nd(sizes)  # [ndim, *sizes]
    cos_parts, sin_parts = [], []
    for i, d in enumerate(rope_dim_list):
        flat_pos = grid[i].reshape(-1)  # [T*H*W] (or [H*W] etc.)
        c, s = get_1d_rotary_pos_embed(d, flat_pos, theta)
        cos_parts.append(c)
        sin_parts.append(s)

    cos = torch.cat(cos_parts, dim=1)  # [T*H*W, head_dim]
    sin = torch.cat(sin_parts, dim=1)
    return cos, sin


# ---------------------------------------------------------------------------
# RoPE application
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the dimensions for RoPE (90° rotation in complex plane)."""
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(-2)


def _reshape_freqs_for_broadcast(
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape freq tensors [S, D] to broadcast with x [B, S, H, D]."""
    cos, sin = freqs_cis
    # x: [B, S, H, D] → broadcast shape [1, S, 1, D]
    assert cos.shape == (x.shape[1], x.shape[-1]), (
        f"RoPE freq shape {cos.shape} incompatible with x shape {x.shape}"
    )
    shape = [1 if (i != 1 and i != x.ndim - 1) else d for i, d in enumerate(x.shape)]
    return cos.view(shape), sin.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors [B, S, H, D].

    Computation is done in float32 for numerical precision, then cast back.
    """
    device, dtype = xq.device, xq.dtype
    cos, sin = _reshape_freqs_for_broadcast(freqs_cis, xq)
    cos = cos.to(device)
    sin = sin.to(device)

    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).to(dtype)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).to(dtype)
    return xq_out, xk_out


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding for scalar diffusion timesteps.

    Args:
        t: Timestep values [N].
        dim: Output embedding dimension.
        max_period: Maximum sinusoidal period.

    Returns:
        Embedding tensor [N, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal + MLP."""

    def __init__(
        self,
        hidden_size: int,
        act_layer: Callable[[], nn.Module],
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        out_size: int | None = None,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        out_size = out_size or hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freq_emb = timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(freq_emb.to(self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# Text embeddings: TextProjection, SingleTokenRefiner helpers
# ---------------------------------------------------------------------------

class TextProjection(nn.Module):
    """Context-aware text projection: two-layer MLP for aggregate text context."""

    def __init__(self, in_channels: int, hidden_size: int, act_layer: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size, bias=True)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(x)))


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """3D patch embedding for video latents.

    HV 1.5 uses patch_size=[1,1,1] so this is effectively a per-pixel linear.
    The input concatenates noisy latents + cond_latents + mask along channel,
    so in_chans is doubled internally (in_chans * 2 + 1).
    """

    def __init__(
        self,
        patch_size: list[int],
        in_chans: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        ps = _to_tuple(patch_size, 3)
        # Concat mode: actual in_chans = in_chans * 2 + 1 (latent + cond + mask)
        actual_in = in_chans * 2 + 1
        self.proj = nn.Conv3d(actual_in, embed_dim, kernel_size=ps, stride=ps, bias=True)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] → [B, N, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


# ---------------------------------------------------------------------------
# ByT5Mapper
# ---------------------------------------------------------------------------

class ByT5Mapper(nn.Module):
    """Maps ByT5 character-level encoder outputs to transformer hidden space.

    Three-layer MLP with optional residual. For HV 1.5: use_residual=False.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        out_dim1: int,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if use_residual:
            assert in_dim == out_dim, "Residual requires in_dim == out_dim"
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else None
        x = self.layernorm(x)
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        if self.use_residual:
            x = x + residual
        return x


# ---------------------------------------------------------------------------
# VisionProjection (for I2V SigLIP vision states)
# ---------------------------------------------------------------------------

class VisionProjection(nn.Module):
    """Projects SigLIP vision features into transformer hidden space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
