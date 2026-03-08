"""Flux 1 positional and conditioning embeddings.

Key components:
- Flux1RoPE: 3D rotary position embeddings with axes (16, 56, 56)
- TimestepEmbedding: sinusoidal -> MLP for denoising timestep
- GuidanceEmbedding: sinusoidal -> MLP for CFG guidance scale (dev only)
- MLPEmbedder: projects CLIP-L pooled text embedding into hidden dim
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class Flux1RoPE(nn.Module):
    """3D Rotary Position Embeddings for Flux 1.

    Axes: (channel=16, y=56, x=56) - each axis gets its own frequency band.
    Total RoPE dim = sum(axes_dim) = 128 = head_dim.

    Unlike Flux 2's N-D RoPE (which uses rotation matrices), this implementation
    uses cos/sin interleaved frequencies for straightforward apply_rope in blocks.
    """

    def __init__(self, axes_dim: tuple[int, int, int] = (16, 56, 56), theta: float = 10000.0):
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def _compute_freqs(self, positions: Tensor, dim: int) -> tuple[Tensor, Tensor]:
        """Compute cos and sin frequencies for one axis.

        Args:
            positions: (..., N) integer position indices.
            dim: Feature dimension for this axis (must be even).

        Returns:
            (cos, sin) each of shape (..., N, dim//2).
        """
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
        angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (..., N, dim//2)
        return torch.cos(angles), torch.sin(angles)

    def forward(self, positions: Tensor) -> Tensor:
        """Compute RoPE frequencies from 3D position IDs.

        Args:
            positions: (B, N, 3) float tensor - [channel_idx, y, x]

        Returns:
            (B, N, sum(axes_dim)) - format: [all_cos | all_sin] where cos and sin
            each have dim sum(axes_dim)//2 = head_dim//2. This format matches
            _apply_rope_flux1() which splits at the midpoint.
        """
        cos_parts = []
        sin_parts = []
        for i, dim in enumerate(self.axes_dim):
            cos_i, sin_i = self._compute_freqs(positions[..., i], dim)
            cos_parts.append(cos_i)
            sin_parts.append(sin_i)
        # Concatenate: [all_cos_axes | all_sin_axes] so consumer can split at midpoint
        return torch.cat(cos_parts + sin_parts, dim=-1)  # (B, N, sum(axes_dim))


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP.

    Inputs: timesteps in [0, 1] (Flux 1 does not scale to 1000).
    Output: (B, hidden_size) conditioning vector.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]  # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, freq_dim)

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self._sinusoidal(t))


class GuidanceEmbedding(nn.Module):
    """Guidance scale embedding (same structure as TimestepEmbedding).

    Used by Flux 1 dev (guidance-distilled); not present for schnell.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, g: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=g.device, dtype=torch.float32)
            / half
        )
        args = g[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, g: Tensor) -> Tensor:
        return self.mlp(self._sinusoidal(g))


class MLPEmbedder(nn.Module):
    """Simple MLP for pooled text projection: Linear -> SiLU -> Linear.

    Used to project CLIP-L pooled text embedding (768-dim) into hidden_size.
    """

    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_size)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.act(self.linear1(x)))
