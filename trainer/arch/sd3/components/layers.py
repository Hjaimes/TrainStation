"""Core layer building blocks for the SD3 MMDiT architecture.

Includes adaptive layer norms for timestep/pooled conditioning and the
feed-forward MLP used in both joint and single transformer blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdaLayerNormZero(nn.Module):
    """Adaptive layer norm with timestep + pooled conditioning.

    Produces 6 modulation params: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
    Returns the modulated x (shift+scale applied) plus the remaining 4 params for gating.

    Forward returns: (x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        # Project conditioning to 6 modulation params per feature
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: Tensor, emb: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x:   (B, L, D) — input sequence
            emb: (B, D)    — conditioning (timestep + pooled text)

        Returns:
            x_normed:  (B, L, D) — norm(x) * (1 + scale_msa) + shift_msa
            gate_msa:  (B, 1, D) — gating for attention output
            shift_mlp: (B, 1, D)
            scale_mlp: (B, 1, D)
            gate_mlp:  (B, 1, D)
        """
        # (B, 6D) -> 6 tensors of (B, 1, D)
        params = self.linear(self.silu(emb)).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        x_normed = self.norm(x) * (1 + scale_msa) + shift_msa
        return x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Adaptive layer norm for single (image-only) transformer blocks.

    Produces 6 modulation params (same as AdaLayerNormZero):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.

    Forward returns: (x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: Tensor, emb: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x:   (B, L, D)
            emb: (B, D)

        Returns:
            x_normed:  (B, L, D) — norm(x) * (1 + scale_msa) + shift_msa
            gate_msa:  (B, 1, D)
            shift_mlp: (B, 1, D)
            scale_mlp: (B, 1, D)
            gate_mlp:  (B, 1, D)
        """
        params = self.linear(self.silu(emb)).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        x_normed = self.norm(x) * (1 + scale_msa) + shift_msa
        return x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(nn.Module):
    """Final adaptive layer norm for output projection.

    Uses a continuous conditioning vector (e.g., timestep embedding) to
    produce shift and scale for the final norm before output projection.
    """

    def __init__(self, embedding_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 2 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """
        Args:
            x:           (B, L, D)
            conditioning: (B, D)

        Returns:
            (B, L, D) — normalized and modulated
        """
        emb = self.linear(self.silu(conditioning))
        # (B, 2D) -> unsqueeze to (B, 1, 2D) -> split to (B, 1, D) each
        shift, scale = emb.unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class FeedForward(nn.Module):
    """GELU MLP: Linear(d, 4d) -> GELU(approx=tanh) -> Linear(4d, d).

    Uses tanh-approximate GELU as in SD3 reference implementation.
    """

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
