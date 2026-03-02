"""Primitive layer building blocks for HunyuanVideo 1.5.

Includes: MLP, RMSNorm, modulate, apply_gate, _to_tuple.
Self-contained — no imports from other arch packages.

Porting improvements over Musubi_Tuner source:
- print() removed, logging used instead
- Dead/commented code removed
- Consistent torch API (torch.cat not torch.concat)
- Docstrings only where non-obvious
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_tuple(x: int | float | Sequence, dim: int = 2) -> tuple:
    """Convert scalar or sequence to tuple of length *dim*."""
    if isinstance(x, (int, float)):
        return (x,) * dim
    seq = tuple(x)
    if len(seq) == dim:
        return seq
    raise ValueError(f"Expected length {dim} or scalar, got {x!r}")


# ---------------------------------------------------------------------------
# Norm layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm — no mean computation.
    Weight is cast to output dtype to allow fp8 linear layers to coexist.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def reset_parameters(self) -> None:
        self.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight.to(out.dtype)


# ---------------------------------------------------------------------------
# AdaLN modulation
# ---------------------------------------------------------------------------

def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adaptive layer-norm modulation: x * (1 + scale) + shift."""
    if scale is None and shift is None:
        return x
    if shift is None:
        return x * (1.0 + scale.unsqueeze(1))
    if scale is None:
        return x + shift.unsqueeze(1)
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x: torch.Tensor, gate: torch.Tensor | None = None, tanh: bool = False) -> torch.Tensor:
    """Multiply *x* by *gate* (with optional tanh), used in residual paths."""
    if gate is None:
        return x
    g = gate.unsqueeze(1).tanh() if tanh else gate.unsqueeze(1)
    return x * g


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard two-layer MLP with configurable activation and optional dropout.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width (defaults to in_channels).
        out_features: Output dimension (defaults to in_channels).
        act_layer: Callable returning an activation module.
        norm_layer: Optional callable returning a normalization module.
        bias: Bias flag(s) — bool or 2-tuple for each layer.
        drop: Dropout rate(s) — float or 2-tuple for each layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] | None = None,
        bias: bool | tuple[bool, bool] = True,
        drop: float | tuple[float, float] = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        b1, b2 = _to_tuple(bias, 2)
        d1, d2 = _to_tuple(drop, 2)

        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=b1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(d1)
        self.norm = norm_layer(hidden_channels) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_channels, out_features, bias=b2)
        self.drop2 = nn.Dropout(d2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------------------------------------------------------
# ModulateDiT (projects timestep vec into multiple AdaLN params)
# ---------------------------------------------------------------------------

class ModulateDiT(nn.Module):
    """Projects conditioning vector into *factor* modulation parameters for AdaLN."""

    def __init__(self, hidden_size: int, factor: int, act_layer: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))
