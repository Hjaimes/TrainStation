"""MLP, normalization, activation, and modulation layers for HunyuanVideo.

Ported from Musubi_Tuner's hunyuan_model/{mlp_layers,norm_layers,activation_layers,modulate_layers}.py.
Improvements over source:
  - Removed logging.basicConfig() calls
  - Replaced print() with logger calls
  - Removed dead commented-out code
  - Consistent factory_kwargs pattern
"""
from __future__ import annotations

import collections.abc
from functools import partial
from itertools import repeat
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ntuple(n: int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------

def get_activation_layer(act_type: str) -> Callable[[], nn.Module]:
    """Return a zero-argument factory for the requested activation type."""
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type!r}")


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            out = out * self.weight.to(out.dtype)
        return out


def get_norm_layer(norm_layer: str) -> type:
    """Return norm layer class by name."""
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer!r} is not implemented")


# ---------------------------------------------------------------------------
# Modulation layers
# ---------------------------------------------------------------------------

class ModulateDiT(nn.Module):
    """Modulation layer for DiT: projects conditioning vector into shift/scale/gate factors."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)
        # Zero-initialize so modulation starts as identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


def modulate(
    x: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift."""
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# MLP layers
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Optional[type] = None,
        bias: bool = True,
        drop: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias_pair = to_2tuple(bias)
        drop_pair = to_2tuple(drop)

        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias_pair[0], **factory_kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_pair[0])
        self.norm = norm_layer(hidden_channels, **factory_kwargs) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_channels, out_features, bias=bias_pair[1], **factory_kwargs)
        self.drop2 = nn.Dropout(drop_pair[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLPEmbedder(nn.Module):
    """Two-layer MLP for embedding vectors (e.g., pooled text embedding)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class FinalLayer(nn.Module):
    """Final projection layer: LayerNorm + adaLN modulation + linear to patch pixels."""

    def __init__(
        self,
        hidden_size: int,
        patch_size,
        out_channels: int,
        act_layer: Callable,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        if isinstance(patch_size, int):
            out_dim = patch_size * patch_size * out_channels
        else:
            out_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_channels

        self.linear = nn.Linear(hidden_size, out_dim, bias=True, **factory_kwargs)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x
