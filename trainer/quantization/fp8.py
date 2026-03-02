"""FP8 weight-only quantization. Pure PyTorch, no external dependencies."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import QuantizedLinear


class LinearFp8(QuantizedLinear):
    """Linear layer with FP8 weight storage."""

    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias, compute_dtype)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self._weight_scale: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(dtype=self.compute_dtype)
        if self._weight_scale is not None:
            w = w * self._weight_scale
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
        return F.linear(x.to(self.compute_dtype), w, bias)

    def dequantize_weight(self) -> Tensor:
        w = self.weight.float()
        if self._weight_scale is not None:
            w = w * self._weight_scale
        return w

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, scaled: bool = False, **kwargs) -> "LinearFp8":
        q = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=kwargs.get("compute_dtype", torch.bfloat16),
        )
        with torch.no_grad():
            w = linear.weight.data.float()
            if scaled:
                max_val = w.abs().max().clamp(min=1e-12)
                scale = torch.finfo(torch.float8_e4m3fn).max / max_val
                q.weight.data = (w * scale).to(torch.float8_e4m3fn)
                q._weight_scale = scale.reciprocal()
            else:
                q.weight.data = w.to(torch.float8_e4m3fn)
            if linear.bias is not None:
                q.bias.data.copy_(linear.bias.data)
        return q


def quantize_linear_fp8(linear: nn.Linear, scaled: bool = False, **kwargs) -> LinearFp8:
    """Convert a single nn.Linear to FP8."""
    return LinearFp8.from_linear(linear, scaled=scaled, **kwargs)
