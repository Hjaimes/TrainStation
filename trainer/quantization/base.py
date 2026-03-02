"""Base classes for quantized linear layers."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QuantizedLinear(nn.Module):
    """Abstract base for quantized linear layers.

    Subclasses store weights in a compressed format and dequantize
    on-the-fly during forward(). Base model weights are always frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.bias = None

    def dequantize_weight(self) -> Tensor:
        """Return full-precision weight for inspection/DoRA."""
        raise NotImplementedError

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "QuantizedLinear":
        """Convert an nn.Linear to this quantized type."""
        raise NotImplementedError
