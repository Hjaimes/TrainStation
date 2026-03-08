"""NF4 and INT8 quantization via bitsandbytes. Deferred import - bitsandbytes is optional."""
from __future__ import annotations

import logging
from functools import lru_cache

import torch
import torch.nn as nn
from torch import Tensor

from .base import QuantizedLinear

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def is_bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def _require_bnb():
    if not is_bnb_available():
        raise ImportError(
            "bitsandbytes is required for NF4/INT8 quantization. "
            "Install with: pip install bitsandbytes"
        )


class LinearNf4(QuantizedLinear):
    """Linear layer with NF4 (4-bit) weight storage via bitsandbytes."""

    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias, compute_dtype)
        _require_bnb()
        self._quantized_weight: Tensor | None = None
        self._quant_state = None

    def forward(self, x: Tensor) -> Tensor:
        import bitsandbytes as bnb
        x = x.to(self.compute_dtype)
        return bnb.matmul_4bit(
            x, self._quantized_weight.t(),
            bias=self.bias, quant_state=self._quant_state,
        )

    def dequantize_weight(self) -> Tensor:
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(
            self._quantized_weight, self._quant_state
        ).float()

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "LinearNf4":
        import bitsandbytes as bnb
        q = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=kwargs.get("compute_dtype", torch.bfloat16),
        )
        with torch.no_grad():
            w = linear.weight.data.float()
            q._quantized_weight, q._quant_state = bnb.functional.quantize_4bit(
                w, blocksize=64, compress_statistics=True,
                quant_type="nf4", quant_storage=torch.uint8,
            )
            if linear.bias is not None:
                q.bias.data.copy_(linear.bias.data)
        return q


class LinearInt8(QuantizedLinear):
    """Linear layer with INT8 weight storage via bitsandbytes."""

    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias, compute_dtype)
        _require_bnb()
        self._int8_module: nn.Module | None = None

    def forward(self, x: Tensor) -> Tensor:
        return self._int8_module(x)

    def dequantize_weight(self) -> Tensor:
        m = self._int8_module
        return (m.weight.data.float() * m.state.SCB.float().unsqueeze(1) / 127.0)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "LinearInt8":
        import bitsandbytes as bnb
        q = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=kwargs.get("compute_dtype", torch.bfloat16),
        )
        int8_linear = bnb.nn.Linear8bitLt(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            has_fp16_weights=False,
        )
        int8_linear.weight = nn.Parameter(linear.weight.data, requires_grad=False)
        if linear.bias is not None:
            int8_linear.bias = nn.Parameter(linear.bias.data, requires_grad=False)
        q._int8_module = int8_linear
        return q


def quantize_linear_nf4(linear: nn.Linear, **kwargs) -> LinearNf4:
    return LinearNf4.from_linear(linear, **kwargs)


def quantize_linear_int8(linear: nn.Linear, **kwargs) -> LinearInt8:
    return LinearInt8.from_linear(linear, **kwargs)
