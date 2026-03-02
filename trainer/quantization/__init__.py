"""Multi-type model quantization for VRAM reduction.

Supported types:
- "nf4"        — 4-bit NormalFloat via bitsandbytes (~75% VRAM reduction)
- "int8"       — 8-bit integer via bitsandbytes (~50% reduction)
- "fp8"        — 8-bit float, pure PyTorch (~50% reduction)
- "fp8_scaled" — 8-bit float with per-tensor scaling (~50% reduction)
- None         — no quantization

Usage in strategy setup():
    from trainer.quantization import quantize_model
    if cfg.model.quantization:
        quantize_model(model, cfg.model.quantization, compute_dtype=train_dtype)
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

from .base import QuantizedLinear
from .utils import replace_linear_layers

logger = logging.getLogger(__name__)

_QUANTIZERS: dict[str, Callable] = {}


def _register_quantizers():
    """Lazily register all quantization backends."""
    if _QUANTIZERS:
        return

    from .fp8 import quantize_linear_fp8
    _QUANTIZERS["fp8"] = lambda **kw: lambda lin: quantize_linear_fp8(lin, scaled=False, **kw)
    _QUANTIZERS["fp8_scaled"] = lambda **kw: lambda lin: quantize_linear_fp8(lin, scaled=True, **kw)

    from .bnb import is_bnb_available
    if is_bnb_available():
        from .bnb import quantize_linear_nf4, quantize_linear_int8
        _QUANTIZERS["nf4"] = lambda **kw: lambda lin: quantize_linear_nf4(lin, **kw)
        _QUANTIZERS["int8"] = lambda **kw: lambda lin: quantize_linear_int8(lin, **kw)
    else:
        def _bnb_stub(name):
            def _factory(**kw):
                raise ImportError(
                    f"bitsandbytes is required for '{name}' quantization. "
                    f"Install with: pip install bitsandbytes"
                )
            return _factory
        _QUANTIZERS["nf4"] = _bnb_stub("nf4")
        _QUANTIZERS["int8"] = _bnb_stub("int8")


def get_quantizer(quantization_type: str | None):
    """Get a quantizer factory for the given type, or None."""
    if quantization_type is None:
        return None
    _register_quantizers()
    if quantization_type not in _QUANTIZERS:
        raise ValueError(
            f"Unknown quantization type '{quantization_type}'. "
            f"Supported: {sorted(_QUANTIZERS.keys())}"
        )
    return _QUANTIZERS[quantization_type]


def quantize_model(
    model: nn.Module,
    quantization_type: str,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, int]:
    """Quantize all Linear layers in a model.

    Returns dict with "quantized" and "skipped" counts.
    """
    factory = get_quantizer(quantization_type)
    if factory is None:
        return {"quantized": 0, "skipped": 0}

    convert_fn = factory(compute_dtype=compute_dtype)
    logger.info("Quantizing model with '%s' (%s compute)", quantization_type, compute_dtype)
    stats = replace_linear_layers(model, convert_fn)
    logger.info(
        "Quantized %d linear layers, skipped %d norm/embed layers",
        stats["quantized"], stats["skipped"],
    )
    return stats
