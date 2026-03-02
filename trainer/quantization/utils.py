"""Utilities for applying quantization across a model graph."""
from __future__ import annotations

import logging
from typing import Callable

import torch.nn as nn

logger = logging.getLogger(__name__)

_SKIP_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.Embedding)


def replace_linear_layers(
    model: nn.Module,
    convert_fn: Callable[[nn.Linear], nn.Module],
) -> dict[str, int]:
    """Replace all nn.Linear in model with convert_fn(linear).

    Walks the graph recursively. Skips norm/embed layers.
    Returns dict with "quantized" and "skipped" counts.
    """
    stats = {"quantized": 0, "skipped": 0}
    visited: set[int] = set()
    _replace_recursive(model, convert_fn, stats, visited)
    return stats


def _replace_recursive(
    parent: nn.Module,
    convert_fn: Callable[[nn.Linear], nn.Module],
    stats: dict[str, int],
    visited: set[int],
) -> None:
    if id(parent) in visited:
        return
    visited.add(id(parent))

    for name, child in list(parent.named_children()):
        if isinstance(child, _SKIP_TYPES):
            stats["skipped"] += 1
            continue

        if isinstance(child, nn.Linear):
            converted = convert_fn(child)
            setattr(parent, name, converted)
            stats["quantized"] += 1
        else:
            _replace_recursive(child, convert_fn, stats, visited)
