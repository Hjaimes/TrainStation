"""Configurable loss functions for training.

All loss functions accept (pred, target) tensors and return a scalar loss.
Functions are resolved once at setup time and cached - no per-step dispatch.
"""
from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

LossFn = Callable[[Tensor, Tensor], Tensor]


def get_loss_fn(loss_type: str, *, delta: float = 1.0) -> LossFn:
    """Return a loss callable for the given type.

    Args:
        loss_type: One of "mse", "l1", "mae", "huber".
        delta: Huber loss delta parameter (only used when loss_type="huber").

    Returns:
        A callable (pred, target) -> scalar loss tensor.
    """
    match loss_type:
        case "mse":
            return _mse_loss
        case "l1" | "mae":
            return _l1_loss
        case "huber":
            return functools.partial(_huber_loss, delta=delta)
        case _:
            raise ValueError(
                f"Unknown loss type '{loss_type}'. "
                f"Supported: 'mse', 'l1', 'mae', 'huber'."
            )


def get_unreduced_loss_fn(loss_type: str, *, delta: float = 1.0) -> LossFn:
    """Return a loss callable with reduction='none' for per-element loss.

    Same interface as get_loss_fn but returns unreduced (per-element) loss.
    Used by weighted loss computation (SNR weighting, etc.).
    """
    match loss_type:
        case "mse":
            return _mse_loss_unreduced
        case "l1" | "mae":
            return _l1_loss_unreduced
        case "huber":
            return functools.partial(_huber_loss_unreduced, delta=delta)
        case _:
            raise ValueError(
                f"Unknown loss type '{loss_type}'. "
                f"Supported: 'mse', 'l1', 'mae', 'huber'."
            )


def compute_loss(
    pred: Tensor,
    target: Tensor,
    loss_type: str = "mse",
    *,
    delta: float = 1.0,
) -> Tensor:
    """Compute loss between prediction and target.

    Convenience wrapper that resolves the loss function and applies it.
    For hot-path usage, prefer caching the result of get_loss_fn() instead.
    """
    fn = get_loss_fn(loss_type, delta=delta)
    return fn(pred, target)


# --- Reduced (scalar) loss functions ---

def _mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target, reduction="mean")


def _l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target, reduction="mean")


def _huber_loss(pred: Tensor, target: Tensor, *, delta: float = 1.0) -> Tensor:
    return F.huber_loss(pred, target, reduction="mean", delta=delta)


# --- Unreduced (per-element) loss functions ---

def _mse_loss_unreduced(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target, reduction="none")


def _l1_loss_unreduced(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target, reduction="none")


def _huber_loss_unreduced(pred: Tensor, target: Tensor, *, delta: float = 1.0) -> Tensor:
    return F.huber_loss(pred, target, reduction="none", delta=delta)
