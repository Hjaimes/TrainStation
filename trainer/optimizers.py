"""Optimizer factory.

Provides ``create_optimizer()``, an ``OPTIMIZERS`` registry dict, and
``list_optimizers()`` for discoverability.

Optional dependencies (bitsandbytes, prodigyopt, transformers) are imported
lazily so the module loads even when they are not installed.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

import torch

from trainer.errors import TrainerError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _get_adamw() -> type[torch.optim.Optimizer]:
    return torch.optim.AdamW


def _get_adam() -> type[torch.optim.Optimizer]:
    return torch.optim.Adam


def _get_sgd() -> type[torch.optim.Optimizer]:
    return torch.optim.SGD


def _get_adamw8bit() -> type[torch.optim.Optimizer]:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise TrainerError(
            "adamw8bit requires the 'bitsandbytes' package. "
            "Install it with: pip install bitsandbytes"
        )
    return bnb.optim.AdamW8bit


def _get_adafactor() -> type[torch.optim.Optimizer]:
    try:
        from transformers.optimization import Adafactor
    except ImportError:
        raise TrainerError(
            "adafactor requires the 'transformers' package. "
            "Install it with: pip install transformers"
        )
    return Adafactor


def _get_prodigy() -> type[torch.optim.Optimizer]:
    try:
        from prodigyopt import Prodigy
    except ImportError:
        raise TrainerError(
            "prodigy requires the 'prodigyopt' package. "
            "Install it with: pip install prodigyopt"
        )
    return Prodigy


def _get_lion() -> type[torch.optim.Optimizer]:
    try:
        from lion_pytorch import Lion
    except ImportError:
        raise TrainerError(
            "lion requires the 'lion-pytorch' package. "
            "Install it with: pip install lion-pytorch"
        )
    return Lion


def _get_came() -> type[torch.optim.Optimizer]:
    try:
        from came_pytorch import CAME
    except ImportError:
        raise TrainerError(
            "came requires the 'came-pytorch' package. "
            "Install it with: pip install came-pytorch"
        )
    return CAME


def _get_schedule_free_adamw() -> type[torch.optim.Optimizer]:
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError:
        raise TrainerError(
            "schedule_free_adamw requires the 'schedulefree' package. "
            "Install it with: pip install schedulefree"
        )
    return AdamWScheduleFree


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPTIMIZERS: dict[str, Callable[[], type[torch.optim.Optimizer]]] = {
    "adamw": _get_adamw,
    "adam": _get_adam,
    "sgd": _get_sgd,
    "adamw8bit": _get_adamw8bit,
    "adafactor": _get_adafactor,
    "prodigy": _get_prodigy,
    "lion": _get_lion,
    "came": _get_came,
    "schedule_free_adamw": _get_schedule_free_adamw,
}


def list_optimizers() -> list[str]:
    """Return the list of built-in optimizer names."""
    return sorted(OPTIMIZERS.keys())


# ---------------------------------------------------------------------------
# Dynamic import for dotted-path optimizer types
# ---------------------------------------------------------------------------

def _import_optimizer_class(dotted_path: str) -> type[torch.optim.Optimizer]:
    """Import an optimizer class from a fully-qualified dotted path.

    Example: ``"bitsandbytes.optim.AdEMAMix8bit"``
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise TrainerError(
            f"Invalid optimizer path '{dotted_path}'. "
            "Expected 'module.ClassName' (e.g. 'bitsandbytes.optim.AdamW8bit')."
        )
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise TrainerError(
            f"Could not import module '{module_path}' for optimizer '{dotted_path}': {e}"
        ) from e
    cls = getattr(module, class_name, None)
    if cls is None:
        raise TrainerError(
            f"Module '{module_path}' has no attribute '{class_name}'."
        )
    return cls


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_optimizer(
    optimizer_type: str,
    params: list[dict[str, Any]],
    lr: float,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create an optimizer by name.

    Args:
        optimizer_type: Name from ``OPTIMIZERS`` (case-insensitive) or a
            fully-qualified dotted path (e.g. ``"bitsandbytes.optim.AdamW8bit"``).
        params: Iterable of parameter groups (list of dicts with ``"params"`` key).
        lr: Learning rate.
        weight_decay: Weight decay (default 0.01).
        **kwargs: Extra keyword arguments forwarded to the optimizer constructor.

    Returns:
        A configured ``torch.optim.Optimizer`` instance.

    Raises:
        TrainerError: If the optimizer type is unknown or its dependency is missing.
    """
    name = optimizer_type.lower()

    if name in OPTIMIZERS:
        optimizer_cls = OPTIMIZERS[name]()
    elif "." in optimizer_type:
        # Dynamic import — use original casing for the class name
        optimizer_cls = _import_optimizer_class(optimizer_type)
    else:
        available = ", ".join(list_optimizers())
        raise TrainerError(
            f"Unknown optimizer '{optimizer_type}'. "
            f"Built-in options: {available}. "
            f"Or provide a dotted path like 'some.module.OptimizerClass'."
        )

    # Build kwargs dict — most optimizers accept lr + weight_decay
    opt_kwargs: dict[str, Any] = {"lr": lr, **kwargs}

    # SGD doesn't accept weight_decay by default in all versions,
    # but torch.optim.SGD does support it. Include it for all.
    if weight_decay != 0.0:
        opt_kwargs["weight_decay"] = weight_decay

    # Adafactor with relative_step=True ignores lr; let the user handle
    # that via optimizer_kwargs in the config rather than special-casing here.

    try:
        optimizer = optimizer_cls(params, **opt_kwargs)
    except TypeError as e:
        # Some optimizers don't accept weight_decay (e.g. Adafactor with
        # certain kwargs). Retry without it.
        if "weight_decay" in str(e) and "weight_decay" in opt_kwargs:
            logger.warning(
                f"Optimizer {optimizer_type} does not accept 'weight_decay'; "
                "retrying without it."
            )
            opt_kwargs.pop("weight_decay")
            optimizer = optimizer_cls(params, **opt_kwargs)
        else:
            raise TrainerError(
                f"Failed to create optimizer '{optimizer_type}': {e}"
            ) from e

    fqn = f"{optimizer_cls.__module__}.{optimizer_cls.__qualname__}"
    logger.info(f"Created optimizer: {fqn} (lr={lr}, weight_decay={weight_decay})")
    return optimizer
