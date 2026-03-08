"""Learning rate scheduler factory.

Provides ``create_scheduler()``, a ``SCHEDULERS`` registry dict, and
``list_schedulers()`` for discoverability.

Includes the custom REX scheduler inline (ported from Musubi_Tuner,
originally from https://arxiv.org/abs/2107.04197, Apache-2.0 License).
"""
from __future__ import annotations

import importlib
import logging
import math
from typing import Any, Callable

import torch
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from trainer.errors import TrainerError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REX scheduler (inline port)
# ---------------------------------------------------------------------------

class RexLR(LRScheduler):
    """Reflected Exponential (REX) learning rate scheduler.

    Reference: https://arxiv.org/abs/2107.04197
    Ported from Musubi_Tuner (originally https://github.com/IvanVassi/REX_LR,
    Apache-2.0 License).

    Linearly warms up from ``min_lr`` to ``max_lr`` over ``num_warmup_steps``,
    then follows the REX decay curve back toward ``min_lr``.

    Args:
        optimizer: The optimizer to schedule.
        max_lr: Peak learning rate (reached after warmup).
        min_lr: Minimum learning rate at start/end (default 0.0).
        num_steps: Total number of training steps.
        num_warmup_steps: Number of linear warmup steps.
        rex_alpha: Denominator constant preventing division-by-zero (default 0.1).
        rex_beta: Controls how quickly the decay flattens (default 0.9).
        last_epoch: Index of the last completed step (default -1).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float = 0.0,
        num_steps: int = 0,
        num_warmup_steps: int = 0,
        rex_alpha: float = 0.1,
        rex_beta: float = 0.9,
        last_epoch: int = -1,
    ):
        if min_lr > max_lr:
            raise ValueError(
                f"min_lr ({min_lr}) must be <= max_lr ({max_lr})."
            )
        if num_warmup_steps > num_steps:
            raise ValueError(
                f"num_warmup_steps ({num_warmup_steps}) must be <= "
                f"num_steps ({num_steps})."
            )

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        self.rex_alpha = rex_alpha
        self.rex_beta = rex_beta

        # Ensure each param group has an initial_lr for resume compatibility
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        # Single warmup step edge case
        if self.num_warmup_steps == 1 and self.last_epoch == 1:
            return [self.min_lr for _ in self.base_lrs]

        # Linear warmup phase
        if (
            self.num_warmup_steps > 1
            and 1 <= self.last_epoch <= self.num_warmup_steps - 1
        ):
            progress = (self.last_epoch - 1) / (self.num_warmup_steps - 1)
            lr = self.min_lr + (self.max_lr - self.min_lr) * progress
            return [lr for _ in self.base_lrs]

        # Post-warmup REX decay
        step_after = self.last_epoch - self.num_warmup_steps
        remaining = self.num_steps - self.num_warmup_steps

        if step_after >= remaining or step_after < 0 or remaining <= 0:
            return [self.min_lr for _ in self.base_lrs]

        z = (remaining - (step_after % remaining)) / remaining
        ratio = self.min_lr / self.max_lr
        rex_factor = ratio + (1.0 - ratio) * (
            z / (self.rex_alpha + self.rex_beta * z)
        )
        return [base_lr * rex_factor for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Lazy import helpers for transformers schedulers
# ---------------------------------------------------------------------------

def _require_transformers() -> Any:
    """Import and return the transformers module, or raise a clear error."""
    try:
        import transformers
        return transformers
    except ImportError:
        raise TrainerError(
            "Scheduler requires the 'transformers' package. "
            "Install it with: pip install transformers"
        )


# ---------------------------------------------------------------------------
# Registry: name -> factory(optimizer, num_training_steps, warmup_steps,
#                            min_lr_ratio, **kwargs) -> LRScheduler
# ---------------------------------------------------------------------------

def _make_cosine(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    t = _require_transformers()
    return t.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )


def _make_constant(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    t = _require_transformers()
    return t.get_constant_schedule(optimizer, **kwargs)


def _make_constant_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    t = _require_transformers()
    return t.get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, **kwargs,
    )


def _make_linear(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    t = _require_transformers()
    return t.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )


def _make_cosine_with_restarts(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    t = _require_transformers()
    num_cycles = kwargs.pop("num_cycles", 1)
    return t.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        **kwargs,
    )


def _make_polynomial(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    t = _require_transformers()
    power = kwargs.pop("power", 1.0)
    return t.get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        power=power,
        **kwargs,
    )


def _make_rex(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    # Determine max_lr from the first param group
    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = max_lr * min_lr_ratio if min_lr_ratio > 0.0 else max_lr * 0.01
    return RexLR(
        optimizer,
        max_lr=max_lr,
        min_lr=min_lr,
        num_steps=num_training_steps,
        num_warmup_steps=warmup_steps,
        **kwargs,
    )


def _make_exponential(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    gamma = kwargs.pop("gamma", 0.9999)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return gamma ** (step - warmup_steps)

    return LambdaLR(optimizer, lr_lambda)


def _make_inverse_sqrt(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    **kwargs: Any,
) -> LRScheduler:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return (max(warmup_steps, 1) ** 0.5) / (max(step, 1) ** 0.5)

    return LambdaLR(optimizer, lr_lambda)


# Type alias for scheduler factory callables
SchedulerFactory = Callable[
    [torch.optim.Optimizer, int, int, float],
    LRScheduler,
]

SCHEDULERS: dict[str, SchedulerFactory] = {
    "cosine": _make_cosine,
    "constant": _make_constant,
    "constant_with_warmup": _make_constant_with_warmup,
    "linear": _make_linear,
    "cosine_with_restarts": _make_cosine_with_restarts,
    "polynomial": _make_polynomial,
    "rex": _make_rex,
    "exponential": _make_exponential,
    "inverse_sqrt": _make_inverse_sqrt,
}


def list_schedulers() -> list[str]:
    """Return the list of built-in scheduler names."""
    return sorted(SCHEDULERS.keys())


# ---------------------------------------------------------------------------
# Dynamic import for dotted-path scheduler types
# ---------------------------------------------------------------------------

def _import_scheduler_class(dotted_path: str) -> type[LRScheduler]:
    """Import a scheduler class from a fully-qualified dotted path.

    Example: ``"torch.optim.lr_scheduler.StepLR"``
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise TrainerError(
            f"Invalid scheduler path '{dotted_path}'. "
            "Expected 'module.ClassName' (e.g. 'torch.optim.lr_scheduler.StepLR')."
        )
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise TrainerError(
            f"Could not import module '{module_path}' for scheduler "
            f"'{dotted_path}': {e}"
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

def create_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    **kwargs: Any,
) -> LRScheduler:
    """Create a learning rate scheduler by name.

    Args:
        scheduler_type: Name from ``SCHEDULERS`` (case-insensitive) or a
            fully-qualified dotted path (e.g. ``"torch.optim.lr_scheduler.StepLR"``).
        optimizer: The optimizer whose learning rate will be scheduled.
        num_training_steps: Total number of training steps.
        warmup_steps: Number of warmup steps (default 0).
        min_lr_ratio: Minimum LR as a fraction of peak LR (default 0.0).
            Used by ``rex`` and can be forwarded to custom schedulers.
        **kwargs: Extra keyword arguments forwarded to the scheduler constructor.

    Returns:
        A configured ``LRScheduler`` instance.

    Raises:
        TrainerError: If the scheduler type is unknown or its dependency is missing.
    """
    name = scheduler_type.lower()

    if name in SCHEDULERS:
        factory = SCHEDULERS[name]
        scheduler = factory(
            optimizer, num_training_steps, warmup_steps, min_lr_ratio, **kwargs,
        )
        logger.info(
            f"Created scheduler: {name} "
            f"(steps={num_training_steps}, warmup={warmup_steps})"
        )
        return scheduler

    if "." in scheduler_type:
        # Dynamic import - use original casing for the class name
        scheduler_cls = _import_scheduler_class(scheduler_type)
        try:
            scheduler = scheduler_cls(optimizer, **kwargs)
        except TypeError as e:
            raise TrainerError(
                f"Failed to create scheduler '{scheduler_type}': {e}"
            ) from e
        logger.info(f"Created scheduler: {scheduler_type} (dynamic import)")
        return scheduler

    available = ", ".join(list_schedulers())
    raise TrainerError(
        f"Unknown scheduler '{scheduler_type}'. "
        f"Built-in options: {available}. "
        f"Or provide a dotted path like 'torch.optim.lr_scheduler.StepLR'."
    )
