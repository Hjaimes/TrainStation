"""Fused backward pass - per-parameter optimizer stepping during backward.

Instead of the standard pattern:
    loss.backward()       # accumulate ALL gradients
    optimizer.step()      # step ALL params at once
    optimizer.zero_grad() # clear ALL grads

Fused backward does:
    loss.backward()  ->  as each param's gradient is computed,
                         immediately step that param and zero its grad.

This means only one parameter's gradient is in memory at a time, saving ~25-40%
VRAM. The savings come purely from not storing all gradients simultaneously - 
optimizer states (exp_avg, exp_avg_sq) are still fully in memory.

Uses torch.Tensor.register_post_accumulate_grad_hook() (PyTorch 2.1+).

Constraints:
    - Single-GPU only (bypasses Accelerate gradient sync).
    - No VRAM benefit with gradient_accumulation > 1 (full grads needed for
      accumulation steps; only the final micro-step would benefit).
    - Gradient clipping (max_grad_norm) is incompatible - grads are freed
      immediately and cannot be globally normalised.
    - Only AdamW is supported natively. Other optimizers fall back to a plain
      SGD-style update (no momentum/variance) with a warning.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# Recognised AdamW class names (covers PyTorch built-in and common variants).
_ADAMW_CLASS_NAMES = frozenset({
    "AdamW",
    "AdamWBnb",         # bitsandbytes
    "PagedAdamW",       # bitsandbytes paged
    "AdamW8bit",
    "AdamW32bit",
})


def _is_adamw(optimizer: Optimizer) -> bool:
    """Return True if the optimizer is an AdamW variant we can handle."""
    return type(optimizer).__name__ in _ADAMW_CLASS_NAMES


class FusedBackwardManager:
    """Manages per-parameter gradient hooks for fused backward stepping.

    After register(), each trainable parameter gets a post-accumulate-grad hook
    that:
        1. Steps the optimizer for just that parameter (AdamW or SGD fallback).
        2. Zeros the gradient immediately to free the memory.

    The normal optimizer.step() and optimizer.zero_grad() calls MUST be skipped
    when fused backward is active - see Trainer integration notes.

    Example::

        manager = FusedBackwardManager(optimizer)
        manager.register()
        loss.backward()           # hooks fire; grads freed parameter-by-parameter
        scheduler.step()          # still needed
        # Do NOT call optimizer.step() or optimizer.zero_grad()
        manager.remove()          # clean up at end of training
    """

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self._hooks: list[Any] = []
        self._registered = False
        self._use_adamw = _is_adamw(optimizer)

        if not self._use_adamw:
            logger.warning(
                "Fused backward: optimizer %s is not a recognised AdamW variant. "
                "Falling back to plain SGD-style update (no momentum/variance). "
                "This will produce different results than a full optimizer step.",
                type(optimizer).__name__,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Register post-accumulate-grad hooks on all trainable parameters.

        Raises:
            RuntimeError: If already registered.
            AttributeError: If PyTorch < 2.1 (hook API not available).
        """
        if self._registered:
            raise RuntimeError("FusedBackwardManager is already registered")

        if not hasattr(Tensor, "register_post_accumulate_grad_hook"):
            raise AttributeError(
                "Fused backward requires PyTorch 2.1+ "
                "(Tensor.register_post_accumulate_grad_hook not available)"
            )

        n_hooks = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self._use_adamw:
                    # Initialise optimizer state eagerly so hooks can access it.
                    self._init_adamw_state(p)
                    hook = p.register_post_accumulate_grad_hook(
                        self._make_adamw_hook(p, group)
                    )
                else:
                    hook = p.register_post_accumulate_grad_hook(
                        self._make_sgd_hook(p, group)
                    )
                self._hooks.append(hook)
                n_hooks += 1

        self._registered = True
        logger.info(
            "Fused backward: registered %d hooks (%s)",
            n_hooks,
            "AdamW" if self._use_adamw else "SGD fallback",
        )

    def remove(self) -> None:
        """Remove all registered hooks and reset state."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._registered = False
        logger.info("Fused backward: removed all hooks")

    @property
    def is_registered(self) -> bool:
        """True if hooks are currently registered."""
        return self._registered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_adamw_state(self, param: Tensor) -> None:
        """Eagerly create AdamW moment buffers in optimizer.state for param."""
        state = self.optimizer.state[param]
        if not state:
            state["step"] = 0  # Python int - avoids GPU sync from .item()
            state["exp_avg"] = torch.zeros_like(param.data)
            state["exp_avg_sq"] = torch.zeros_like(param.data)

    def _make_adamw_hook(self, param: Tensor, group: dict):
        """Return a closure implementing the AdamW update for a single param.

        The update follows the standard decoupled AdamW formula:
            p = p * (1 - lr * wd)                         [weight decay]
            m = beta1 * m + (1 - beta1) * g               [first moment]
            v = beta2 * v + (1 - beta2) * g^2             [second moment]
            p = p - lr / bc1 * m / (sqrt(v / bc2) + eps)  [param update]
        """
        optimizer = self.optimizer
        state = optimizer.state[param]

        # Capture hyperparams at registration time - only lr changes via
        # scheduler; betas/eps/wd are fixed for the lifetime of the param group.
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps: float = group.get("eps", 1e-8)
        wd: float = group.get("weight_decay", 0.01)

        def hook(p: Tensor) -> None:
            if p.grad is None:
                return

            grad = p.grad

            # Re-read lr each step so scheduler changes are respected.
            lr: float = group["lr"]

            with torch.no_grad():
                # Python int step counter - avoids GPU->CPU sync from tensor .item()
                step = state["step"] + 1
                state["step"] = step

                # Decoupled weight decay (AdamW style, applied before gradient).
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                # First moment: m = beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Second moment: v = beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction (scalar math, no tensor ops)
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step

                # Compute denominator: sqrt(v) / sqrt(bc2) + eps
                # Using scalar sqrt(bc2) avoids allocating a full-size temp tensor
                # that (exp_avg_sq / bc2) would produce.
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bc2)).add_(eps)

                # Param update: p = p - (lr / bc1) * m / denom
                step_size = lr / bc1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Free gradient immediately to reclaim VRAM.
                p.grad = None

        return hook

    def _make_sgd_hook(self, param: Tensor, group: dict):
        """Return a plain SGD-style hook (fallback for non-AdamW optimizers).

        Only applies weight decay and a gradient step - no momentum or variance.
        This is mathematically equivalent to SGD with the given lr/weight_decay,
        NOT to whatever the original optimizer does.
        """

        def hook(p: Tensor) -> None:
            if p.grad is None:
                return

            grad = p.grad
            lr: float = group["lr"]
            wd: float = group.get("weight_decay", 0.0)

            with torch.no_grad():
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(grad, alpha=-lr)
                p.grad = None

        return hook
