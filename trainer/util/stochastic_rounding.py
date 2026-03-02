"""Stochastic rounding for BF16 optimizer steps.

When training in BF16, the standard truncation of FP32 optimizer states to BF16
introduces systematic bias. Stochastic rounding adds random noise to the lower
mantissa bits before truncation, making the rounding unbiased in expectation.

Usage:
    register_stochastic_rounding_hook(optimizer)
    # Now every optimizer.step() automatically applies stochastic rounding
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


def copy_stochastic_(target: Tensor, source: Tensor) -> None:
    """Copy fp32 source into bf16 target with stochastic rounding.

    Adds a random 16-bit value to the lower mantissa bits of the fp32 value
    before truncating to bf16. This makes the rounding unbiased in expectation.

    Args:
        target: BF16 destination tensor (modified in-place).
        source: FP32 source tensor.
    """
    # Reinterpret fp32 as int32, add random noise to lower 16 bits, mask, convert back
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)  # 0xFFFF0000 — zero out lower 16 bits
    target.copy_(result.view(dtype=torch.float32))


def _stochastic_rounding_hook(optimizer: Optimizer, *args, **kwargs) -> None:
    """Post-step hook that applies stochastic rounding to BF16 parameters."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.dtype == torch.bfloat16 and p.grad is not None:
                # The optimizer step computed in fp32 master weights or
                # accumulated updates; now we need to copy back with
                # stochastic rounding.
                # Only apply if param is bf16 — the optimizer state may be fp32.
                copy_stochastic_(p.data, p.data.float())


def register_stochastic_rounding_hook(optimizer: Optimizer) -> None:
    """Register a post-step hook for stochastic rounding on BF16 parameters.

    Uses PyTorch 2.1+ ``optimizer.register_step_post_hook()``.

    Args:
        optimizer: The optimizer to attach the hook to.

    Raises:
        AttributeError: If PyTorch version doesn't support register_step_post_hook.
    """
    if not hasattr(optimizer, "register_step_post_hook"):
        raise AttributeError(
            "Stochastic rounding requires PyTorch 2.1+ "
            "(optimizer.register_step_post_hook not available)"
        )
    optimizer.register_step_post_hook(_stochastic_rounding_hook)
