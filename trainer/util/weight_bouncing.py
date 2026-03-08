"""Weight bouncing - weights on CPU pinned memory, bounced to GPU per layer.

Keeps model weights on CPU pinned memory. During forward pass, each layer's
weights are moved to GPU, used, then moved back. During backward, weights
are moved to GPU again for gradient computation, then back to CPU.

This reduces GPU VRAM to ~1 layer's weights at a time, at the cost of
CPU-GPU transfer overhead per layer.

Inspired by AI-Toolkit's approach.

Usage:
    apply_weight_bouncing(model, device)
    # Now model weights are on CPU pinned memory
    # Forward/backward automatically bounces weights to GPU as needed
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class _BouncingLinearFn(torch.autograd.Function):
    """Custom autograd.Function that bounces Linear weights from CPU to GPU.

    Forward: move weight/bias to GPU -> compute -> saved tensors used for backward.
    Backward: gradients computed on GPU -> moved back to CPU to match parameter location.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight_cpu: Tensor,
        bias_cpu: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        # Move weights to GPU for computation
        weight_gpu = weight_cpu.to(device, non_blocking=True)
        bias_gpu = bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None

        # Save GPU weight and input for backward pass
        ctx.save_for_backward(input, weight_gpu)
        ctx.bias_gpu = bias_gpu
        ctx.device = device
        # Keep a reference to the CPU tensors so backward knows where to send grads
        ctx.weight_cpu_device = weight_cpu.device
        ctx.bias_cpu_device = bias_cpu.device if bias_cpu is not None else None

        return torch.nn.functional.linear(input, weight_gpu, bias_gpu)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):  # type: ignore[override]
        input, weight_gpu = ctx.saved_tensors
        bias_gpu = ctx.bias_gpu

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad w.r.t. input: (*, out) @ (out, in) -> (*, in)
            grad_input = grad_output @ weight_gpu

        if ctx.needs_input_grad[1]:
            # grad w.r.t. weight: (out, in) = grad_output^T @ input
            if grad_output.dim() == 2:
                grad_weight = grad_output.t() @ input
            else:
                # Batched case: flatten all leading dims before matmul
                grad_weight = (
                    grad_output.reshape(-1, grad_output.shape[-1]).t()
                    @ input.reshape(-1, input.shape[-1])
                )
            # Move gradient back to CPU to match where the parameter lives
            grad_weight = grad_weight.to(ctx.weight_cpu_device, non_blocking=True)

        if bias_gpu is not None and ctx.needs_input_grad[2]:
            # Sum over all dims except the last (feature dim)
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            grad_bias = grad_bias.to(ctx.bias_cpu_device, non_blocking=True)

        # Return None for the 'device' argument (no grad for non-tensor)
        return grad_input, grad_weight, grad_bias, None


class BouncingLinear(nn.Module):
    """Drop-in replacement for nn.Linear that keeps weights on CPU pinned memory.

    Weights are "bounced" to GPU only during forward/backward via _BouncingLinearFn.
    This reduces VRAM by keeping parameters off the GPU between layer invocations,
    at the cost of CPU-GPU transfer bandwidth per forward/backward call.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        # Detach and move to CPU pinned memory for fast async transfers
        self.weight = nn.Parameter(weight.detach().cpu().pin_memory())
        if bias is not None:
            self.bias: nn.Parameter | None = nn.Parameter(bias.detach().cpu().pin_memory())
        else:
            self.bias = None
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        return _BouncingLinearFn.apply(x, self.weight, self.bias, self.device)

    @classmethod
    def from_linear(cls, linear: nn.Linear, device: torch.device) -> "BouncingLinear":
        """Convert an existing nn.Linear to a BouncingLinear."""
        return cls(linear.weight, linear.bias, device)

    def extra_repr(self) -> str:
        in_f = self.weight.shape[1]
        out_f = self.weight.shape[0]
        return f"in_features={in_f}, out_features={out_f}, bias={self.bias is not None}, device={self.device}"


def apply_weight_bouncing(model: nn.Module, device: torch.device) -> int:
    """Replace all nn.Linear layers in the model with BouncingLinear.

    Walks the module tree and replaces each nn.Linear (that is not already a
    BouncingLinear) with a BouncingLinear that stores weights on CPU pinned
    memory and transfers them to ``device`` only during forward/backward.

    Args:
        model:  The model to modify in-place.
        device: The GPU device to bounce weights to during forward/backward.

    Returns:
        Number of layers converted.
    """
    count = 0
    # Collect replacements first to avoid mutating the module tree during iteration
    replacements: list[tuple[nn.Module, str, BouncingLinear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, BouncingLinear):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        bouncing = BouncingLinear.from_linear(module, device)
        replacements.append((parent, parts[-1], bouncing))

    for parent, attr, bouncing in replacements:
        setattr(parent, attr, bouncing)
        count += 1

    if count > 0:
        logger.info("Weight bouncing: converted %d Linear layers to BouncingLinear", count)

    return count
