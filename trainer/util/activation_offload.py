"""Activation offloading - CPU-offloaded gradient checkpointing.

Moves saved-for-backward tensors to CPU pinned memory during forward pass,
pre-fetches back to GPU during backward pass. Reduces GPU VRAM at the cost
of CPU-GPU transfer overhead.

Usage:
    ctx = ActivationOffloadContext()
    with ctx:
        output = model(input)
        loss = criterion(output, target)
    loss.backward()  # activations fetched from CPU as needed
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ActivationOffloadContext:
    """Context manager that offloads saved tensors to CPU pinned memory.

    When entering the context:
    - pack_hook: GPU tensor → CPU pinned tensor (saves GPU VRAM)
    - unpack_hook: CPU pinned tensor → GPU tensor (restores for backward)

    Uses CUDA streams for async transfer when available.

    The context wraps the forward pass only. Call ``loss.backward()`` outside
    the context; PyTorch will invoke unpack_hook as each activation is needed.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize activation offload context.

        Args:
            enabled: If False, the context manager is a no-op. This allows
                callers to construct the object unconditionally and control
                behaviour via config without branching at the call site.
        """
        self.enabled = enabled
        self._ctx: Any | None = None
        # Create stream once at init and reuse across enter/exit cycles.
        # Creating a new CUDA stream per training step adds overhead from
        # repeated CUDA driver allocation calls.
        self._stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if enabled and torch.cuda.is_available() else None
        )

    def __enter__(self) -> "ActivationOffloadContext":
        if not self.enabled:
            return self

        stream = self._stream

        def pack_hook(tensor: Tensor) -> tuple[Tensor, torch.device]:
            """Move tensor to CPU pinned memory during forward pass."""
            if not tensor.is_cuda:
                # Non-CUDA tensors pass through unchanged.
                return (tensor, tensor.device)

            # Allocate pinned CPU tensor and async-copy from GPU.
            cpu_tensor = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            cpu_tensor.copy_(tensor, non_blocking=True)
            return (cpu_tensor, tensor.device)

        def unpack_hook(packed: tuple[Tensor, torch.device]) -> Tensor:
            """Move tensor back to GPU during backward pass."""
            cpu_tensor, device = packed
            if device.type != "cuda":
                # Was never on GPU - return as-is.
                return cpu_tensor

            if stream is not None:
                with torch.cuda.stream(stream):
                    gpu_tensor = cpu_tensor.to(device, non_blocking=True)
                # Ensure transfer completes before autograd reads the tensor.
                stream.synchronize()
                return gpu_tensor

            return cpu_tensor.to(device)

        self._ctx = torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)
        self._ctx.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if not self.enabled or self._ctx is None:
            return
        self._ctx.__exit__(*args)
        self._ctx = None
        # Keep self._stream alive - reused across enter/exit cycles.
