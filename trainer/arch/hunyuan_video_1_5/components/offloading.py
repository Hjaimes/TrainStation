"""Block-swap CPU/GPU offloader for HunyuanVideo 1.5.

HV 1.5 is simpler than original HV: a single ModelOffloader for double blocks
only (no single-stream blocks to manage).

This is a self-contained copy of ModelOffloader from Musubi_Tuner, ported
with improvements:
- print() → logger.info()/logger.debug()
- Removed logging.basicConfig()
- Removed unused non-CUDA path for pinned-memory mode
- Type annotations added
"""
from __future__ import annotations

import gc
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _clean_memory_on_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _weights_to_device(layer: nn.Module, device: torch.device) -> None:
    """Move only Linear weight data to *device* (non-blocking when going to GPU)."""
    for module in layer.modules():
        if (
            hasattr(module, "weight")
            and module.weight is not None
            and module.__class__.__name__.endswith("Linear")
        ):
            module.weight.data = module.weight.data.to(
                device, non_blocking=(device.type != "cpu")
            )


def _swap_weights_no_cuda(
    device: torch.device,
    layer_to_cpu: nn.Module,
    layer_to_cuda: nn.Module,
) -> None:
    """Synchronous weight swap for non-CUDA devices."""
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__
    jobs = []
    for mod_cpu, mod_gpu in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(mod_cpu, "weight") and mod_cpu.weight is not None:
            jobs.append((mod_cpu, mod_gpu, mod_cpu.weight.data, mod_gpu.weight.data))

    for mod_cpu, _, cuda_view, _ in jobs:
        mod_cpu.weight.data = cuda_view.to("cpu", non_blocking=True)
    _synchronize_device(device)

    for _, mod_gpu, cuda_view, cpu_view in jobs:
        cuda_view.copy_(mod_gpu.weight.data, non_blocking=True)
        mod_gpu.weight.data = cuda_view
    _synchronize_device(device)


# ---------------------------------------------------------------------------
# Offloader base
# ---------------------------------------------------------------------------

class Offloader:
    """Thread-pool-backed asynchronous weight swapper."""

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
    ) -> None:
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures: dict = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        self.staging_buffer_a: list | None = None
        self.staging_buffer_b: list | None = None
        self.pinned_buffer: list | None = None

    def _swap_weights_cuda(self, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        """Asynchronous CUDA weight swap using staging buffers."""
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        # Build job list by matching named modules
        modules_by_name = {k: v for k, v in layer_to_cpu.named_modules()}
        jobs = []
        for name, mod_gpu in layer_to_cuda.named_modules():
            if (
                hasattr(mod_gpu, "weight")
                and mod_gpu.weight is not None
                and mod_gpu.__class__.__name__.endswith("Linear")
            ):
                mod_cpu = modules_by_name.get(name)
                if mod_cpu is not None and mod_cpu.weight.shape == mod_gpu.weight.shape:
                    jobs.append((mod_cpu, mod_gpu, mod_cpu.weight.data, mod_gpu.weight.data))
                elif mod_gpu.weight.data.device.type != self.device.type:
                    mod_gpu.weight.data = mod_gpu.weight.data.to(self.device)

        torch.cuda.current_stream().synchronize()

        if not self.use_pinned_memory:
            stream = self.stream
            with torch.cuda.stream(stream):
                if self.staging_buffer_a is None:
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_v, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_v, _ in jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_v, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_v, _ in jobs
                    ]

                event_b = None
                for sbuf_a, sbuf_b, (mod_cpu, mod_gpu, cuda_v, cpu_v) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, jobs
                ):
                    event_a = torch.cuda.Event()
                    sbuf_a.copy_(cuda_v.data, non_blocking=True)
                    event_a.record(stream)

                    if event_b is not None:
                        event_b.synchronize()

                    sbuf_b.copy_(mod_gpu.weight.data)

                    event_a.synchronize()

                    event_b = torch.cuda.Event()
                    cuda_v.copy_(sbuf_b, non_blocking=True)
                    event_b.record(stream)

                    cpu_v.copy_(sbuf_a)

            for sbuf_a, sbuf_b, (mod_cpu, mod_gpu, cuda_v, cpu_v) in zip(
                self.staging_buffer_a, self.staging_buffer_b, jobs
            ):
                mod_gpu.weight.data = cuda_v
                mod_cpu.weight.data = cpu_v

            return event_b

        else:
            if self.pinned_buffer is None:
                with torch.cuda.stream(self.stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_v, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_v, _ in jobs
                    ]
                self.stream.synchronize()

            events = [torch.cuda.Event() for _ in jobs]
            for event, pin_buf, (mod_cpu, mod_gpu, cuda_v, cpu_v) in zip(events, self.pinned_buffer, jobs):
                with torch.cuda.stream(self.stream):
                    pin_buf.copy_(cuda_v, non_blocking=True)
                    event.record(self.stream)

            released = []
            for event, (mod_cpu, mod_gpu, cuda_v, cpu_v) in zip(events, jobs):
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(event)
                    cuda_v.copy_(cpu_v, non_blocking=True)

            for pin_buf, (mod_cpu, mod_gpu, cuda_v, cpu_v) in zip(self.pinned_buffer, jobs):
                mod_gpu.weight.data = cuda_v
                mod_cpu.weight.data = pin_buf
                released.append(cpu_v)

            if not released[0].is_pinned():
                with torch.cuda.stream(self.stream):
                    released = [
                        torch.empty_like(cuda_v, device="cpu").pin_memory(device=self.device)
                        for _, _, cuda_v, _ in jobs
                    ]
            self.pinned_buffer = released
            return self.stream.record_event()

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            return self._swap_weights_cuda(block_to_cpu, block_to_cuda)
        _swap_weights_no_cuda(self.device, block_to_cpu, block_to_cuda)
        return None

    def _submit_move_blocks(self, blocks: list[nn.Module], idx_to_cpu: int, idx_to_cuda: int) -> None:
        def move(bidx_cpu: int, block_cpu: nn.Module, bidx_cuda: int, block_cuda: nn.Module):
            if self.cuda_available:
                dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
                torch.cuda.set_device(dev)
            sync_event = self.swap_weight_devices(block_cpu, block_cuda)
            return bidx_cpu, bidx_cuda, sync_event

        self.futures[idx_to_cuda] = self.thread_pool.submit(
            move, idx_to_cpu, blocks[idx_to_cpu], idx_to_cuda, blocks[idx_to_cuda]
        )

    def _wait_blocks_move(self, block_idx: int) -> None:
        if block_idx not in self.futures:
            return
        future = self.futures.pop(block_idx)
        _, bidx_cuda, sync_event = future.result()
        assert block_idx == bidx_cuda
        if self.cuda_available and sync_event is not None:
            torch.cuda.current_stream().wait_event(sync_event)


# ---------------------------------------------------------------------------
# ModelOffloader (forward + backward)
# ---------------------------------------------------------------------------

class ModelOffloader(Offloader):
    """Supports forward-pass and backward-pass block swapping."""

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
    ) -> None:
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory)
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward
        self.remove_handles: list = []

        if supports_backward:
            for i, block in enumerate(blocks):
                hook = self._create_backward_hook(blocks, i)
                if hook is not None:
                    self.remove_handles.append(block.register_full_backward_hook(hook))

    def __del__(self) -> None:
        for handle in self.remove_handles:
            handle.remove()

    def set_forward_only(self, forward_only: bool) -> None:
        for idx in list(self.futures.keys()):
            self._wait_blocks_move(idx)
        self.forward_only = forward_only

    def _create_backward_hook(
        self, blocks: list[nn.Module], block_index: int
    ) -> Optional[callable]:
        num_propagated = self.num_blocks - block_index - 1
        swapping = 0 < num_propagated <= self.blocks_to_swap
        waiting = 0 < block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        idx_to_cpu = self.num_blocks - num_propagated
        idx_to_cuda = self.blocks_to_swap - num_propagated
        idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if swapping:
                self._submit_move_blocks(blocks, idx_to_cpu, idx_to_cuda)
            if waiting:
                self._wait_blocks_move(idx_to_wait)

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]) -> None:
        """Ensure first N blocks are on GPU and last blocks_to_swap on CPU."""
        if not self.blocks_to_swap:
            return

        on_device = self.num_blocks - self.blocks_to_swap
        cpu = torch.device("cpu")

        for b in blocks[:on_device]:
            b.to(self.device)
            _weights_to_device(b, self.device)

        for b in blocks[on_device:]:
            b.to(self.device)  # move buffers to device
            _weights_to_device(b, cpu)  # but weights stay on CPU

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int) -> None:
        if self.blocks_to_swap:
            self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int) -> None:
        if not self.blocks_to_swap:
            return

        if not self.forward_only:
            if block_idx >= self.blocks_to_swap:
                return
            idx_cpu = block_idx
            idx_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            self._submit_move_blocks(blocks, idx_cpu, idx_cuda)
            return

        idx_cpu = block_idx
        if self.blocks_to_swap < self.num_blocks // 2:
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                idx_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                idx_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            idx_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks

        self._submit_move_blocks(blocks, idx_cpu, idx_cuda)
