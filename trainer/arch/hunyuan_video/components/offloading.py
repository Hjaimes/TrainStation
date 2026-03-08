"""Block-swap offloading for HunyuanVideo.

Self-contained copy of Musubi_Tuner's custom_offloading_utils.py.
Improvements:
  - Removed print() statements, replaced with logger calls
  - Removed dead/commented-out code
  - Module is self-contained - do NOT import from wan's utils.py
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
    """Move only Linear weight tensors to device (non-blocking where possible)."""
    non_blocking = device.type != "cpu"
    for module in layer.modules():
        if (
            hasattr(module, "weight")
            and module.weight is not None
            and module.__class__.__name__.endswith("Linear")
        ):
            module.weight.data = module.weight.data.to(device, non_blocking=non_blocking)


# ---------------------------------------------------------------------------
# Offloader base class
# ---------------------------------------------------------------------------

class Offloader:
    """Base class for concurrent CPU↔GPU weight swapping."""

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ) -> None:
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self.debug = debug
        self.debug_block_count = 0

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures: dict = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for overlap of CPU↔GPU transfers
        self.staging_buffer_a: Optional[list] = None
        self.staging_buffer_b: Optional[list] = None
        self.pinned_buffer: Optional[list] = None

    def _swap_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        """Swap weights between CPU and CUDA with overlap using staging buffers."""
        debug_print = self.debug and (self.debug_block_count % 10 == 0)
        self.debug_block_count += 1

        class _Timer:
            def __init__(self, enabled: bool):
                self.enabled = enabled
                self.totals: dict = defaultdict(float)

            @contextmanager
            def section(self, name: str):
                if not self.enabled:
                    yield
                    return
                t0 = time.perf_counter()
                try:
                    yield
                finally:
                    self.totals[name] += time.perf_counter() - t0

        T = _Timer(debug_print)

        # Collect weight swap jobs
        with T.section("find modules"):
            modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
            weight_swap_jobs = []
            for name, mod_cuda in layer_to_cuda.named_modules():
                if (
                    hasattr(mod_cuda, "weight")
                    and mod_cuda.weight is not None
                    and mod_cuda.__class__.__name__.endswith("Linear")
                ):
                    mod_cpu = modules_to_cpu.get(name)
                    if mod_cpu is not None and mod_cpu.weight.shape == mod_cuda.weight.shape:
                        weight_swap_jobs.append(
                            (mod_cpu, mod_cuda, mod_cpu.weight.data, mod_cuda.weight.data)
                        )
                    elif mod_cuda.weight.data.device.type != device.type:
                        mod_cuda.weight.data = mod_cuda.weight.data.to(device)

        with T.section("sync before swap"):
            torch.cuda.current_stream().synchronize()

        if not self.use_pinned_memory:
            stream = self.stream
            with torch.cuda.stream(stream):
                if self.staging_buffer_a is None:
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_view, _ in weight_swap_jobs
                    ]

                event_b = None
                for sbuf_a, sbuf_b, (mod_cpu, mod_cuda, cuda_view, cpu_view) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    event_a = torch.cuda.Event()
                    with T.section("cuda to staging A"):
                        sbuf_a.copy_(cuda_view.data, non_blocking=True)
                        event_a.record(stream)

                    if event_b is not None:
                        with T.section("wait staging B"):
                            event_b.synchronize()

                    with T.section("cpu to staging B"):
                        sbuf_b.copy_(mod_cuda.weight.data)

                    with T.section("wait staging A"):
                        event_a.synchronize()

                    event_b = torch.cuda.Event()
                    with T.section("staging B to CUDA"):
                        cuda_view.copy_(sbuf_b, non_blocking=True)
                        event_b.record(stream)

                    with T.section("staging A to CPU"):
                        cpu_view.copy_(sbuf_a)

            for sbuf_a, sbuf_b, (mod_cpu, mod_cuda, cuda_view, cpu_view) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                mod_cuda.weight.data = cuda_view
                mod_cpu.weight.data = cpu_view

            sync_event = event_b

        else:
            # Pinned memory path
            if self.pinned_buffer is None:
                with torch.cuda.stream(self.stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_view, _ in weight_swap_jobs
                    ]
                self.stream.synchronize()

            released = []
            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            for event, pin_buf, (_, mod_cuda, cuda_view, _) in zip(events, self.pinned_buffer, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    pin_buf.copy_(cuda_view, non_blocking=True)
                    event.record(self.stream)

            for event, (mod_cpu, mod_cuda, cuda_view, cpu_view) in zip(events, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(event)
                    cuda_view.copy_(cpu_view, non_blocking=True)

            for pin_buf, (mod_cpu, mod_cuda, cuda_view, cpu_view) in zip(self.pinned_buffer, weight_swap_jobs):
                mod_cuda.weight.data = cuda_view
                mod_cpu.weight.data = pin_buf
                released.append(cpu_view)

            if not released[0].is_pinned():
                with torch.cuda.stream(self.stream):
                    released = [
                        torch.empty_like(cuda_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_view, _ in weight_swap_jobs
                    ]
            self.pinned_buffer = released
            sync_event = self.stream.record_event()

        if debug_print:
            logger.debug(
                "[%s] Weight swap at step %d: %s",
                self.block_type,
                self.debug_block_count - 1,
                {k: f"{v * 1000:.1f}ms" for k, v in T.totals.items()},
            )

        return sync_event

    def _swap_non_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module) -> None:
        weight_jobs = []
        for mod_cpu, mod_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
            if hasattr(mod_cpu, "weight") and mod_cpu.weight is not None:
                weight_jobs.append((mod_cpu, mod_cuda, mod_cpu.weight.data, mod_cuda.weight.data))

        for mod_cpu, mod_cuda, cuda_view, cpu_view in weight_jobs:
            mod_cpu.weight.data = cuda_view.to("cpu", non_blocking=True)

        _synchronize_device(device)

        for mod_cpu, mod_cuda, cuda_view, cpu_view in weight_jobs:
            cuda_view.copy_(mod_cuda.weight.data, non_blocking=True)
            mod_cuda.weight.data = cuda_view

        _synchronize_device(device)

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            return self._swap_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            self._swap_non_cuda(self.device, block_to_cpu, block_to_cuda)
            return None

    def _submit_move_blocks(self, blocks: list, block_idx_to_cpu: int, block_idx_to_cuda: int) -> None:
        def move_blocks(bidx_cpu, block_cpu, bidx_cuda, block_cuda):
            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)
            sync_event = self.swap_weight_devices(block_cpu, block_cuda)
            return bidx_cpu, bidx_cuda, sync_event

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks,
            block_idx_to_cpu, blocks[block_idx_to_cpu],
            block_idx_to_cuda, blocks[block_idx_to_cuda],
        )

    def _wait_blocks_move(self, block_idx: int) -> None:
        if block_idx not in self.futures:
            return
        future = self.futures.pop(block_idx)
        _, bidx_cuda, sync_event = future.result()
        assert block_idx == bidx_cuda, f"Block index mismatch: {block_idx} != {bidx_cuda}"
        if self.cuda_available and sync_event is not None:
            torch.cuda.current_stream().wait_event(sync_event)


# ---------------------------------------------------------------------------
# ModelOffloader - forward + backward support
# ---------------------------------------------------------------------------

class ModelOffloader(Offloader):
    """Block-swap offloader supporting both forward and backward passes."""

    def __init__(
        self,
        block_type: str,
        blocks: list,
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory, debug)
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward
        self.remove_handles: list = []

        if self.supports_backward:
            for i, block in enumerate(blocks):
                hook = self._create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def __del__(self) -> None:
        for handle in self.remove_handles:
            handle.remove()

    def set_forward_only(self, forward_only: bool) -> None:
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)
        self.forward_only = forward_only

    def _create_backward_hook(self, blocks: list, block_index: int):
        num_propagated = self.num_blocks - block_index - 1
        swapping = 0 < num_propagated <= self.blocks_to_swap
        waiting = 0 < block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        bidx_to_cpu = self.num_blocks - num_propagated
        bidx_to_cuda = self.blocks_to_swap - num_propagated
        bidx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if swapping:
                self._submit_move_blocks(blocks, bidx_to_cpu, bidx_to_cuda)
            if waiting:
                self._wait_blocks_move(bidx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list) -> None:
        if not self.blocks_to_swap:
            return

        # Blocks that stay on GPU throughout
        for b in blocks[: self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            _weights_to_device(b, self.device)

        # Blocks that start on CPU (weights), but need buffers on GPU
        cpu_device = torch.device("cpu")
        for b in blocks[self.num_blocks - self.blocks_to_swap:]:
            b.to(self.device)
            _weights_to_device(b, cpu_device)

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int) -> None:
        if self.blocks_to_swap:
            self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list, block_idx: int) -> None:
        if not self.blocks_to_swap:
            return

        if not self.forward_only:
            if block_idx >= self.blocks_to_swap:
                return
            bidx_cpu = block_idx
            bidx_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            self._submit_move_blocks(blocks, bidx_cpu, bidx_cuda)
            return

        # Forward-only: two strategies based on ratio of blocks_to_swap / num_blocks
        bidx_cpu = block_idx
        if self.blocks_to_swap < (self.num_blocks // 2):
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                bidx_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                bidx_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            bidx_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks

        self._submit_move_blocks(blocks, bidx_cpu, bidx_cuda)
