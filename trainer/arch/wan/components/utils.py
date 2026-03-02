# Consolidated utility module for WanModel dependencies.
#
# Sources (extracted and adapted — originals are read-only reference):
#   - musubi_tuner/utils/device_utils.py
#   - musubi_tuner/modules/custom_offloading_utils.py
#   - musubi_tuner/utils/safetensors_utils.py
#   - musubi_tuner/modules/fp8_optimization_utils.py
#   - musubi_tuner/utils/lora_utils.py
#
# Copyright notices are reproduced inline where applicable.
# All musubi_tuner.* imports have been replaced with local references.

# ---------------------------------------------------------------------------
# Standard-library / third-party imports
# ---------------------------------------------------------------------------
import gc
import json
import logging
import os
import re
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ===========================================================================
# Section 1 — device_utils
# Source: musubi_tuner/utils/device_utils.py
# ===========================================================================


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    """Free cached memory on *device* (cuda / mps).  cpu is a no-op."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # not tested
        torch.mps.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    """Block until all operations on *device* are complete."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ===========================================================================
# Section 2 — custom_offloading_utils  (ModelOffloader + Offloader)
# Source: musubi_tuner/modules/custom_offloading_utils.py
# ===========================================================================

# Private helpers (kept private to avoid confusion with the public versions above)

def _clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    _synchronize_device(device)

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    _synchronize_device(device)


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None and module.__class__.__name__.endswith("Linear"):
            module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")


class Offloader:
    """
    common offloading class
    """

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory

        # check if debug is enabled from os environment variable
        if not debug:
            debug = os.getenv("MUSUBI_TUNER_OFFLOADER_DEBUG", "0") == "1"

        self.debug = debug
        self.debug_block_count = 0

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for cuda offloading without large pinned memory.
        # These are pinned memory buffers to speed up the transfer between CPU and GPU.
        self.staging_buffer_a = None
        self.staging_buffer_b = None

        # Pinned buffer for cuda offloading with pinned memory.
        # We need only one pinned buffer per layer transfer.
        self.pinned_buffer = None

    def swap_weight_devices_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        debug_print = False
        if self.debug:
            debug_print = self.debug_block_count % 10 == 0
            self.debug_block_count += 1

        class Timer:
            def __init__(self, enabled=False):
                self.enabled = enabled
                self.totals = defaultdict(float)
                self.start_time = time.perf_counter()

            @contextmanager
            def section(self, name):
                if not self.enabled:
                    yield
                    return
                t0 = time.perf_counter()
                try:
                    yield
                finally:
                    self.totals[name] += time.perf_counter() - t0

        T = Timer(enabled=debug_print)

        weight_swap_jobs = []

        with T.section("find modules"):
            modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
            for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
                if (
                    hasattr(module_to_cuda, "weight")
                    and module_to_cuda.weight is not None
                    and module_to_cuda.__class__.__name__.endswith("Linear")
                ):
                    module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                    if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                        weight_swap_jobs.append(
                            (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                        )
                    else:
                        if module_to_cuda.weight.data.device.type != device.type:
                            module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        with T.section("synchronize before swap"):
            torch.cuda.current_stream().synchronize()

        if not self.use_pinned_memory:
            stream = self.stream
            with torch.cuda.stream(stream):
                if self.staging_buffer_a is None:
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]

                event_b = None
                for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    event_a = torch.cuda.Event()
                    with T.section("cuda to staging A"):
                        sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                        event_a.record(stream)

                    if event_b is not None:
                        with T.section("wait staging B"):
                            event_b.synchronize()

                    with T.section("cpu to staging B"):
                        sbuf_b.copy_(module_to_cuda.weight.data)  # BOTTLENECK

                    with T.section("wait staging A"):
                        event_a.synchronize()

                    event_b = torch.cuda.Event()
                    with T.section("staging B to CUDA"):
                        cuda_data_view.copy_(sbuf_b, non_blocking=True)
                        event_b.record(stream)

                    with T.section("staging A to CPU"):
                        cpu_data_view.copy_(sbuf_a)  # BOTTLENECK

            for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

            sync_event = event_b

        else:
            if self.pinned_buffer is None:
                with torch.cuda.stream(self.stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                self.stream.synchronize()
            released_pinned_buffer = []

            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            for event, module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                events, self.pinned_buffer, weight_swap_jobs
            ):
                with torch.cuda.stream(self.stream):
                    with T.section("cuda to cpu"):
                        module_pin_buf.copy_(cuda_data_view, non_blocking=True)
                        event.record(self.stream)

            for event, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(events, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    with T.section("wait cpu"):
                        self.stream.wait_event(event)
                    with T.section("cpu to cuda"):
                        cuda_data_view.copy_(cpu_data_view, non_blocking=True)

            for module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.pinned_buffer, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = module_pin_buf
                released_pinned_buffer.append(cpu_data_view)

            if not released_pinned_buffer[0].is_pinned():
                with torch.cuda.stream(self.stream):
                    released_pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
            self.pinned_buffer = released_pinned_buffer

            sync_event = self.stream.record_event()

        if debug_print:
            print(f"[{self.block_type}] Weight swap timing at {self.debug_block_count - 1}:")
            for name, total in T.totals.items():
                print(f"  {name}: {total * 1000:.2f}ms")
            print(
                f"Overall time: {(time.perf_counter() - T.start_time) * 1000:.2f}ms, "
                f"total time in sections: {sum(T.totals.values()) * 1000:.2f}ms"
            )

        return sync_event

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            sync_event = self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)
            sync_event = None
        return sync_event

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} "
                    f"to {'CUDA' if self.cuda_available else 'device'}"
                )

            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} to CPU and {bidx_to_cuda} "
                    f"to {'CUDA' if self.cuda_available else 'device'} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda, sync_event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda, sync_event = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.cuda_available and sync_event is not None:
            torch.cuda.current_stream().wait_event(sync_event)

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward

        if self.supports_backward:
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)

        cpu_device = torch.device("cpu")
        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)
            weighs_to_device(b, cpu_device)

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if not self.forward_only:
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        # Two strategies for forward-only offloading:
        # 1. If blocks_to_swap < num_blocks/2: swap without wrapping (fewer swaps, for small/light models)
        # 2. If blocks_to_swap >= num_blocks/2: swap with wrapping (all blocks, for large/heavy models)

        block_idx_to_cpu = block_idx

        if self.blocks_to_swap < (self.num_blocks // 2):
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)


# ===========================================================================
# Section 3 — safetensors_utils  (MemoryEfficientSafeOpen + helpers)
# Source: musubi_tuner/utils/safetensors_utils.py
# ===========================================================================


@dataclass
class WeightTransformHooks:
    split_hook: Optional[callable] = None
    concat_hook: Optional[callable] = None


class MemoryEfficientSafeOpen:
    """Memory-efficient reader for safetensors files.

    This class provides a memory-efficient way to read tensors from safetensors
    files by using memory mapping for large tensors and avoiding unnecessary copies.
    """

    def __init__(self, filename, disable_numpy_memmap=False):
        """Initialize the SafeTensor reader.

        Args:
            filename (str): Path to the safetensors file to read.
            disable_numpy_memmap (bool): If True, disable numpy memory mapping for
                large tensors, using standard file read instead.
        """
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()
        self.disable_numpy_memmap = disable_numpy_memmap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        """Return all tensor keys (excludes __metadata__)."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """Return the file metadata dict."""
        return self.header.get("__metadata__", {})

    def _read_header(self):
        """Parse the safetensors header.

        Returns:
            tuple: (header_dict, header_size)
        """
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a tensor from the file with memory-efficient strategies.

        **Note:**
        If device is 'cuda', the transfer to GPU is done efficiently using pinned
        memory and non-blocking transfer.  Ensure the transfer is complete before
        using the tensor (e.g. ``torch.cuda.synchronize()``).

        If the tensor is large (>10 MB) and the target device is CUDA, memory
        mapping with numpy.memmap is used to avoid intermediate copies.

        Args:
            key (str): Name of the tensor to load.
            device (Optional[torch.device]): Target device for the tensor.
            dtype (Optional[torch.dtype]): Target dtype for the tensor.

        Returns:
            torch.Tensor: The loaded tensor.

        Raises:
            KeyError: If the tensor key is not found in the file.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        non_blocking = device is not None and device.type == "cuda"

        # Absolute file offset (header_size + 8-byte length prefix)
        tensor_offset = self.header_size + 8 + offset_start

        # Use memmap for large tensors going to a non-CPU device to avoid copies
        if (
            not self.disable_numpy_memmap
            and num_bytes > 10 * 1024 * 1024
            and device is not None
            and device.type != "cpu"
        ):
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            byte_tensor = torch.from_numpy(mm)
            del mm

            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)
            del byte_tensor

            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        self.file.seek(tensor_offset)
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array

        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        return deserialized_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        """Deserialize raw bytes into the correct shape and dtype."""
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        """Convert a safetensors dtype string to a torch.dtype."""
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        """Convert byte tensor to float8 format if supported.

        Raises:
            ValueError: If float8 type is not supported in the current PyTorch version.
        """
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(
                f"Unsupported float8 type: {dtype_str} "
                "(upgrade PyTorch to support float8 types)"
            )


class TensorWeightAdapter:
    """
    A wrapper for weight conversion hooks (split and concat) for use with
    MemoryEfficientSafeOpen.

    split_hook: callable(original_key, original_tensor | None)
        -> (new_keys, new_tensors) or (new_keys, None) when tensor is None
    concat_hook: callable(original_key, tensors_dict | None)
        -> (new_key, concatenated_tensor) or (new_key, None) when dict is None

    Do NOT use this as a context manager directly.
    concat_hook is not tested yet.
    """

    def __init__(self, weight_convert_hook: WeightTransformHooks, original_f: MemoryEfficientSafeOpen):
        self.original_f = original_f
        self.new_key_to_original_key_map: dict[str, Union[str, list[str]]] = {}
        self.concat_key_set = set()
        self.new_keys = []
        self.tensor_cache = {}
        self.split_hook = weight_convert_hook.split_hook
        self.concat_hook = weight_convert_hook.concat_hook

        for key in self.original_f.keys():
            if self.split_hook is not None:
                converted_keys, _ = self.split_hook(key, None)
                if converted_keys is not None:
                    for new_key in converted_keys:
                        self.new_key_to_original_key_map[new_key] = key
                    self.new_keys.extend(converted_keys)
                    continue

            if self.concat_hook is not None:
                converted_key, _ = self.concat_hook(key, None)
                if converted_key is not None:
                    if converted_key not in self.concat_key_set:
                        self.concat_key_set.add(converted_key)
                        self.new_key_to_original_key_map[converted_key] = []
                    self.new_key_to_original_key_map[converted_key].append(key)
                    self.new_keys.append(converted_key)
                    continue

            self.new_keys.append(key)

    def keys(self) -> list[str]:
        return self.new_keys

    def get_tensor(
        self,
        new_key: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if new_key not in self.new_key_to_original_key_map:
            return self.original_f.get_tensor(new_key, device=device, dtype=dtype)

        elif new_key not in self.concat_key_set:
            # split hook — cached because the original tensor is split once
            original_key = self.new_key_to_original_key_map[new_key]
            if original_key not in self.tensor_cache:
                original_tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                new_keys, new_tensors = self.split_hook(original_key, original_tensor)
                for k, t in zip(new_keys, new_tensors):
                    self.tensor_cache[k] = t
            return self.tensor_cache.pop(new_key)

        else:
            # concat hook — not cached; requested once
            tensors = {}
            for original_key in self.new_key_to_original_key_map[new_key]:
                tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                tensors[original_key] = tensor
            _, concatenated_tensors = self.concat_hook(
                self.new_key_to_original_key_map[new_key][0], tensors
            )
            return concatenated_tensors


def get_split_weight_filenames(file_path: str) -> Optional[list[str]]:
    """Return all shard paths if *file_path* follows the ``00001-of-00004`` naming
    convention, else return ``None``.
    """
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        filenames = []
        for i in range(count):
            filename = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                filenames.append(filepath)
            else:
                raise FileNotFoundError(f"File {filepath} not found")
        return filenames
    else:
        return None


# ===========================================================================
# Section 4 — fp8_optimization_utils
# Source: musubi_tuner/modules/fp8_optimization_utils.py
# ===========================================================================


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """Return the maximum representable value in FP8 format.

    Defaults to E4M3 (4-bit exponent, 3-bit mantissa, 1-bit sign).
    Only E4M3 and E5M2 with sign bit are supported.

    Args:
        exp_bits (int): Number of exponent bits.
        mantissa_bits (int): Number of mantissa bits.
        sign_bits (int): Number of sign bits (0 or 1).

    Returns:
        float: Maximum value representable in the given FP8 format.
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, "Total bits must be 8"
    if exp_bits == 4 and mantissa_bits == 3 and sign_bits == 1:
        return torch.finfo(torch.float8_e4m3fn).max
    elif exp_bits == 5 and mantissa_bits == 2 and sign_bits == 1:
        return torch.finfo(torch.float8_e5m2).max
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits} with sign_bits={sign_bits}")


def quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value):
    """Quantize *tensor* to FP8 using PyTorch's native FP8 dtype support.

    Args:
        tensor (torch.Tensor): Tensor to quantize.
        scale (float | torch.Tensor): Scale factor.
        fp8_dtype (torch.dtype): Target FP8 dtype.
        max_value (float): Maximum representable FP8 value.
        min_value (float): Minimum representable FP8 value.

    Returns:
        torch.Tensor: Quantized tensor in FP8 format.
    """
    tensor = tensor.to(torch.float32)
    tensor = torch.div(tensor, scale).nan_to_num_(0.0)
    tensor = tensor.clamp_(min=min_value, max=max_value)
    tensor = tensor.to(fp8_dtype)
    return tensor


def quantize_weight(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    max_value: float,
    min_value: float,
    quantization_mode: str = "block",
    block_size: int = 64,
):
    """Quantize a single weight tensor to FP8.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (quantized_weight, scale_tensor)
    """
    original_shape = tensor.shape

    if quantization_mode == "block":
        if tensor.ndim != 2:
            quantization_mode = "tensor"
        else:
            out_features, in_features = tensor.shape
            if in_features % block_size != 0:
                quantization_mode = "channel"
                logger.warning(
                    f"Layer {key} with shape {tensor.shape} is not divisible by "
                    f"block_size {block_size}, fallback to per-channel quantization."
                )
            else:
                num_blocks = in_features // block_size
                tensor = tensor.contiguous().view(out_features, num_blocks, block_size)
    elif quantization_mode == "channel":
        if tensor.ndim != 2:
            quantization_mode = "tensor"

    if quantization_mode in ("channel", "block"):
        scale_dim = 1 if quantization_mode == "channel" else 2
        abs_w = torch.abs(tensor)
        row_max = torch.max(abs_w, dim=scale_dim, keepdim=True).values
        scale = row_max / max_value
    else:
        tensor_max = torch.max(torch.abs(tensor).view(-1))
        scale = tensor_max / max_value

    scale = torch.clamp(scale, min=1e-8)
    scale = scale.to(torch.float32)

    quantized_weight = quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value)

    if quantization_mode == "block":
        quantized_weight = quantized_weight.view(original_shape)

    return quantized_weight, scale


def optimize_state_dict_with_fp8(
    state_dict: dict,
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[list[str]] = None,
    exclude_layer_keys: Optional[list[str]] = None,
    exp_bits: int = 4,
    mantissa_bits: int = 3,
    move_to_device: bool = False,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
):
    """Quantize Linear-layer weights in *state_dict* to FP8 in-place.

    This is a static version of ``load_safetensors_with_fp8_optimization``
    that operates on an already-loaded state dict rather than loading from
    files.

    Args:
        state_dict (dict): State dict to optimize, modified in-place.
        calc_device: Device to perform quantization on.
        target_layer_keys: Substrings that a key must contain to be targeted.
            ``None`` means all Linear weight keys are targeted.
        exclude_layer_keys: Substrings that disqualify a key from targeting.
        exp_bits (int): Number of exponent bits (4 or 5).
        mantissa_bits (int): Number of mantissa bits (3 or 2).
        move_to_device (bool): Leave quantized tensors on *calc_device*.
        quantization_mode (str): One of ``"block"``, ``"channel"``, ``"tensor"``.
        block_size (int | None): Block size for block-wise quantization.

    Returns:
        dict: The modified *state_dict*.
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value

    optimized_count = 0

    target_state_dict_keys = []
    for key in state_dict.keys():
        is_target = (
            target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            target_state_dict_keys.append(key)

    for key in tqdm(target_state_dict_keys):
        value = state_dict[key]

        original_device = value.device
        original_dtype = value.dtype

        if calc_device is not None:
            value = value.to(calc_device)

        quantized_weight, scale_tensor = quantize_weight(
            key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size
        )

        fp8_key = key
        scale_key = key.replace(".weight", ".scale_weight")

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        scale_tensor = scale_tensor.to(dtype=original_dtype, device=quantized_weight.device)

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor

        optimized_count += 1

        if calc_device is not None:
            clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def fp8_linear_forward_patch(self: nn.Linear, x, use_scaled_mm=False, max_value=None):
    """Patched forward method for Linear layers with FP8 weights.

    Args:
        self: Linear layer instance.
        x (torch.Tensor): Input tensor.
        use_scaled_mm (bool): Use ``torch._scaled_mm`` (requires SM 8.9+ / RTX 40xx).
        max_value (float | None): If given, quantize the input tensor before matmul.

    Returns:
        torch.Tensor: Result of the linear transformation.
    """
    if use_scaled_mm:
        if self.scale_weight.ndim != 1:
            raise ValueError("scaled_mm only supports per-tensor scale_weight for now.")

        input_dtype = x.dtype
        original_weight_dtype = self.scale_weight.dtype
        target_dtype = self.weight.dtype

        if max_value is None:
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)
            fp8_max_value = torch.finfo(target_dtype).max
            fp8_min_value = torch.finfo(target_dtype).min
            x = quantize_fp8(x, scale_x, target_dtype, fp8_max_value, fp8_min_value)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).to(target_dtype)

        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        if self.bias is not None:
            o = torch._scaled_mm(
                x, weight, out_dtype=original_weight_dtype, bias=self.bias,
                scale_a=scale_x, scale_b=scale_weight,
            )
        else:
            o = torch._scaled_mm(x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight)

        o = (
            o.reshape(original_shape[0], original_shape[1], -1)
            if len(original_shape) == 3
            else o.reshape(original_shape[0], -1)
        )
        return o.to(input_dtype)

    else:
        original_dtype = self.scale_weight.dtype
        if self.scale_weight.ndim < 3:
            dequantized_weight = self.weight.to(original_dtype) * self.scale_weight
        else:
            out_features, num_blocks, _ = self.scale_weight.shape
            dequantized_weight = self.weight.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
            dequantized_weight = dequantized_weight * self.scale_weight
            dequantized_weight = dequantized_weight.view(self.weight.shape)

        if self.bias is not None:
            return F.linear(x, dequantized_weight, self.bias)
        else:
            return F.linear(x, dequantized_weight)


def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False):
    """Apply monkey patching to *model* using an FP8-optimised state dict.

    Registers ``scale_weight`` buffers on each patched ``nn.Linear`` and
    replaces its ``forward`` method with the FP8 dequantising version.

    Args:
        model (nn.Module): Model to patch in-place.
        optimized_state_dict (dict): State dict containing ``*.scale_weight`` keys.
        use_scaled_mm (bool): Use ``torch._scaled_mm`` (SM 8.9+ required).

    Returns:
        nn.Module: The patched model (same instance).
    """
    max_value = None  # do not quantize input tensor

    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

    patched_count = 0

    for name, module in model.named_modules():
        has_scale = name in patched_module_paths

        if isinstance(module, nn.Linear) and has_scale:
            scale_shape = scale_shape_info[name]
            module.register_buffer("scale_weight", torch.ones(scale_shape, dtype=module.weight.dtype))

            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            module.forward = new_forward.__get__(module, type(module))
            patched_count += 1

    logger.info(f"Number of monkey-patched Linear layers: {patched_count}")
    return model


def load_safetensors_with_fp8_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict:
    """Load safetensors files and quantize Linear weights to FP8 on the fly.

    Args:
        model_files: Paths to the safetensors files to load.
        calc_device: Device used for quantisation.
        target_layer_keys: Substrings a key must contain to be targeted.
        exclude_layer_keys: Substrings that exclude a key.
        exp_bits: Exponent bits (4 or 5).
        mantissa_bits: Mantissa bits (3 or 2).
        move_to_device: Keep quantised tensors on *calc_device*.
        weight_hook: Optional callable applied to each weight before quantisation.
        quantization_mode: ``"block"``, ``"channel"``, or ``"tensor"``.
        block_size: Block size for block-wise quantisation.
        disable_numpy_memmap: Disable numpy memmap when reading.
        weight_transform_hooks: Optional split/concat hooks for the reader.

    Returns:
        dict: FP8-optimised state dict.
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value

    def is_target_key(key):
        is_target = (
            target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        return is_target and not is_excluded

    optimized_count = 0
    state_dict = {}

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
            f = (
                TensorWeightAdapter(weight_transform_hooks, original_f)
                if weight_transform_hooks is not None
                else original_f
            )

            keys = f.keys()
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)
                original_device = value.device

                if weight_hook is not None:
                    value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                if not is_target_key(key):
                    target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    value = value.to(target_device)
                    state_dict[key] = value
                    continue

                if calc_device is not None:
                    value = value.to(calc_device)

                original_dtype = value.dtype
                if original_dtype.itemsize == 1:
                    raise ValueError(
                        f"Layer {key} is already in {original_dtype} format. "
                        "`--fp8_scaled` optimization should not be applied. "
                        "Please use fp16/bf16/float32 model weights."
                    )

                quantized_weight, scale_tensor = quantize_weight(
                    key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size
                )

                fp8_key = key
                scale_key = key.replace(".weight", ".scale_weight")
                assert fp8_key != scale_key, "FP8 key and scale key must be different"

                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                scale_tensor = scale_tensor.to(dtype=original_dtype, device=quantized_weight.device)

                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


# ===========================================================================
# Section 5 — lora_utils  (load_safetensors_with_lora_and_fp8 + helpers)
# Source: musubi_tuner/utils/lora_utils.py
#
# The dynamic imports of musubi_tuner.networks.loha / lokr are replaced by
# inlined merge_weights_to_tensor implementations below so that this file
# has no dependency on the musubi_tuner package.
# ===========================================================================


# ---------------------------------------------------------------------------
# Inlined LoHa merge helper
# Source: musubi_tuner/networks/loha.py — merge_weights_to_tensor()
# ---------------------------------------------------------------------------

def _loha_merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge LoHa weights directly into a model weight tensor.

    Returns *model_weight* unchanged if no matching LoHa keys are found.
    Consumed keys are removed from *lora_weight_keys*.
    """
    w1a_key = lora_name + ".hada_w1_a"
    w1b_key = lora_name + ".hada_w1_b"
    w2a_key = lora_name + ".hada_w2_a"
    w2b_key = lora_name + ".hada_w2_b"
    alpha_key = lora_name + ".alpha"

    if w1a_key not in lora_weight_keys:
        return model_weight

    w1a = lora_sd[w1a_key].to(calc_device)
    w1b = lora_sd[w1b_key].to(calc_device)
    w2a = lora_sd[w2a_key].to(calc_device)
    w2b = lora_sd[w2b_key].to(calc_device)

    dim = w1b.shape[0]
    alpha = lora_sd.get(alpha_key, torch.tensor(dim))
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / dim

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        model_weight = model_weight.to(torch.float16)
        w1a, w1b, w2a, w2b = (
            w1a.to(torch.float16),
            w1b.to(torch.float16),
            w2a.to(torch.float16),
            w2b.to(torch.float16),
        )

    # ΔW = ((w1a @ w1b) * (w2a @ w2b)) * scale
    diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
    model_weight = model_weight + multiplier * diff_weight

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    for key in [w1a_key, w1b_key, w2a_key, w2b_key, alpha_key]:
        lora_weight_keys.discard(key)

    return model_weight


# ---------------------------------------------------------------------------
# Inlined LoKr helpers
# Source: musubi_tuner/networks/lokr.py — make_kron() + merge_weights_to_tensor()
# ---------------------------------------------------------------------------

def _make_kron(w1, w2, scale):
    """Compute Kronecker product of w1 and w2, scaled by *scale*."""
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


def _lokr_merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge LoKr weights directly into a model weight tensor.

    Returns *model_weight* unchanged if no matching LoKr keys are found.
    Consumed keys are removed from *lora_weight_keys*.
    """
    w1_key = lora_name + ".lokr_w1"
    w2_key = lora_name + ".lokr_w2"
    w2a_key = lora_name + ".lokr_w2_a"
    w2b_key = lora_name + ".lokr_w2_b"
    alpha_key = lora_name + ".alpha"

    if w1_key not in lora_weight_keys:
        return model_weight

    w1 = lora_sd[w1_key].to(calc_device)

    if w2a_key in lora_weight_keys:
        # low-rank mode: w2 = w2_a @ w2_b
        w2a = lora_sd[w2a_key].to(calc_device)
        w2b = lora_sd[w2b_key].to(calc_device)
        dim = w2a.shape[1]
        consumed_keys = [w1_key, w2a_key, w2b_key, alpha_key]
    elif w2_key in lora_weight_keys:
        # full matrix mode
        w2a = None
        w2b = None
        dim = None
        consumed_keys = [w1_key, w2_key, alpha_key]
    else:
        return model_weight

    alpha = lora_sd.get(alpha_key, None)
    if alpha is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    if w2a is not None:
        if alpha is None:
            alpha = dim
        scale = alpha / dim
    else:
        scale = 1.0

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        model_weight = model_weight.to(torch.float16)
        w1 = w1.to(torch.float16)
        if w2a is not None:
            w2a, w2b = w2a.to(torch.float16), w2b.to(torch.float16)

    if w2a is not None:
        w2 = w2a @ w2b
    else:
        w2 = lora_sd[w2_key].to(calc_device)
        if original_dtype.itemsize == 1:
            w2 = w2.to(torch.float16)

    # ΔW = kron(w1, w2) * scale
    diff_weight = _make_kron(w1, w2, scale)
    model_weight = model_weight + multiplier * diff_weight

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    for key in consumed_keys:
        lora_weight_keys.discard(key)

    return model_weight


# ---------------------------------------------------------------------------
# Public helpers (also used internally)
# ---------------------------------------------------------------------------

def detect_network_type(lora_sd: Dict[str, torch.Tensor]) -> str:
    """Detect network type (``"lora"``, ``"loha"``, or ``"lokr"``) from state dict keys."""
    for key in lora_sd:
        if "lora_down" in key:
            return "lora"
        if "hada_w1_a" in key:
            return "loha"
        if "lokr_w1" in key:
            return "lokr"
    return "lora"  # default


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Filter a LoRA state dict by include/exclude regex patterns."""
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(
            f"Filtered keys with include pattern {include_pattern}: "
            f"{original_key_count} -> {len(weights_sd.keys())}"
        )

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
        logger.info(
            f"Filtered keys with exclude pattern {exclude_pattern}: "
            f"{original_key_count_ex} -> {len(weights_sd.keys())}"
        )

    if len(weights_sd) != original_key_count:
        remaining_keys = sorted(set(k.split(".", 1)[0] for k in weights_sd.keys()))
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


def load_safetensors_with_fp8_optimization_and_hook(
    model_files: list[str],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    weight_hook: callable = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors files with optional FP8 optimisation and weight hook.

    This is the internal workhorse called by ``load_safetensors_with_lora_and_fp8``.
    """
    if fp8_optimization:
        logger.info(
            f"Loading state dict with FP8 optimization. "
            f"Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        state_dict = load_safetensors_with_fp8_optimization(
            model_files,
            calc_device,
            target_keys,
            exclude_keys,
            move_to_device=move_to_device,
            weight_hook=weight_hook,
            disable_numpy_memmap=disable_numpy_memmap,
            weight_transform_hooks=weight_transform_hooks,
        )
    else:
        logger.info(
            f"Loading state dict without FP8 optimization. "
            f"Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = (
                    TensorWeightAdapter(weight_transform_hooks, original_f)
                    if weight_transform_hooks is not None
                    else original_f
                )
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    if weight_hook is None and move_to_device:
                        value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
                    else:
                        value = f.get_tensor(key)
                        if weight_hook is not None:
                            value = weight_hook(key, value, keep_on_calc_device=move_to_device)
                        if move_to_device:
                            value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
                        elif dit_weight_dtype is not None:
                            value = value.to(dit_weight_dtype)

                    state_dict[key] = value
        if move_to_device:
            synchronize_device(calc_device)

    return state_dict


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """Merge LoRA weights into a model's state dict with optional FP8 optimisation.

    Args:
        model_files: Path to the model file(s).  Shard patterns like
            ``00001-of-00004`` are expanded automatically.
        lora_weights_list: List of LoRA state dicts to merge.  Pass ``None``
            or an empty list to skip LoRA merging.
        lora_multipliers: Scale factors for each LoRA; defaults to 1.0 each.
        fp8_optimization: Whether to apply FP8 quantisation.
        calc_device: Device used for merging / quantisation calculations.
        move_to_device: Leave tensors on *calc_device* after loading.
        dit_weight_dtype: Cast weights to this dtype when not using FP8.
        target_keys: Substrings a key must contain to be FP8-targeted.
        exclude_keys: Substrings that exclude a key from FP8 targeting.
        disable_numpy_memmap: Disable numpy memmap for safetensors reads.
        weight_transform_hooks: Optional split/concat hooks for the reader.

    Returns:
        dict[str, torch.Tensor]: Merged (and optionally FP8-quantised) state dict.
    """
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for lora_sd in lora_weights_list:
            list_of_lora_weight_keys.append(set(lora_sd.keys()))

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        lora_network_types = [detect_network_type(lora_sd) for lora_sd in lora_weights_list]

        logger.info(
            f"Merging LoRA weights into state dict. "
            f"multipliers: {lora_multipliers}, network types: {lora_network_types}"
        )

        def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, lora_network_types, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)

            for lora_weight_keys, lora_sd, multiplier, net_type in zip(
                list_of_lora_weight_keys, lora_weights_list, lora_multipliers, lora_network_types
            ):
                lora_name = model_weight_key.rsplit(".", 1)[0]  # strip trailing ".weight"
                lora_name = "lora_unet_" + lora_name.replace(".", "_")

                if net_type == "loha":
                    model_weight = _loha_merge_weights_to_tensor(
                        model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device
                    )
                elif net_type == "lokr":
                    model_weight = _lokr_merge_weights_to_tensor(
                        model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device
                    )
                else:
                    # standard LoRA (lora_down / lora_up)
                    down_key = lora_name + ".lora_down.weight"
                    up_key = lora_name + ".lora_up.weight"
                    alpha_key = lora_name + ".alpha"
                    if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
                        continue

                    down_weight = lora_sd[down_key]
                    up_weight = lora_sd[up_key]

                    dim = down_weight.size()[0]
                    alpha = lora_sd.get(alpha_key, dim)
                    scale = alpha / dim

                    down_weight = down_weight.to(calc_device)
                    up_weight = up_weight.to(calc_device)

                    original_dtype = model_weight.dtype
                    if original_dtype.itemsize == 1:  # fp8
                        model_weight = model_weight.to(torch.float16)
                        down_weight = down_weight.to(torch.float16)
                        up_weight = up_weight.to(torch.float16)

                    # W <- W + U * D
                    if len(model_weight.size()) == 2:
                        if len(up_weight.size()) == 4:
                            up_weight = up_weight.squeeze(3).squeeze(2)
                            down_weight = down_weight.squeeze(3).squeeze(2)
                        model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        # conv2d 1x1
                        model_weight = (
                            model_weight
                            + multiplier
                            * (
                                up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)
                            ).unsqueeze(2).unsqueeze(3)
                            * scale
                        )
                    else:
                        # conv2d 3x3
                        conved = torch.nn.functional.conv2d(
                            down_weight.permute(1, 0, 2, 3), up_weight
                        ).permute(1, 0, 2, 3)
                        model_weight = model_weight + multiplier * conved * scale

                    if original_dtype.itemsize == 1:
                        model_weight = model_weight.to(original_dtype)

                    lora_weight_keys.remove(down_key)
                    lora_weight_keys.remove(up_key)
                    if alpha_key in lora_weight_keys:
                        lora_weight_keys.remove(alpha_key)

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device)
            return model_weight

        weight_hook = weight_hook_func

    state_dict = load_safetensors_with_fp8_optimization_and_hook(
        model_files,
        fp8_optimization,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        target_keys,
        exclude_keys,
        weight_hook=weight_hook,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        if len(lora_weight_keys) > 0:
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}")

    return state_dict


# ---------------------------------------------------------------------------
# load_safetensors — simple state_dict loader (used by clip.py)
# Source: musubi_tuner/utils/safetensors_utils.py
# ---------------------------------------------------------------------------

def load_safetensors(
    path: str,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
) -> dict[str, torch.Tensor]:
    if disable_mmap:
        state_dict = {}
        device = torch.device(device) if device is not None else None
        with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key, device=device, dtype=dtype)
        synchronize_device(device)
        return state_dict
    else:
        from safetensors.torch import load_file
        try:
            state_dict = load_file(path, device=str(device))
        except Exception:
            state_dict = load_file(path)
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict
