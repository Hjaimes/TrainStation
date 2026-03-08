"""Tests for ActivationOffloadContext (CPU-offloaded gradient checkpointing)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Context-manager lifecycle
# ---------------------------------------------------------------------------

class TestContextManager:
    """Basic context-manager protocol."""

    def test_enters_and_exits_cleanly(self):
        """ActivationOffloadContext can be used as a context manager without error."""
        from trainer.util.activation_offload import ActivationOffloadContext

        ctx = ActivationOffloadContext(enabled=True)
        with ctx:
            pass  # should not raise

    def test_disabled_is_noop(self):
        """When enabled=False the context manager is a no-op (no hook installed)."""
        from trainer.util.activation_offload import ActivationOffloadContext

        ctx = ActivationOffloadContext(enabled=False)
        with ctx:
            # Internal state should remain None - no hook was registered.
            assert ctx._ctx is None

    def test_internal_state_reset_after_exit(self):
        """After exiting the context, _ctx is None; _stream is retained for reuse."""
        from trainer.util.activation_offload import ActivationOffloadContext

        ctx = ActivationOffloadContext(enabled=True)
        with ctx:
            pass
        assert ctx._ctx is None
        # Stream is intentionally kept alive for reuse across enter/exit cycles
        # to avoid repeated CUDA driver allocation overhead.

    def test_reentrant(self):
        """The context can be entered and exited multiple times."""
        from trainer.util.activation_offload import ActivationOffloadContext

        ctx = ActivationOffloadContext(enabled=True)
        for _ in range(3):
            with ctx:
                pass


# ---------------------------------------------------------------------------
# CPU tensor pass-through
# ---------------------------------------------------------------------------

class TestCPUTensorPassThrough:
    """CPU tensors must pass through the pack/unpack hooks unchanged."""

    def test_cpu_tensor_not_moved(self):
        """pack_hook returns the original tensor when it is already on CPU."""
        from trainer.util.activation_offload import ActivationOffloadContext

        ctx = ActivationOffloadContext(enabled=True)

        # We intercept pack/unpack by running an actual forward/backward.
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # CPU
        with ctx:
            y = (x * 2).sum()
        # Backward runs outside the context - activations are retrieved via unpack_hook.
        y.backward()

        # If pass-through is correct, gradients are populated without error.
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones(3) * 2)

    def test_simple_linear_cpu(self):
        """Full forward+backward cycle works for a simple CPU linear model."""
        from trainer.util.activation_offload import ActivationOffloadContext

        layer = nn.Linear(4, 4, bias=False)
        x = torch.randn(2, 4, requires_grad=True)

        ctx = ActivationOffloadContext(enabled=True)
        with ctx:
            out = layer(x)
            loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None


# ---------------------------------------------------------------------------
# GPU-specific behaviour
# ---------------------------------------------------------------------------

class TestGPUOffload:
    """Verify pack_hook moves tensors to CPU and unpack_hook restores to GPU."""

    @_skip_no_cuda
    def test_pack_hook_moves_to_cpu(self):
        """pack_hook explicitly called with a CUDA tensor places result on CPU."""
        # We test the pack/unpack logic directly by constructing the hooks
        # the same way ActivationOffloadContext does internally.

        stream = torch.cuda.Stream()

        def pack_hook(tensor):
            if not tensor.is_cuda:
                return (tensor, tensor.device)
            cpu_tensor = torch.empty(
                tensor.shape, dtype=tensor.dtype,
                device="cpu", pin_memory=True,
            )
            cpu_tensor.copy_(tensor, non_blocking=True)
            return (cpu_tensor, tensor.device)

        def unpack_hook(packed):
            cpu_tensor, device = packed
            if device.type != "cuda":
                return cpu_tensor
            with torch.cuda.stream(stream):
                gpu_tensor = cpu_tensor.to(device, non_blocking=True)
            stream.synchronize()
            return gpu_tensor

        cuda_device = torch.device("cuda")
        tensor = torch.randn(4, 4, device=cuda_device)

        packed_tensor, packed_device = pack_hook(tensor)
        assert packed_tensor.device.type == "cpu", (
            f"Expected packed tensor on CPU, got {packed_tensor.device}"
        )
        assert packed_device == cuda_device or packed_device.type == "cuda"
        assert packed_tensor.is_pinned(), "Packed CPU tensor should be pinned"

        restored = unpack_hook((packed_tensor, cuda_device))
        assert restored.device.type == "cuda", (
            f"Expected restored tensor on CUDA, got {restored.device}"
        )
        assert torch.allclose(tensor.cpu(), restored.cpu(), atol=1e-6)

    @_skip_no_cuda
    def test_forward_backward_gpu(self):
        """Full forward+backward with activation offloading works on GPU."""
        from trainer.util.activation_offload import ActivationOffloadContext

        layer = nn.Linear(8, 8, bias=False).cuda()
        x = torch.randn(4, 8, device="cuda", requires_grad=True)

        ctx = ActivationOffloadContext(enabled=True)
        with ctx:
            out = layer(x)
            loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None

    @_skip_no_cuda
    def test_gradient_values_match_without_offload(self):
        """Gradients computed with activation offloading match the standard path."""
        from trainer.util.activation_offload import ActivationOffloadContext

        torch.manual_seed(42)
        layer_ref = nn.Linear(8, 8, bias=False).cuda()
        layer_off = nn.Linear(8, 8, bias=False).cuda()
        layer_off.weight.data.copy_(layer_ref.weight.data)

        x = torch.randn(4, 8, device="cuda")

        # Reference (no offload)
        x_ref = x.clone().requires_grad_(True)
        out_ref = layer_ref(x_ref)
        out_ref.sum().backward()

        # With offload
        x_off = x.clone().requires_grad_(True)
        ctx = ActivationOffloadContext(enabled=True)
        with ctx:
            out_off = layer_off(x_off)
        out_off.sum().backward()

        assert torch.allclose(x_ref.grad, x_off.grad, atol=1e-5), (
            "Input gradients differ between standard and offloaded paths"
        )
        assert torch.allclose(layer_ref.weight.grad, layer_off.weight.grad, atol=1e-5), (
            "Weight gradients differ between standard and offloaded paths"
        )


# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------

class TestActivationOffloadingConfig:
    """Tests for ModelConfig.activation_offloading field."""

    def test_default_is_false(self):
        """activation_offloading defaults to False."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(architecture="wan", base_model_path="/tmp/model")
        assert cfg.activation_offloading is False

    def test_can_be_set_to_true(self):
        """activation_offloading can be explicitly enabled."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(
            architecture="wan",
            base_model_path="/tmp/model",
            activation_offloading=True,
        )
        assert cfg.activation_offloading is True

    def test_field_survives_round_trip(self):
        """activation_offloading survives model_dump / model_validate round-trip."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(
            architecture="wan",
            base_model_path="/tmp/model",
            activation_offloading=True,
        )
        dumped = cfg.model_dump()
        restored = ModelConfig.model_validate(dumped)
        assert restored.activation_offloading is True

    def test_field_false_survives_round_trip(self):
        """Default False value also survives model_dump / model_validate round-trip."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(architecture="wan", base_model_path="/tmp/model")
        dumped = cfg.model_dump()
        restored = ModelConfig.model_validate(dumped)
        assert restored.activation_offloading is False
