"""Tests for weight bouncing — per-layer CPU-pinned weight storage."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")

# Device used for "GPU" in CPU-only tests — just CPU, so transfers are no-ops
_CPU_DEVICE = torch.device("cpu")


def _make_linear(in_f: int = 8, out_f: int = 4, bias: bool = True) -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(in_f, out_f, bias=bias)


# ---------------------------------------------------------------------------
# BouncingLinear — basic properties
# ---------------------------------------------------------------------------

class TestBouncingLinearProperties:
    """Basic properties of BouncingLinear instances."""

    def test_weights_on_cpu(self):
        """BouncingLinear stores weight on CPU regardless of the target device."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear()
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        assert bl.weight.device.type == "cpu", (
            f"weight should be on CPU, got {bl.weight.device}"
        )

    def test_bias_on_cpu(self):
        """BouncingLinear stores bias on CPU."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear(bias=True)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        assert bl.bias is not None
        assert bl.bias.device.type == "cpu", (
            f"bias should be on CPU, got {bl.bias.device}"
        )

    def test_no_bias_case(self):
        """BouncingLinear handles bias=None correctly."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear(bias=False)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        assert bl.bias is None

    @_skip_no_cuda
    def test_weights_pinned_on_cuda_device(self):
        """Weights are in pinned memory when CUDA is available."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear()
        bl = BouncingLinear.from_linear(linear, torch.device("cuda"))

        assert bl.weight.is_pinned(), "weight should be in pinned memory"

    def test_from_linear_class_method(self):
        """from_linear produces a BouncingLinear with matching shape."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = nn.Linear(16, 8, bias=True)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        assert isinstance(bl, BouncingLinear)
        assert bl.weight.shape == linear.weight.shape
        assert bl.bias is not None
        assert bl.bias.shape == linear.bias.shape


# ---------------------------------------------------------------------------
# BouncingLinear — forward correctness
# ---------------------------------------------------------------------------

class TestBouncingLinearForward:
    """Forward pass produces correct outputs."""

    def test_output_shape_2d(self):
        """Forward pass produces correct output shape for 2-D input."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear(in_f=8, out_f=4)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(3, 8)
        out = bl(x)

        assert out.shape == (3, 4), f"Expected (3, 4), got {out.shape}"

    def test_output_shape_3d(self):
        """Forward pass handles 3-D input (batch, seq, features)."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear(in_f=8, out_f=4)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(2, 5, 8)
        out = bl(x)

        assert out.shape == (2, 5, 4), f"Expected (2, 5, 4), got {out.shape}"

    def test_output_matches_nn_linear(self):
        """BouncingLinear output exactly matches nn.Linear output."""
        from trainer.util.weight_bouncing import BouncingLinear

        torch.manual_seed(42)
        linear = _make_linear(in_f=8, out_f=4, bias=True)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(5, 8)

        expected = linear(x)
        actual = bl(x)

        assert torch.allclose(expected, actual, atol=1e-6), (
            f"Output mismatch: max diff = {(expected - actual).abs().max().item()}"
        )

    def test_output_matches_nn_linear_no_bias(self):
        """BouncingLinear output matches nn.Linear when bias=False."""
        from trainer.util.weight_bouncing import BouncingLinear

        torch.manual_seed(7)
        linear = _make_linear(in_f=6, out_f=3, bias=False)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(4, 6)

        expected = linear(x)
        actual = bl(x)

        assert torch.allclose(expected, actual, atol=1e-6)


# ---------------------------------------------------------------------------
# BouncingLinear — backward / gradient correctness
# ---------------------------------------------------------------------------

class TestBouncingLinearBackward:
    """Backward pass produces correct gradients."""

    def test_backward_produces_input_grad(self):
        """Backward pass populates gradient for the input tensor."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear()
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(3, 8, requires_grad=True)
        out = bl(x)
        out.sum().backward()

        assert x.grad is not None, "Input gradient should be populated after backward"
        assert x.grad.shape == x.shape

    def test_backward_produces_weight_grad(self):
        """Backward pass populates gradient for the weight parameter."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear()
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(3, 8)
        out = bl(x)
        out.sum().backward()

        assert bl.weight.grad is not None, "Weight gradient should be populated"
        assert bl.weight.grad.shape == bl.weight.shape

    def test_backward_produces_bias_grad(self):
        """Backward pass populates gradient for the bias parameter."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear(bias=True)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(3, 8)
        out = bl(x)
        out.sum().backward()

        assert bl.bias is not None
        assert bl.bias.grad is not None, "Bias gradient should be populated"

    def test_weight_grad_on_cpu(self):
        """Weight gradient lives on CPU to match parameter location."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear()
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x = torch.randn(3, 8)
        bl(x).sum().backward()

        assert bl.weight.grad is not None
        assert bl.weight.grad.device.type == "cpu", (
            f"Weight grad should be on CPU, got {bl.weight.grad.device}"
        )

    def test_gradients_match_nn_linear(self):
        """BouncingLinear gradients match those from an equivalent nn.Linear."""
        from trainer.util.weight_bouncing import BouncingLinear

        torch.manual_seed(99)
        linear = _make_linear(in_f=8, out_f=4, bias=True)
        bl = BouncingLinear.from_linear(linear, _CPU_DEVICE)

        x_ref = torch.randn(5, 8, requires_grad=True)
        x_bl = x_ref.detach().clone().requires_grad_(True)

        # Reference: nn.Linear
        out_ref = linear(x_ref)
        out_ref.sum().backward()

        # Bouncing
        out_bl = bl(x_bl)
        out_bl.sum().backward()

        assert torch.allclose(x_ref.grad, x_bl.grad, atol=1e-5), (
            "Input gradients differ"
        )
        assert torch.allclose(linear.weight.grad, bl.weight.grad, atol=1e-5), (
            "Weight gradients differ"
        )
        assert torch.allclose(linear.bias.grad, bl.bias.grad, atol=1e-5), (
            "Bias gradients differ"
        )

    def test_gradcheck(self):
        """torch.autograd.gradcheck verifies the custom Function's Jacobian."""
        from trainer.util.weight_bouncing import _BouncingLinearFn

        torch.manual_seed(0)
        # gradcheck requires float64
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(2, dtype=torch.float64, requires_grad=True)
        device = torch.device("cpu")

        result = torch.autograd.gradcheck(
            _BouncingLinearFn.apply,
            (x, w, b, device),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        assert result, "gradcheck failed for _BouncingLinearFn"

    def test_gradcheck_no_bias(self):
        """gradcheck with bias=None."""
        from trainer.util.weight_bouncing import _BouncingLinearFn

        torch.manual_seed(1)
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        device = torch.device("cpu")

        result = torch.autograd.gradcheck(
            _BouncingLinearFn.apply,
            (x, w, None, device),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        assert result, "gradcheck failed for _BouncingLinearFn without bias"


# ---------------------------------------------------------------------------
# apply_weight_bouncing
# ---------------------------------------------------------------------------

class TestApplyWeightBouncing:
    """Tests for the apply_weight_bouncing() helper function."""

    def test_converts_all_linear_layers(self):
        """apply_weight_bouncing replaces every nn.Linear in the model."""
        from trainer.util.weight_bouncing import apply_weight_bouncing, BouncingLinear

        model = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

        apply_weight_bouncing(model, _CPU_DEVICE)

        for name, module in model.named_modules():
            if "linear" in name.lower() or isinstance(module, (nn.Linear, BouncingLinear)):
                # After conversion, there should be no plain nn.Linear left
                assert not (isinstance(module, nn.Linear) and not isinstance(module, BouncingLinear)), (
                    f"Module {name} is still a plain nn.Linear"
                )

    def test_returns_correct_count(self):
        """apply_weight_bouncing returns the number of converted layers."""
        from trainer.util.weight_bouncing import apply_weight_bouncing

        model = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Linear(2, 1),
        )

        count = apply_weight_bouncing(model, _CPU_DEVICE)

        assert count == 3, f"Expected 3 conversions, got {count}"

    def test_returns_zero_for_no_linears(self):
        """apply_weight_bouncing returns 0 when there are no Linear layers."""
        from trainer.util.weight_bouncing import apply_weight_bouncing

        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        count = apply_weight_bouncing(model, _CPU_DEVICE)

        assert count == 0

    def test_nested_model_conversion(self):
        """apply_weight_bouncing handles arbitrarily nested modules."""
        from trainer.util.weight_bouncing import apply_weight_bouncing, BouncingLinear

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 4)
                self.fc2 = nn.Linear(4, 4)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = Block()
                self.block2 = Block()
                self.head = nn.Linear(4, 1)

        model = Net()
        count = apply_weight_bouncing(model, _CPU_DEVICE)

        assert count == 5, f"Expected 5 conversions, got {count}"
        assert isinstance(model.block1.fc1, BouncingLinear)
        assert isinstance(model.block1.fc2, BouncingLinear)
        assert isinstance(model.block2.fc1, BouncingLinear)
        assert isinstance(model.block2.fc2, BouncingLinear)
        assert isinstance(model.head, BouncingLinear)

    def test_converted_model_still_works(self):
        """Model remains functional after apply_weight_bouncing."""
        from trainer.util.weight_bouncing import apply_weight_bouncing

        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))

        x = torch.randn(3, 8)
        apply_weight_bouncing(model, _CPU_DEVICE)

        out = model(x)
        assert out.shape == (3, 2)

    def test_idempotent_on_already_bouncing(self):
        """apply_weight_bouncing does not double-convert BouncingLinear layers."""
        from trainer.util.weight_bouncing import apply_weight_bouncing

        model = nn.Sequential(nn.Linear(4, 2))
        count1 = apply_weight_bouncing(model, _CPU_DEVICE)
        count2 = apply_weight_bouncing(model, _CPU_DEVICE)

        assert count1 == 1
        assert count2 == 0, "Second call should not re-convert BouncingLinear layers"


# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------

class TestWeightBouncingConfig:
    """Tests for ModelConfig.weight_bouncing field."""

    def test_default_is_false(self):
        """weight_bouncing defaults to False."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(architecture="wan", base_model_path="/tmp/model")
        assert cfg.weight_bouncing is False

    def test_can_be_set_to_true(self):
        """weight_bouncing can be explicitly enabled."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(
            architecture="wan",
            base_model_path="/tmp/model",
            weight_bouncing=True,
        )
        assert cfg.weight_bouncing is True

    def test_field_survives_round_trip(self):
        """weight_bouncing survives model_dump / model_validate round-trip."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(
            architecture="wan",
            base_model_path="/tmp/model",
            weight_bouncing=True,
        )
        restored = ModelConfig.model_validate(cfg.model_dump())
        assert restored.weight_bouncing is True

    def test_false_survives_round_trip(self):
        """Default False also survives round-trip."""
        from trainer.config.schema import ModelConfig

        cfg = ModelConfig(architecture="wan", base_model_path="/tmp/model")
        restored = ModelConfig.model_validate(cfg.model_dump())
        assert restored.weight_bouncing is False


# ---------------------------------------------------------------------------
# Pinned memory (CUDA only)
# ---------------------------------------------------------------------------

class TestPinnedMemory:
    """Verify pinned memory behaviour when CUDA is present."""

    @_skip_no_cuda
    def test_weight_is_pinned(self):
        """BouncingLinear weights are in pinned memory with CUDA available."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear()
        bl = BouncingLinear.from_linear(linear, torch.device("cuda"))

        assert bl.weight.is_pinned(), "weight must be pinned for fast async transfers"

    @_skip_no_cuda
    def test_bias_is_pinned(self):
        """BouncingLinear bias is in pinned memory with CUDA available."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear(bias=True)
        bl = BouncingLinear.from_linear(linear, torch.device("cuda"))

        assert bl.bias is not None
        assert bl.bias.is_pinned(), "bias must be pinned for fast async transfers"

    @_skip_no_cuda
    def test_forward_backward_on_cuda_input(self):
        """BouncingLinear forward/backward works with CUDA input tensor."""
        from trainer.util.weight_bouncing import BouncingLinear

        linear = _make_linear()
        cuda_device = torch.device("cuda")
        bl = BouncingLinear.from_linear(linear, cuda_device)

        x = torch.randn(4, 8, device=cuda_device, requires_grad=True)
        out = bl(x)
        out.sum().backward()

        assert out.device.type == "cuda"
        assert x.grad is not None
        assert bl.weight.grad is not None
