"""DoRA (Weight-Decomposed LoRA) module tests.

Tests cover:
- Forward shape preservation for Linear
- Backward produces gradients on magnitude, lora_down, lora_up
- Initialization: magnitude matches original weight column norms
- At init (zero lora_up), output approximately equals original output
- Conv2d raises ValueError
- Module dropout works
- save/load roundtrip via state_dict
- Registered in get_module_class("dora")
- use_dora config field works
- Works with container apply_to
"""
import math
import tempfile
import os
import pytest
import torch
import torch.nn as nn

from trainer.networks.dora import DoRAModule
from trainer.networks.container import NetworkContainer
from trainer.networks import get_module_class
from trainer.config.schema import NetworkConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear(in_dim: int = 32, out_dim: int = 16, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    # Non-trivial weights so column norms are meaningful
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    return layer


def _make_dora(in_dim: int = 32, out_dim: int = 16, lora_dim: int = 4, bias: bool = True) -> DoRAModule:
    layer = _make_linear(in_dim, out_dim, bias=bias)
    return DoRAModule("test_dora", layer, multiplier=1.0, lora_dim=lora_dim, alpha=lora_dim)


class _TestBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class _TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _TestBlock()
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(self.block(x))


def _freeze(model):
    for p in model.parameters():
        p.requires_grad = False


# ---------------------------------------------------------------------------
# 1. Forward shape preservation
# ---------------------------------------------------------------------------

class TestDoRAForwardShape:
    def test_2d_input(self):
        dora = _make_dora(32, 16)
        dora.apply_to()
        x = torch.randn(8, 32)
        out = dora(x)
        assert out.shape == (8, 16)

    def test_3d_input(self):
        """Sequence-shaped input as used in transformer attention."""
        dora = _make_dora(32, 16)
        dora.apply_to()
        x = torch.randn(2, 10, 32)
        out = dora(x)
        assert out.shape == (2, 10, 16)

    def test_no_bias(self):
        dora = _make_dora(32, 16, bias=False)
        dora.apply_to()
        x = torch.randn(4, 32)
        out = dora(x)
        assert out.shape == (4, 16)


# ---------------------------------------------------------------------------
# 2. Backward - gradients on all trainable params
# ---------------------------------------------------------------------------

class TestDoRABackward:
    def test_gradients_flow(self):
        dora = _make_dora(32, 16, lora_dim=4)
        dora.apply_to()

        x = torch.randn(4, 32)
        out = dora(x)
        out.sum().backward()

        assert dora.magnitude.grad is not None, "magnitude should have gradient"
        assert dora.lora_down.weight.grad is not None, "lora_down.weight should have gradient"
        assert dora.lora_up.weight.grad is not None, "lora_up.weight should have gradient"

    def test_magnitude_grad_nonzero(self):
        """Magnitude gradient must be non-zero (it is part of the computation graph)."""
        dora = _make_dora(32, 16, lora_dim=4)
        dora.apply_to()

        x = torch.randn(4, 32)
        out = dora(x)
        out.sum().backward()

        assert dora.magnitude.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 3. Magnitude initialized from original weight column norms
# ---------------------------------------------------------------------------

class TestDoRAInit:
    def test_magnitude_matches_column_norms(self):
        layer = _make_linear(32, 16)
        expected_norms = layer.weight.detach().float().norm(dim=1)

        dora = DoRAModule("test", layer, lora_dim=4)

        assert torch.allclose(dora.magnitude.data.float(), expected_norms, atol=1e-5), (
            "Magnitude should be initialized to column (dim=1) norms of the original weight"
        )

    def test_alpha_buffer_in_state_dict(self):
        dora = _make_dora(32, 16)
        sd = dora.state_dict()
        assert "alpha" in sd, "alpha buffer should be in state_dict"

    def test_magnitude_in_state_dict(self):
        dora = _make_dora(32, 16)
        sd = dora.state_dict()
        assert "magnitude" in sd, "magnitude parameter should be in state_dict"

    def test_magnitude_shape(self):
        out_dim = 16
        dora = _make_dora(32, out_dim)
        assert dora.magnitude.shape == (out_dim,)


# ---------------------------------------------------------------------------
# 4. At init (lora_up zeros), output matches original
# ---------------------------------------------------------------------------

class TestDoRAInitEquivalence:
    def test_output_matches_original_at_init(self):
        """With lora_up zeros, merged_weight = W_0 and DoRA should reproduce
        the original output (up to floating-point precision from the norm/magnitude ops)."""
        torch.manual_seed(42)
        layer = _make_linear(32, 16)

        x = torch.randn(4, 32)
        expected = layer(x).detach()

        dora = DoRAModule("test", layer, lora_dim=4, alpha=4)
        # Note: after DoRAModule.__init__, lora_up is zeros.
        # We call apply_to AFTER capturing the expected output so we're
        # comparing against the pre-apply forward.
        dora.apply_to()

        out = dora(x)

        # The equivalence holds because:
        # merged_weight = W_0 + 0 = W_0
        # weight_norm = ||W_0||_col  (detached)
        # mag_weight = magnitude * (W_0 / weight_norm) = ||W_0||_col * (W_0 / ||W_0||_col) = W_0
        assert torch.allclose(out, expected, atol=1e-4), (
            f"DoRA with zero lora_up should match original output. "
            f"Max diff: {(out - expected).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# 5. Conv2d raises ValueError
# ---------------------------------------------------------------------------

class TestDoRAConv2dRejection:
    def test_conv2d_raises(self):
        conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        with pytest.raises(ValueError, match="DoRA only supports Linear"):
            DoRAModule("test_conv", conv, lora_dim=4)

    def test_conv2d_1x1_raises(self):
        conv = nn.Conv2d(8, 16, kernel_size=1)
        with pytest.raises(ValueError, match="DoRA only supports Linear"):
            DoRAModule("test_conv1x1", conv, lora_dim=4)


# ---------------------------------------------------------------------------
# 6. Module dropout
# ---------------------------------------------------------------------------

class TestDoRAModuleDropout:
    def test_module_dropout_returns_org_output(self):
        """With module_dropout=1.0, DoRA should always skip and return org output."""
        layer = _make_linear(32, 16)
        x = torch.randn(4, 32)
        expected = layer(x).detach()

        dora = DoRAModule("test", layer, lora_dim=4, module_dropout=1.0)
        dora.apply_to()
        dora.train()  # dropout only active in training mode

        # With p=1.0, should always fall through to org_forward
        out = dora(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_module_dropout_zero_passes_through(self):
        """module_dropout=0.0 never skips DoRA path."""
        dora = _make_dora(32, 16)
        # Replace lora_up with non-zero weights to distinguish DoRA from original
        nn.init.normal_(dora.lora_up.weight)
        dora.module_dropout = 0.0
        dora.apply_to()
        dora.train()

        x = torch.randn(4, 32)
        # Just verify it runs without error
        out = dora(x)
        assert out.shape == (4, 16)

    def test_module_dropout_inactive_in_eval(self):
        """module_dropout should have no effect in eval mode."""
        layer = _make_linear(32, 16)
        dora = DoRAModule("test", layer, lora_dim=4, module_dropout=1.0)
        dora.apply_to()
        dora.eval()  # eval mode disables dropout

        x = torch.randn(4, 32)
        out1 = dora(x)
        out2 = dora(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# 7. State dict save/load roundtrip
# ---------------------------------------------------------------------------

class TestDoRAStateDictRoundtrip:
    def test_roundtrip_preserves_values(self):
        dora = _make_dora(32, 16, lora_dim=4)
        # Give lora_up non-zero weights so the save/load is meaningful
        nn.init.normal_(dora.lora_up.weight)

        sd_before = {k: v.clone() for k, v in dora.state_dict().items()}
        dora.apply_to()

        # Simulate saving and loading by creating a fresh module and loading state
        layer2 = _make_linear(32, 16)
        dora2 = DoRAModule("test_dora", layer2, lora_dim=4, alpha=4)

        dora2.load_state_dict(sd_before)

        sd_after = dora2.state_dict()
        for key in sd_before:
            assert torch.allclose(sd_before[key].float(), sd_after[key].float(), atol=1e-6), (
                f"State dict mismatch for key '{key}'"
            )

    def test_state_dict_keys(self):
        """After apply_to(), org_module is deleted, so only DoRA's own params/buffers remain."""
        dora = _make_dora(32, 16, lora_dim=4)
        dora.apply_to()
        sd = dora.state_dict()
        expected_keys = {"lora_down.weight", "lora_up.weight", "magnitude", "alpha"}
        assert expected_keys == set(sd.keys()), (
            f"Unexpected state_dict keys. Got: {set(sd.keys())}"
        )


# ---------------------------------------------------------------------------
# 8. Registered in get_module_class
# ---------------------------------------------------------------------------

class TestDoRARegistry:
    def test_get_module_class_dora(self):
        cls = get_module_class("dora")
        assert cls is DoRAModule

    def test_get_module_class_unknown_still_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_module_class("nonexistent_module_xyz")

    def test_dora_in_error_message(self):
        """The error message for unknown type should mention dora."""
        with pytest.raises(ValueError, match="dora"):
            get_module_class("nonexistent_module_xyz")


# ---------------------------------------------------------------------------
# 9. use_dora config field
# ---------------------------------------------------------------------------

class TestDoRAConfigField:
    def test_use_dora_default_false(self):
        cfg = NetworkConfig()
        assert cfg.use_dora is False

    def test_use_dora_can_be_set_true(self):
        cfg = NetworkConfig(use_dora=True)
        assert cfg.use_dora is True

    def test_use_dora_with_network_type(self):
        cfg = NetworkConfig(network_type="lora", use_dora=True)
        assert cfg.use_dora is True
        assert cfg.network_type == "lora"


# ---------------------------------------------------------------------------
# 10. Works with container apply_to
# ---------------------------------------------------------------------------

class TestDoRAContainerIntegration:
    def test_container_apply_and_forward(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=DoRAModule,
            target_modules=["_TestBlock"],
            exclude_patterns=[],
            rank=4,
            alpha=4.0,
        )
        net.apply_to(model)
        assert len(net.lora_modules) == 2

        out = model(torch.randn(2, 64))
        assert out.shape == (2, 10)

    def test_container_backward(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=DoRAModule,
            target_modules=["_TestBlock"],
            exclude_patterns=[],
            rank=4,
            alpha=4.0,
        )
        net.apply_to(model)

        out = model(torch.randn(2, 64))
        out.sum().backward()

        for lora in net.lora_modules:
            assert any(p.grad is not None for p in lora.parameters()), (
                f"Module {lora.lora_name} has no gradients"
            )

    def test_container_optimizer_params(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=DoRAModule,
            target_modules=["_TestBlock"],
            exclude_patterns=[],
            rank=4,
            alpha=4.0,
        )
        net.apply_to(model)
        params, _ = net.prepare_optimizer_params(unet_lr=1e-4)
        assert len(params) > 0
        total = sum(sum(p.numel() for p in g["params"]) for g in params)
        assert total > 0

    def test_container_save_load(self):
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=DoRAModule,
            target_modules=["_TestBlock"],
            exclude_patterns=[],
            rank=4,
            alpha=4.0,
        )
        net.apply_to(model)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dora_test.safetensors")
            net.save_weights(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_magnitude_trainable_in_container(self):
        """magnitude must appear as a trainable parameter via the container."""
        model = _TestModel()
        _freeze(model)
        net = NetworkContainer(
            module_class=DoRAModule,
            target_modules=["_TestBlock"],
            exclude_patterns=[],
            rank=4,
            alpha=4.0,
        )
        net.apply_to(model)

        magnitude_params = [
            (name, p)
            for name, p in net.named_parameters()
            if "magnitude" in name
        ]
        assert len(magnitude_params) == 2, (
            f"Expected 2 magnitude params (one per linear), got {len(magnitude_params)}"
        )
        for name, p in magnitude_params:
            assert p.requires_grad, f"{name} should require grad"
