"""Tests for fused backward pass (per-parameter optimizer stepping)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(shape=(4, 4), requires_grad=True) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape, requires_grad=requires_grad))


def _make_adamw(params, lr=1e-3) -> AdamW:
    return AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


def _simple_loss(params: list[nn.Parameter]) -> torch.Tensor:
    """Dummy loss that exercises all params."""
    return sum((p ** 2).sum() for p in params)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class TestImport:
    def test_module_imports(self):
        from trainer.training.fused_backward import FusedBackwardManager
        assert FusedBackwardManager is not None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_sets_is_registered(self):
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)
        assert not mgr.is_registered

        mgr.register()
        assert mgr.is_registered

    def test_register_creates_hooks(self):
        from trainer.training.fused_backward import FusedBackwardManager

        params = [_make_param(), _make_param()]
        opt = _make_adamw(params)
        mgr = FusedBackwardManager(opt)
        mgr.register()

        # Both trainable params should have a hook registered.
        assert len(mgr._hooks) == 2

    def test_double_register_raises(self):
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)
        mgr.register()

        with pytest.raises(RuntimeError, match="already registered"):
            mgr.register()

    def test_non_trainable_params_skipped(self):
        from trainer.training.fused_backward import FusedBackwardManager

        frozen = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        trainable = _make_param()
        opt = _make_adamw([frozen, trainable])
        mgr = FusedBackwardManager(opt)
        mgr.register()

        # Only the trainable param should have a hook.
        assert len(mgr._hooks) == 1


# ---------------------------------------------------------------------------
# is_registered property
# ---------------------------------------------------------------------------

class TestIsRegistered:
    def test_false_before_register(self):
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)
        assert mgr.is_registered is False

    def test_true_after_register(self):
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)
        mgr.register()
        assert mgr.is_registered is True

    def test_false_after_remove(self):
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)
        mgr.register()
        mgr.remove()
        assert mgr.is_registered is False


# ---------------------------------------------------------------------------
# Hook firing: grad is None after backward
# ---------------------------------------------------------------------------

class TestHookFiring:
    def test_grad_is_none_after_backward(self):
        """After backward with fused hooks, all param grads must be None."""
        from trainer.training.fused_backward import FusedBackwardManager

        params = [_make_param(), _make_param()]
        opt = _make_adamw(params)
        mgr = FusedBackwardManager(opt)
        mgr.register()

        loss = _simple_loss(params)
        loss.backward()

        for p in params:
            assert p.grad is None, (
                "Expected grad to be None after fused backward step"
            )

    def test_params_change_after_backward(self):
        """Param values must change after a fused backward pass (update applied)."""
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        original_data = param.data.clone()

        opt = _make_adamw([param], lr=1.0)  # Large lr to ensure visible change.
        mgr = FusedBackwardManager(opt)
        mgr.register()

        loss = (param ** 2).sum()
        loss.backward()

        assert not torch.equal(param.data, original_data), (
            "Param data should have changed after fused backward"
        )


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_clears_hooks(self):
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)
        mgr.register()
        assert len(mgr._hooks) == 1

        mgr.remove()
        assert len(mgr._hooks) == 0

    def test_remove_disables_hooks(self):
        """After remove(), backward should NOT apply fused updates (hooks gone)."""
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)
        mgr.register()
        mgr.remove()

        # After removal, grads should accumulate normally.
        loss = (param ** 2).sum()
        loss.backward()

        # Grad should now be non-None (normal behaviour).
        assert param.grad is not None, (
            "After remove(), backward should accumulate grads normally"
        )


# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------

class TestConfigField:
    def test_default_is_false(self):
        from trainer.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.fused_backward is False

    def test_can_be_set_true(self):
        from trainer.config.schema import TrainingConfig

        cfg = TrainingConfig(fused_backward=True)
        assert cfg.fused_backward is True

    def test_round_trip(self):
        from trainer.config.schema import TrainingConfig

        cfg = TrainingConfig(fused_backward=True)
        dumped = cfg.model_dump()
        restored = TrainingConfig.model_validate(dumped)
        assert restored.fused_backward is True


# ---------------------------------------------------------------------------
# AdamW update direction
# ---------------------------------------------------------------------------

class TestAdamWUpdateDirection:
    """Verify that fused AdamW steps move parameters in the right direction."""

    def test_gradient_descent(self):
        """For loss = sum(p^2), grad is 2*p, so the update must decrease p toward 0."""
        from trainer.training.fused_backward import FusedBackwardManager

        # Use a positive-valued parameter so the gradient is clearly positive.
        param = nn.Parameter(torch.ones(8, 8) * 2.0)
        original_data = param.data.clone()

        opt = _make_adamw([param], lr=1e-2)
        mgr = FusedBackwardManager(opt)
        mgr.register()

        loss = (param ** 2).sum()
        loss.backward()

        # For a positive param with loss = p^2, update should reduce magnitude.
        assert (param.data.abs() < original_data.abs()).all(), (
            "Fused AdamW should reduce parameter magnitude toward 0 for loss=p^2"
        )

    def test_same_direction_as_standard_adamw(self):
        """Fused update and standard AdamW must move params in the same direction."""
        from trainer.training.fused_backward import FusedBackwardManager

        torch.manual_seed(42)
        data = torch.randn(8, 8)

        # --- Fused path ---
        param_fused = nn.Parameter(data.clone())
        opt_fused = _make_adamw([param_fused], lr=1e-3)
        mgr = FusedBackwardManager(opt_fused)
        mgr.register()
        loss_fused = (param_fused ** 2).sum()
        loss_fused.backward()
        fused_update = param_fused.data.clone()

        # --- Standard path ---
        param_std = nn.Parameter(data.clone())
        opt_std = _make_adamw([param_std], lr=1e-3)
        loss_std = (param_std ** 2).sum()
        loss_std.backward()
        opt_std.step()
        opt_std.zero_grad()
        std_update = param_std.data.clone()

        # Both should have moved from the initial value.
        assert not torch.equal(fused_update, data), "Fused param should have changed"
        assert not torch.equal(std_update, data), "Standard param should have changed"

        # The directions should agree (both decrease, or both increase for each element).
        fused_delta = fused_update - data
        std_delta = std_update - data
        # Sign agreement for at least 90% of elements.
        sign_match = (torch.sign(fused_delta) == torch.sign(std_delta)).float().mean()
        assert sign_match >= 0.9, (
            f"Fused and standard AdamW disagree on update direction "
            f"for {(1 - sign_match) * 100:.1f}% of elements"
        )


# ---------------------------------------------------------------------------
# PyTorch version guard
# ---------------------------------------------------------------------------

class TestPyTorchVersionGuard:
    def test_raises_on_missing_hook_api(self):
        """FusedBackwardManager.register() raises AttributeError if hook API absent."""
        from trainer.training.fused_backward import FusedBackwardManager
        import torch

        param = _make_param()
        opt = _make_adamw([param])
        mgr = FusedBackwardManager(opt)

        # Temporarily hide the API to simulate old PyTorch.
        original = getattr(torch.Tensor, "register_post_accumulate_grad_hook", None)
        try:
            if original is not None:
                delattr(torch.Tensor, "register_post_accumulate_grad_hook")
            with pytest.raises(AttributeError, match="PyTorch 2.1"):
                mgr.register()
        finally:
            if original is not None:
                torch.Tensor.register_post_accumulate_grad_hook = original  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# SGD fallback for non-AdamW optimizers
# ---------------------------------------------------------------------------

class TestSGDFallback:
    def test_sgd_fallback_still_updates_params(self):
        """Non-AdamW optimizer falls back to SGD; params must still be updated."""
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        original = param.data.clone()
        opt = SGD([param], lr=1e-2)
        mgr = FusedBackwardManager(opt)
        mgr.register()

        loss = (param ** 2).sum()
        loss.backward()

        assert not torch.equal(param.data, original), (
            "SGD fallback hook should still update param data"
        )

    def test_sgd_fallback_frees_grad(self):
        """SGD fallback must free the gradient after stepping."""
        from trainer.training.fused_backward import FusedBackwardManager

        param = _make_param()
        opt = SGD([param], lr=1e-2)
        mgr = FusedBackwardManager(opt)
        mgr.register()

        loss = (param ** 2).sum()
        loss.backward()

        assert param.grad is None, "SGD fallback should free gradient after stepping"


# ---------------------------------------------------------------------------
# Multi-param AdamW state isolation
# ---------------------------------------------------------------------------

class TestAdamWStateIsolation:
    def test_each_param_has_independent_state(self):
        """Optimizer state must be initialised independently for each parameter."""
        from trainer.training.fused_backward import FusedBackwardManager

        params = [_make_param((4,)), _make_param((8,)), _make_param((16,))]
        opt = _make_adamw(params)
        mgr = FusedBackwardManager(opt)
        mgr.register()

        # Each param should have its own state entry.
        for p in params:
            assert p in opt.state, "Param missing from optimizer.state after register"
            state = opt.state[p]
            assert "exp_avg" in state
            assert "exp_avg_sq" in state
            assert state["exp_avg"].shape == p.shape, (
                f"exp_avg shape {state['exp_avg'].shape} != param shape {p.shape}"
            )
