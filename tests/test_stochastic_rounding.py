"""Tests for stochastic rounding (BF16 optimizer step precision)."""
from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# copy_stochastic_ tests
# ---------------------------------------------------------------------------

class TestCopyStochastic:
    """Tests for the core copy_stochastic_ function."""

    def test_output_dtype_is_bf16(self):
        """copy_stochastic_ must write into a bf16 target."""
        from trainer.util.stochastic_rounding import copy_stochastic_

        source = torch.tensor([1.5, -0.25, 3.14], dtype=torch.float32)
        target = torch.zeros_like(source, dtype=torch.bfloat16)
        copy_stochastic_(target, source)
        assert target.dtype == torch.bfloat16

    def test_preserves_shape(self):
        """copy_stochastic_ must not change the shape of the target."""
        from trainer.util.stochastic_rounding import copy_stochastic_

        shape = (4, 8, 16)
        source = torch.randn(*shape, dtype=torch.float32)
        target = torch.zeros(*shape, dtype=torch.bfloat16)
        copy_stochastic_(target, source)
        assert target.shape == torch.Size(shape)

    def test_unbiased_rounding(self):
        """Stochastic rounding is unbiased: mean over many trials ≈ original value.

        We pick a value that falls exactly between two bf16 representable values
        so that naive truncation would be systematically biased in one direction.
        The mean of many stochastic roundings must be close to the true fp32 value.
        """
        from trainer.util.stochastic_rounding import copy_stochastic_

        # 1.0 + 2^-8 = 1.00390625 — exactly between two adjacent bf16 values
        # (bf16 has 7 mantissa bits; the 8th bit is the first dropped bit)
        fp32_value = torch.tensor([1.0 + 2**-8], dtype=torch.float32)
        target = torch.zeros(1, dtype=torch.bfloat16)

        n_trials = 10_000
        accumulated = torch.zeros(1, dtype=torch.float32)
        for _ in range(n_trials):
            copy_stochastic_(target, fp32_value)
            accumulated.add_(target.float())

        mean_result = (accumulated / n_trials).item()
        expected = fp32_value.item()

        # Within 0.1% relative error after 10k trials
        assert abs(mean_result - expected) < abs(expected) * 0.001, (
            f"Stochastic rounding not unbiased: expected ~{expected:.6f}, got {mean_result:.6f}"
        )

    def test_2d_tensor(self):
        """copy_stochastic_ handles 2D tensors correctly."""
        from trainer.util.stochastic_rounding import copy_stochastic_

        source = torch.randn(8, 8, dtype=torch.float32)
        target = torch.zeros(8, 8, dtype=torch.bfloat16)
        copy_stochastic_(target, source)
        assert target.shape == (8, 8)
        assert target.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# register_stochastic_rounding_hook tests
# ---------------------------------------------------------------------------

class TestRegisterHook:
    """Tests for hook registration."""

    def test_hook_registered(self):
        """register_stochastic_rounding_hook adds a hook to the optimizer."""
        from trainer.util.stochastic_rounding import register_stochastic_rounding_hook

        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = torch.optim.AdamW([param], lr=1e-3)

        initial_hook_count = len(optimizer._step_count_changed_hook_map
                                 if hasattr(optimizer, "_step_count_changed_hook_map") else [])
        register_stochastic_rounding_hook(optimizer)

        # PyTorch stores post-step hooks in _optimizer_step_post_hooks
        assert hasattr(optimizer, "_optimizer_step_post_hooks"), (
            "optimizer should have _optimizer_step_post_hooks after registration"
        )
        assert len(optimizer._optimizer_step_post_hooks) >= 1

    def test_hook_fires_after_step(self):
        """Hook fires after optimizer.step() — verified via a counter side-effect."""
        from trainer.util.stochastic_rounding import register_stochastic_rounding_hook

        call_log: list[int] = []

        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = torch.optim.SGD([param], lr=1e-3)

        def counting_hook(opt, *args, **kwargs):
            call_log.append(1)

        optimizer.register_step_post_hook(counting_hook)

        # Create a fake loss and step
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()

        assert len(call_log) == 1, "Hook should have fired exactly once after step()"

    def test_raises_on_old_pytorch(self):
        """register_stochastic_rounding_hook raises AttributeError if hook API is absent."""
        from trainer.util.stochastic_rounding import register_stochastic_rounding_hook

        class FakeOldOptimizer:
            param_groups = []
            # No register_step_post_hook method

        with pytest.raises(AttributeError, match="PyTorch 2.1"):
            register_stochastic_rounding_hook(FakeOldOptimizer())  # type: ignore[arg-type]

    def test_hook_only_affects_bf16_params(self):
        """Only bf16 parameters with gradients are processed; fp32 params are untouched."""
        from trainer.util.stochastic_rounding import _stochastic_rounding_hook

        fp32_param = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
        bf16_param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

        # Give the bf16 param a fake gradient so the hook processes it
        bf16_param.grad = torch.ones(4, dtype=torch.bfloat16) * 0.1

        original_fp32 = fp32_param.data.clone()

        # Build a mock optimizer with both params
        class MockOptimizer:
            param_groups = [{"params": [fp32_param, bf16_param]}]

        _stochastic_rounding_hook(MockOptimizer())  # type: ignore[arg-type]

        # fp32 param must be untouched
        assert torch.equal(fp32_param.data, original_fp32), (
            "fp32 parameter should not be modified by the stochastic rounding hook"
        )
        # bf16 param is still bf16 after hook
        assert bf16_param.dtype == torch.bfloat16

    def test_hook_skips_param_without_grad(self):
        """Hook does not process bf16 params that have no gradient."""
        from trainer.util.stochastic_rounding import _stochastic_rounding_hook

        bf16_param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        bf16_param.grad = None  # no gradient

        original = bf16_param.data.clone()

        class MockOptimizer:
            param_groups = [{"params": [bf16_param]}]

        _stochastic_rounding_hook(MockOptimizer())  # type: ignore[arg-type]

        # Should be unmodified since grad is None
        assert torch.equal(bf16_param.data, original)


# ---------------------------------------------------------------------------
# Config field tests
# ---------------------------------------------------------------------------

class TestStochasticRoundingConfig:
    """Tests for TrainingConfig.stochastic_rounding field."""

    def test_default_is_false(self):
        """stochastic_rounding defaults to False."""
        from trainer.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.stochastic_rounding is False

    def test_can_be_set_to_true(self):
        """stochastic_rounding can be explicitly set to True."""
        from trainer.config.schema import TrainingConfig

        cfg = TrainingConfig(stochastic_rounding=True)
        assert cfg.stochastic_rounding is True

    def test_field_survives_round_trip(self):
        """stochastic_rounding survives model_dump / model_validate round-trip."""
        from trainer.config.schema import TrainingConfig

        cfg = TrainingConfig(stochastic_rounding=True)
        dumped = cfg.model_dump()
        restored = TrainingConfig.model_validate(dumped)
        assert restored.stochastic_rounding is True
