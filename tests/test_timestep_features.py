"""Tests for dynamic timestep shifting (Task 13) and progressive timestep blending (Task 14)."""
from __future__ import annotations

import pytest
import torch

from trainer.arch.base import ModelStrategy


class TestApplyDynamicShift:
    """Test _apply_dynamic_shift on the base ModelStrategy class."""

    def test_returns_tensor_same_shape(self):
        t = torch.rand(8)
        result = ModelStrategy._apply_dynamic_shift(t, seq_len=512, shift_base=0.5, shift_max=1.15)
        assert result.shape == t.shape

    def test_values_in_range(self):
        """Shifted t should remain in [0, 1] for valid t inputs."""
        t = torch.linspace(0.0, 1.0, 100)
        result = ModelStrategy._apply_dynamic_shift(t, seq_len=512, shift_base=0.5, shift_max=1.15)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_identity_at_mu_one(self):
        """When mu=1.0, the formula reduces to identity: t' = t."""
        # mu = seq_len * m + b; set seq_len so mu=1.0
        # m = (1.15 - 0.5) / (4096 - 256) = 0.65 / 3840
        # b = 0.5 - m * 256
        # 1.0 = seq_len * m + b  =>  seq_len = (1.0 - b) / m
        shift_base = 0.5
        shift_max = 1.15
        base_seq_len = 256
        max_seq_len = 4096
        m = (shift_max - shift_base) / (max_seq_len - base_seq_len)
        b = shift_base - m * base_seq_len
        seq_len_identity = round((1.0 - b) / m)

        t = torch.linspace(0.01, 0.99, 50)
        result = ModelStrategy._apply_dynamic_shift(
            t, seq_len=seq_len_identity,
            shift_base=shift_base, shift_max=shift_max,
            base_seq_len=base_seq_len, max_seq_len=max_seq_len,
        )
        assert torch.allclose(result, t, atol=1e-3)

    def test_higher_seq_len_shifts_toward_higher_t(self):
        """Higher sequence lengths (more tokens) should push t toward higher values."""
        t = torch.full((100,), 0.5)
        result_low = ModelStrategy._apply_dynamic_shift(
            t, seq_len=256, shift_base=0.5, shift_max=1.15,
        )
        result_high = ModelStrategy._apply_dynamic_shift(
            t, seq_len=4096, shift_base=0.5, shift_max=1.15,
        )
        # Higher seq_len -> larger mu -> larger shift -> higher t values
        assert result_high.mean() > result_low.mean()

    def test_default_seq_len_params(self):
        """Verify the default base/max seq_len params are applied correctly."""
        t = torch.rand(10)
        # Should not raise and return valid output
        result = ModelStrategy._apply_dynamic_shift(
            t, seq_len=1024, shift_base=0.5, shift_max=1.15,
        )
        assert result.shape == t.shape

    def test_monotonic_mapping(self):
        """The shift function should be monotonically increasing."""
        t = torch.linspace(0.01, 0.99, 100)
        result = ModelStrategy._apply_dynamic_shift(
            t, seq_len=1024, shift_base=0.5, shift_max=1.15,
        )
        diffs = result[1:] - result[:-1]
        assert (diffs >= 0).all(), "Shifted t should be monotonically non-decreasing"

    def test_custom_base_max_seq_len(self):
        """Custom base_seq_len and max_seq_len parameters are respected."""
        t = torch.tensor([0.5])
        result_default = ModelStrategy._apply_dynamic_shift(
            t, seq_len=512, shift_base=0.5, shift_max=1.15,
            base_seq_len=256, max_seq_len=4096,
        )
        result_custom = ModelStrategy._apply_dynamic_shift(
            t, seq_len=512, shift_base=0.5, shift_max=1.15,
            base_seq_len=512, max_seq_len=2048,
        )
        # Different base/max seq_len -> different mu -> different result
        # (Unless by coincidence they happen to produce the same mu; unlikely for these values)
        assert not torch.equal(result_default, result_custom)


class TestApplyProgressiveBlend:
    """Test _apply_progressive_blend on the base ModelStrategy class."""

    def test_returns_tensor_same_shape(self):
        t = torch.rand(8)
        result = ModelStrategy._apply_progressive_blend(t, step=500, warmup_steps=1000)
        assert result.shape == t.shape

    def test_no_op_at_or_after_warmup(self):
        """At step >= warmup_steps, the original t should be returned unchanged."""
        t = torch.rand(16)
        # At exactly warmup_steps
        result_at = ModelStrategy._apply_progressive_blend(t, step=1000, warmup_steps=1000)
        assert result_at is t  # same object - no copy

        # After warmup
        result_after = ModelStrategy._apply_progressive_blend(t, step=2000, warmup_steps=1000)
        assert result_after is t

    def test_no_op_when_warmup_steps_zero(self):
        """warmup_steps=0 means no warmup at all - return t unchanged."""
        t = torch.rand(8)
        result = ModelStrategy._apply_progressive_blend(t, step=0, warmup_steps=0)
        assert result is t

    def test_step_zero_returns_purely_uniform(self):
        """At step=0, blend=0.0 so result = 0.0 * t + 1.0 * uniform = uniform.

        The returned tensor should look uniform and NOT be equal to t.
        With large enough batches, a uniformly sampled tensor from rand_like
        will differ significantly from any non-trivial t.
        """
        # Use a deterministic t that is NOT uniform
        t = torch.ones(1000) * 0.9  # all 0.9
        result = ModelStrategy._apply_progressive_blend(t, step=0, warmup_steps=1000)
        # Result should be close to uniform(0,1) mean of 0.5
        assert abs(result.mean().item() - 0.5) < 0.1
        assert not torch.equal(result, t)

    def test_interpolation_between_steps(self):
        """At step=500 with warmup=1000, blend=0.5 so result is midway."""
        # Use a very large batch so statistics are stable
        t = torch.ones(10000) * 0.9
        result = ModelStrategy._apply_progressive_blend(t, step=500, warmup_steps=1000)
        # blend = 0.5: expected = 0.5 * 0.9 + 0.5 * E[uniform] = 0.5 * 0.9 + 0.5 * 0.5 = 0.7
        expected_mean = 0.5 * 0.9 + 0.5 * 0.5
        assert abs(result.mean().item() - expected_mean) < 0.05

    def test_blend_increases_with_step(self):
        """Later steps should produce values closer to the original t distribution."""
        t = torch.ones(1000) * 0.9
        result_early = ModelStrategy._apply_progressive_blend(t, step=100, warmup_steps=1000)
        result_late = ModelStrategy._apply_progressive_blend(t, step=900, warmup_steps=1000)
        # Later step is closer to t=0.9, so mean should be higher
        assert result_late.mean() > result_early.mean()

    def test_result_values_in_range(self):
        """Blended values should stay in [0, 1]."""
        t = torch.rand(100)
        result = ModelStrategy._apply_progressive_blend(t, step=500, warmup_steps=1000)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6


class TestTimestepFeatureConfigs:
    """Test that the new config fields exist and have correct defaults."""

    def test_noise_offset_type_defaults_to_simple(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.noise_offset_type == "simple"

    def test_dynamic_timestep_shift_defaults_false(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.dynamic_timestep_shift is False

    def test_shift_base_default(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.shift_base == 0.5

    def test_shift_max_default(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.shift_max == 1.15

    def test_progressive_timesteps_defaults_false(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.progressive_timesteps is False

    def test_progressive_warmup_steps_default(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.progressive_warmup_steps == 1000

    def test_can_set_noise_offset_type_generalized(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig(noise_offset_type="generalized")
        assert cfg.noise_offset_type == "generalized"

    def test_can_enable_dynamic_shift(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig(dynamic_timestep_shift=True, shift_base=0.3, shift_max=1.5)
        assert cfg.dynamic_timestep_shift is True
        assert cfg.shift_base == 0.3
        assert cfg.shift_max == 1.5

    def test_can_enable_progressive_timesteps(self):
        from trainer.config.schema import TrainingConfig
        cfg = TrainingConfig(progressive_timesteps=True, progressive_warmup_steps=500)
        assert cfg.progressive_timesteps is True
        assert cfg.progressive_warmup_steps == 500
