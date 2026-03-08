"""Tests for WanStrategy - timestep distributions, training step, hook behavior."""
from __future__ import annotations

import math

import pytest
import torch

from trainer.arch.base import ModelStrategy
from trainer.arch.wan.strategy import WanStrategy


class TestSampleT:
    """Test the _sample_t static method on the base ModelStrategy class."""

    def test_uniform_range(self):
        t = ModelStrategy._sample_t(1000, torch.device("cpu"), method="uniform")
        assert t.shape == (1000,)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_sigmoid_distribution(self):
        t = ModelStrategy._sample_t(
            10000, torch.device("cpu"), method="sigmoid", sigmoid_scale=1.0,
        )
        # Sigmoid of standard normal: mean should be ~0.5
        assert 0.4 < t.mean().item() < 0.6

    def test_logit_normal_distribution(self):
        t = ModelStrategy._sample_t(
            10000, torch.device("cpu"), method="logit_normal",
            logit_mean=0.0, logit_std=1.0,
        )
        # Similar to sigmoid, mean should be ~0.5
        assert 0.4 < t.mean().item() < 0.6

    def test_shift_distribution(self):
        t = ModelStrategy._sample_t(
            10000, torch.device("cpu"), method="shift",
            sigmoid_scale=1.0, flow_shift=math.exp(1.0),
        )
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_min_max_clamping(self):
        t = ModelStrategy._sample_t(
            1000, torch.device("cpu"), method="uniform",
            min_t=0.2, max_t=0.8,
        )
        assert t.min() >= 0.2 - 1e-6
        assert t.max() <= 0.8 + 1e-6

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown timestep"):
            ModelStrategy._sample_t(1, torch.device("cpu"), method="invalid")


class TestWanTimestepScaling:
    """Test that WanStrategy._sample_timesteps applies the correct scaling."""

    def _make_strategy(self):
        from trainer.config.schema import TrainConfig
        config = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake/path"},
            training={"method": "full_finetune"},
            data={"datasets": [{"path": "/fake/data"}]},
        )
        s = WanStrategy(config)
        # Set the cached config values that setup() would set
        s._ts_method = "uniform"
        s._ts_min = 0.0
        s._ts_max = 1.0
        s._ts_sigmoid_scale = 1.0
        s._ts_logit_mean = 0.0
        s._ts_logit_std = 1.0
        s._flow_shift = 1.0
        return s

    def test_timestep_scale(self):
        """timesteps = t * 1000 + 1, so t=0 -> ts=1, t=1 -> ts=1001."""
        s = self._make_strategy()
        t, ts = s._sample_timesteps(1, torch.device("cpu"))
        expected = t * 1000.0 + 1.0
        assert torch.allclose(ts, expected)

    def test_timestep_range(self):
        s = self._make_strategy()
        t, ts = s._sample_timesteps(1000, torch.device("cpu"))
        assert t.shape == (1000,)
        assert t.min() >= 0.0
        assert t.max() <= 1.0
        # timesteps in [1, 1001]
        assert ts.min() >= 1.0
        assert ts.max() <= 1001.0


class TestApplyNoiseOffset:
    """Test _apply_noise_offset on the base class."""

    def test_no_op_when_zero(self):
        noise = torch.randn(2, 4, 8, 8)
        original = noise.clone()
        ModelStrategy._apply_noise_offset(noise, 0.0)
        assert torch.equal(noise, original)

    def test_no_op_when_negative(self):
        noise = torch.randn(2, 4, 8, 8)
        original = noise.clone()
        ModelStrategy._apply_noise_offset(noise, -0.1)
        assert torch.equal(noise, original)

    def test_modifies_4d_noise(self):
        noise = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise, 0.5)
        # After offset, noise should no longer be all zeros
        assert not torch.equal(noise, torch.zeros_like(noise))
        # Channel-wise offset: all spatial positions in a channel should get same offset
        # Check that noise[:, c, :, :] has constant offset per (batch, channel)
        for b in range(2):
            for c in range(4):
                vals = noise[b, c]
                assert torch.allclose(vals, vals[0, 0].expand_as(vals))

    def test_modifies_5d_noise(self):
        noise = torch.zeros(2, 4, 3, 8, 8)
        ModelStrategy._apply_noise_offset(noise, 0.5)
        assert not torch.equal(noise, torch.zeros_like(noise))
        # Channel-wise offset: all spatio-temporal positions per (batch, channel) same
        for b in range(2):
            for c in range(4):
                vals = noise[b, c]
                assert torch.allclose(vals, vals[0, 0, 0].expand_as(vals))


class TestApplyNoiseOffsetGeneralized:
    """Test _apply_noise_offset with generalized mode."""

    def test_simple_mode_unchanged_behavior(self):
        """Simple mode with t=None should behave identically to the old API."""
        torch.manual_seed(0)
        noise1 = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise1, 0.5)

        torch.manual_seed(0)
        noise2 = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise2, 0.5, offset_type="simple")

        assert torch.equal(noise1, noise2)

    def test_generalized_no_op_when_offset_zero(self):
        t = torch.tensor([0.5, 0.8])
        noise = torch.randn(2, 4, 8, 8)
        original = noise.clone()
        ModelStrategy._apply_noise_offset(noise, 0.0, t=t, offset_type="generalized")
        assert torch.equal(noise, original)

    def test_generalized_no_op_when_offset_negative(self):
        t = torch.tensor([0.5, 0.8])
        noise = torch.randn(2, 4, 8, 8)
        original = noise.clone()
        ModelStrategy._apply_noise_offset(noise, -0.1, t=t, offset_type="generalized")
        assert torch.equal(noise, original)

    def test_generalized_modifies_noise(self):
        """Generalized mode should modify the noise tensor."""
        t = torch.tensor([0.5, 0.9])
        noise = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise, 0.5, t=t, offset_type="generalized")
        assert not torch.equal(noise, torch.zeros_like(noise))

    def test_generalized_zero_t_means_zero_offset(self):
        """When t=0 for all samples, psi(t) = offset * sqrt(0) = 0 - no modification."""
        t = torch.zeros(2)
        noise = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise, 1.0, t=t, offset_type="generalized")
        # psi(0) = 0 so channel_offset * 0 = 0; noise stays at zero
        assert torch.equal(noise, torch.zeros_like(noise))

    def test_generalized_fallback_to_simple_when_t_is_none(self):
        """offset_type='generalized' with t=None falls back to simple behavior."""
        torch.manual_seed(42)
        noise1 = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise1, 0.5, t=None, offset_type="generalized")

        torch.manual_seed(42)
        noise2 = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise2, 0.5)

        assert torch.equal(noise1, noise2)

    def test_generalized_channel_wise_offset_4d(self):
        """Each (batch, channel) pair gets a uniform spatial offset in generalized mode."""
        t = torch.tensor([0.5, 0.5])
        noise = torch.zeros(2, 4, 8, 8)
        ModelStrategy._apply_noise_offset(noise, 1.0, t=t, offset_type="generalized")
        # All spatial positions in each (batch, channel) block should be equal
        for b in range(2):
            for c in range(4):
                vals = noise[b, c]
                assert torch.allclose(vals, vals[0, 0].expand_as(vals))

    def test_generalized_higher_t_gives_larger_offset(self):
        """Generalized mode: larger t -> larger expected offset magnitude."""
        # Use a large batch and check that mean abs offset is larger for t=0.9 vs t=0.1
        t_low = torch.full((200,), 0.1)
        t_high = torch.full((200,), 0.9)

        noise_low = torch.zeros(200, 4, 4, 4)
        ModelStrategy._apply_noise_offset(noise_low, 1.0, t=t_low, offset_type="generalized")

        noise_high = torch.zeros(200, 4, 4, 4)
        ModelStrategy._apply_noise_offset(noise_high, 1.0, t=t_high, offset_type="generalized")

        mean_abs_low = noise_low.abs().mean().item()
        mean_abs_high = noise_high.abs().mean().item()
        # psi(0.9) = sqrt(0.9) ~ 0.949, psi(0.1) = sqrt(0.1) ~ 0.316
        # With 200 samples per group, the means should reflect this ratio
        assert mean_abs_high > mean_abs_low


class TestStrategyProperties:
    """Test basic WanStrategy properties without loading a model."""

    def _make_strategy(self):
        """Create a WanStrategy with minimal config (setup not called)."""
        from trainer.config.schema import TrainConfig
        config = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake/path"},
            training={"method": "full_finetune"},
            data={"datasets": [{"path": "/fake/data"}]},
        )
        return WanStrategy(config)

    def test_architecture_name(self):
        s = self._make_strategy()
        assert s.architecture == "wan"

    def test_supports_video(self):
        s = self._make_strategy()
        assert s.supports_video is True

    def test_default_hooks_return_values(self):
        """Before setup() is called, hooks should work with defaults."""
        from trainer.arch.base import ModelComponents
        from unittest.mock import MagicMock

        s = self._make_strategy()
        # Set the internal state that setup() would set
        s._blocks_to_swap = 0
        s._device = torch.device("cpu")

        components = MagicMock(spec=ModelComponents)
        accelerator = MagicMock()

        hints = s.on_before_accelerate_prepare(components, accelerator)
        assert hints == {"device_placement": True}

    def test_block_swap_disables_device_placement(self):
        from trainer.arch.base import ModelComponents
        from unittest.mock import MagicMock

        s = self._make_strategy()
        s._blocks_to_swap = 10
        s._device = torch.device("cpu")

        components = MagicMock(spec=ModelComponents)
        accelerator = MagicMock()

        hints = s.on_before_accelerate_prepare(components, accelerator)
        assert hints == {"device_placement": False}
