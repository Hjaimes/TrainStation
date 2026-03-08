"""Tests for zero-terminal-SNR noise schedule rescaling.

Covers:
- ModelStrategy._rescale_zero_terminal_snr() directly
- TrainingConfig.zero_terminal_snr field round-trip
- TrainingConfig.p2_gamma field round-trip
"""
from __future__ import annotations

import pytest
import torch

from trainer.arch.base import ModelStrategy
from trainer.config.schema import TrainConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _sdxl_alphas_cumprod(T: int = 1000) -> torch.Tensor:
    """SDXL scaled-linear beta schedule (matches diffusers / compute_alphas_cumprod).

    beta_start=0.00085, beta_end=0.012, quadratic spacing.
    Terminal value: alpha_bar[-1] ≈ 0.0047 - clearly non-zero, so the
    rescaling has a measurable effect and the post-rescaled[0] ≈ original[0].
    """
    beta_start = 0.00085
    beta_end = 0.012
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def _short_nonzero_schedule(T: int = 20) -> torch.Tensor:
    """Short schedule with terminal alpha_bar well above zero, for clean numeric tests."""
    betas = torch.linspace(0.001, 0.05, T)  # mild betas → terminal > 0.1
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


# ---------------------------------------------------------------------------
# _rescale_zero_terminal_snr
# ---------------------------------------------------------------------------

class TestRescaleZeroTerminalSnr:
    def test_final_entry_is_exactly_zero(self):
        """After rescaling, alphas_cumprod[-1] is exactly 0.0 (clamped)."""
        alphas = _sdxl_alphas_cumprod()
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert rescaled[-1].item() == pytest.approx(0.0, abs=1e-7)

    def test_rescaling_changes_terminal(self):
        """The SDXL schedule's non-zero terminal is made zero by rescaling."""
        alphas = _sdxl_alphas_cumprod()
        # Confirm the original schedule has a non-trivially non-zero terminal
        assert alphas[-1].item() > 1e-3
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert rescaled[-1].item() == pytest.approx(0.0, abs=1e-7)

    def test_first_entry_less_than_original(self):
        """Rescaling shifts sqrt_one_minus down to zero at the terminal, which
        scales sqrt_alpha by a factor < 1.  The rescaled first alpha_bar is
        therefore strictly less than the original first alpha_bar.
        """
        alphas = _sdxl_alphas_cumprod()
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        # The first entry is compressed but remains positive
        assert rescaled[0].item() > 0.0
        assert rescaled[0].item() < alphas[0].item()

    def test_output_shape_unchanged(self):
        """Rescaled tensor has the same shape as input."""
        alphas = _sdxl_alphas_cumprod(500)
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert rescaled.shape == alphas.shape

    def test_values_non_negative(self):
        """All rescaled alpha_bar values must be >= 0."""
        alphas = _sdxl_alphas_cumprod()
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert (rescaled >= 0.0).all()

    def test_values_at_most_one(self):
        """All rescaled alpha_bar values must be <= 1."""
        alphas = _sdxl_alphas_cumprod()
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert (rescaled <= 1.0 + 1e-6).all()

    def test_no_nan_or_inf(self):
        """No NaN or Inf in output."""
        alphas = _sdxl_alphas_cumprod()
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert torch.isfinite(rescaled).all()

    def test_monotonically_non_increasing(self):
        """Rescaled schedule must remain monotonically non-increasing."""
        alphas = _sdxl_alphas_cumprod()
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        diffs = rescaled[1:] - rescaled[:-1]
        assert (diffs <= 1e-6).all(), "Rescaled schedule is not monotonically non-increasing"

    def test_ratio_structure_preserved(self):
        """Relative spacing between intermediate timesteps is preserved.

        The rescaling is a global affine shift on sqrt_one_minus, so the
        ratio alpha_bar[i] / alpha_bar[j] should be approximately preserved
        for timesteps far from the terminal.
        """
        alphas = _sdxl_alphas_cumprod()
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        # Pick two mid-range timesteps
        ratio_orig = alphas[200].item() / alphas[400].item()
        ratio_rescaled = rescaled[200].item() / rescaled[400].item()
        assert abs(ratio_orig - ratio_rescaled) < 0.05

    def test_idempotent_safe(self):
        """Applying rescaling twice is safe - the final entry remains 0."""
        alphas = _sdxl_alphas_cumprod()
        once = ModelStrategy._rescale_zero_terminal_snr(alphas)
        twice = ModelStrategy._rescale_zero_terminal_snr(once)
        assert twice[-1].item() == pytest.approx(0.0, abs=1e-6)
        assert torch.isfinite(twice).all()

    def test_short_schedule_terminal_zero(self):
        """Works for short schedules with clearly non-zero terminal."""
        alphas = _short_nonzero_schedule(T=20)
        # Confirm non-zero terminal before rescaling
        assert alphas[-1].item() > 0.1
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert rescaled[-1].item() == pytest.approx(0.0, abs=1e-6)
        assert (rescaled >= 0.0).all()
        assert torch.isfinite(rescaled).all()

    def test_different_lengths(self):
        """Works for various schedule lengths (not just 1000)."""
        for T in [100, 500, 1000]:
            alphas = _sdxl_alphas_cumprod(T)
            rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
            assert rescaled[-1].item() == pytest.approx(0.0, abs=1e-6)
            assert rescaled.shape == (T,)

    def test_dtype_preserved_float32(self):
        """Output dtype matches float32 input."""
        alphas = _sdxl_alphas_cumprod().to(torch.float32)
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert rescaled.dtype == torch.float32

    def test_dtype_preserved_float64(self):
        """Output dtype matches float64 input."""
        alphas = _sdxl_alphas_cumprod().to(torch.float64)
        rescaled = ModelStrategy._rescale_zero_terminal_snr(alphas)
        assert rescaled.dtype == torch.float64


# ---------------------------------------------------------------------------
# TrainingConfig schema - zero_terminal_snr field
# ---------------------------------------------------------------------------

def _make_config(**training_kwargs) -> TrainConfig:
    """Build a minimal valid TrainConfig (full_finetune to avoid network requirement)."""
    return TrainConfig(
        model={"architecture": "sdxl", "base_model_path": "/fake/path"},
        training=TrainingConfig(method="full_finetune", **training_kwargs),
        data={"datasets": [{"path": "/fake/data"}]},
    )


class TestZeroTerminalSnrConfig:
    def test_default_is_false(self):
        """zero_terminal_snr defaults to False."""
        cfg = _make_config()
        assert cfg.training.zero_terminal_snr is False

    def test_can_enable(self):
        """zero_terminal_snr can be set to True."""
        cfg = _make_config(zero_terminal_snr=True)
        assert cfg.training.zero_terminal_snr is True

    def test_round_trip_true(self):
        """True value survives dict serialization round-trip."""
        cfg = _make_config(zero_terminal_snr=True)
        data = cfg.to_dict()
        cfg2 = TrainConfig.from_dict(data)
        assert cfg2.training.zero_terminal_snr is True

    def test_round_trip_false(self):
        """Default False survives dict serialization round-trip."""
        cfg = _make_config(zero_terminal_snr=False)
        data = cfg.to_dict()
        cfg2 = TrainConfig.from_dict(data)
        assert cfg2.training.zero_terminal_snr is False


# ---------------------------------------------------------------------------
# P2 gamma config field
# ---------------------------------------------------------------------------

class TestP2GammaConfig:
    def test_default_p2_gamma(self):
        """p2_gamma defaults to 1.0."""
        cfg = _make_config()
        assert cfg.training.p2_gamma == pytest.approx(1.0)

    def test_can_set_p2_gamma(self):
        """p2_gamma can be set to a custom value."""
        cfg = _make_config(p2_gamma=2.5)
        assert cfg.training.p2_gamma == pytest.approx(2.5)

    def test_p2_gamma_round_trip(self):
        """p2_gamma survives dict serialization round-trip."""
        cfg = _make_config(p2_gamma=0.7, weighting_scheme="p2")
        data = cfg.to_dict()
        cfg2 = TrainConfig.from_dict(data)
        assert cfg2.training.p2_gamma == pytest.approx(0.7)
        assert cfg2.training.weighting_scheme == "p2"

    def test_p2_gamma_zero_is_valid(self):
        """p2_gamma=0 is valid (produces uniform weighting)."""
        cfg = _make_config(p2_gamma=0.0, weighting_scheme="p2")
        assert cfg.training.p2_gamma == pytest.approx(0.0)
