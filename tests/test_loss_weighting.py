"""Tests for trainer/loss_weighting.py."""
from __future__ import annotations

import math

import pytest
import torch

from trainer.loss_weighting import (
    compute_snr_ddpm,
    compute_snr_flow_matching,
    debiased_estimation_weights,
    get_weight_fn,
    min_snr_gamma_weights,
    p2_weights,
)


# ---------------------------------------------------------------------------
# compute_snr_flow_matching
# ---------------------------------------------------------------------------

class TestComputeSnrFlowMatching:
    def test_midpoint_snr_equals_one(self):
        """At t=0.5, SNR = (0.5/0.5)^2 = 1."""
        t = torch.tensor([0.5])
        snr = compute_snr_flow_matching(t)
        assert torch.isclose(snr, torch.tensor([1.0]), atol=1e-5)

    def test_small_t_large_snr(self):
        """Near t=0, (1-t)/t is large, so SNR should be large."""
        t = torch.tensor([0.001])
        snr = compute_snr_flow_matching(t)
        # Approx (0.999/0.001)^2 ~ 998001
        assert snr.item() > 1e5

    def test_large_t_small_snr(self):
        """Near t=1, (1-t)/t is small, so SNR should be near 0."""
        t = torch.tensor([0.999])
        snr = compute_snr_flow_matching(t)
        # Approx (0.001/0.999)^2 ~ 1e-6
        assert snr.item() < 1e-3

    def test_exact_t0_clamped_no_inf(self):
        """t=0.0 must be clamped; result must be finite."""
        t = torch.tensor([0.0])
        snr = compute_snr_flow_matching(t)
        assert torch.isfinite(snr).all()

    def test_exact_t1_clamped_no_zero_division(self):
        """t=1.0 must be clamped; result must be finite and non-negative."""
        t = torch.tensor([1.0])
        snr = compute_snr_flow_matching(t)
        assert torch.isfinite(snr).all()
        assert (snr >= 0).all()

    def test_batch_shape_preserved(self):
        """Output shape matches input shape."""
        t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        snr = compute_snr_flow_matching(t)
        assert snr.shape == t.shape

    def test_monotonically_decreasing(self):
        """SNR should decrease as t increases (more noise = lower SNR)."""
        t = torch.linspace(0.1, 0.9, 9)
        snr = compute_snr_flow_matching(t)
        diffs = snr[1:] - snr[:-1]
        assert (diffs < 0).all()

    def test_formula_correctness(self):
        """Verify formula ((1-t)/t)^2 directly for a few known values."""
        t_vals = [0.25, 0.5, 0.75]
        for tv in t_vals:
            t = torch.tensor([tv])
            snr = compute_snr_flow_matching(t)
            expected = ((1.0 - tv) / tv) ** 2
            assert abs(snr.item() - expected) < 1e-5, f"Mismatch at t={tv}"

    def test_all_values_non_negative(self):
        """SNR is always non-negative."""
        t = torch.rand(32).clamp(0.01, 0.99)
        snr = compute_snr_flow_matching(t)
        assert (snr >= 0).all()


# ---------------------------------------------------------------------------
# compute_snr_ddpm
# ---------------------------------------------------------------------------

class TestComputeSnrDdpm:
    def _make_alphas_cumprod(self, T: int = 1000) -> torch.Tensor:
        """Linearly spaced alpha_bar from ~1 to ~0 (simple schedule)."""
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        return torch.cumprod(alphas, dim=0)

    def test_basic_formula(self):
        """SNR = alpha_bar / (1 - alpha_bar)."""
        alphas_cumprod = torch.tensor([0.8, 0.5, 0.2])
        timesteps = torch.tensor([0, 1, 2])
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        expected = torch.tensor([0.8 / 0.2, 0.5 / 0.5, 0.2 / 0.8])
        assert torch.allclose(snr, expected, atol=1e-5)

    def test_high_alpha_bar_high_snr(self):
        """alpha_bar near 1 gives very high SNR."""
        alphas_cumprod = torch.tensor([0.9999])
        timesteps = torch.tensor([0])
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        assert snr.item() > 1000

    def test_low_alpha_bar_low_snr(self):
        """alpha_bar near 0 gives very low SNR."""
        alphas_cumprod = torch.tensor([0.001])
        timesteps = torch.tensor([0])
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        assert snr.item() < 0.01

    def test_equal_alpha_bar_snr_is_one(self):
        """alpha_bar = 0.5 gives SNR = 1."""
        alphas_cumprod = torch.tensor([0.5])
        timesteps = torch.tensor([0])
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        assert torch.isclose(snr, torch.tensor([1.0]), atol=1e-5)

    def test_batch_indexing(self):
        """Each element in timesteps indexes independently into alphas_cumprod."""
        alphas_cumprod = torch.tensor([0.9, 0.6, 0.3])
        timesteps = torch.tensor([2, 0, 1])
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        expected = torch.tensor([
            0.3 / 0.7,
            0.9 / 0.1,
            0.6 / 0.4,
        ])
        assert torch.allclose(snr, expected, atol=1e-5)

    def test_output_shape(self):
        """Output shape matches timesteps shape."""
        alphas_cumprod = self._make_alphas_cumprod(1000)
        timesteps = torch.randint(0, 1000, (8,))
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        assert snr.shape == timesteps.shape

    def test_all_positive(self):
        """All SNR values should be positive."""
        alphas_cumprod = self._make_alphas_cumprod(1000)
        timesteps = torch.randint(0, 1000, (16,))
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        assert (snr > 0).all()

    def test_no_nan_or_inf(self):
        """No NaN or Inf in output."""
        alphas_cumprod = self._make_alphas_cumprod(1000)
        timesteps = torch.arange(0, 1000)
        snr = compute_snr_ddpm(alphas_cumprod, timesteps)
        assert torch.isfinite(snr).all()


# ---------------------------------------------------------------------------
# min_snr_gamma_weights
# ---------------------------------------------------------------------------

class TestMinSnrGammaWeights:
    def test_high_snr_clamped_to_gamma(self):
        """When SNR >> gamma, weight = gamma / SNR (< 1)."""
        snr = torch.tensor([100.0])
        gamma = 5.0
        w = min_snr_gamma_weights(snr, gamma)
        expected = gamma / 100.0
        assert torch.isclose(w, torch.tensor([expected]), atol=1e-6)

    def test_low_snr_weight_equals_one(self):
        """When SNR < gamma, min(SNR, gamma) = SNR, so weight = 1."""
        snr = torch.tensor([2.0])
        gamma = 5.0
        w = min_snr_gamma_weights(snr, gamma)
        assert torch.isclose(w, torch.tensor([1.0]), atol=1e-6)

    def test_snr_equals_gamma_weight_is_one(self):
        """When SNR == gamma, weight = gamma / gamma = 1."""
        gamma = 5.0
        snr = torch.tensor([gamma])
        w = min_snr_gamma_weights(snr, gamma)
        assert torch.isclose(w, torch.tensor([1.0]), atol=1e-6)

    def test_weights_bounded_by_one(self):
        """All weights should be <= 1."""
        snr = torch.logspace(-2, 4, 50)
        w = min_snr_gamma_weights(snr, gamma=5.0)
        assert (w <= 1.0 + 1e-6).all()

    def test_weights_positive(self):
        """All weights should be > 0."""
        snr = torch.logspace(-2, 4, 50)
        w = min_snr_gamma_weights(snr, gamma=5.0)
        assert (w > 0).all()

    def test_batch_shape_preserved(self):
        """Output shape matches input shape."""
        snr = torch.rand(16) * 10
        w = min_snr_gamma_weights(snr)
        assert w.shape == snr.shape

    def test_different_gamma_values(self):
        """Larger gamma allows more high-SNR timesteps through at full weight."""
        snr = torch.tensor([8.0])
        w_small_gamma = min_snr_gamma_weights(snr, gamma=5.0)
        w_large_gamma = min_snr_gamma_weights(snr, gamma=10.0)
        # With gamma=10, SNR=8 < gamma, so weight=1; with gamma=5, weight<1
        assert torch.isclose(w_large_gamma, torch.tensor([1.0]), atol=1e-6)
        assert w_small_gamma.item() < 1.0

    def test_default_gamma_is_5(self):
        """Default gamma=5.0 matches explicit call."""
        snr = torch.tensor([20.0, 3.0, 5.0])
        assert torch.allclose(
            min_snr_gamma_weights(snr),
            min_snr_gamma_weights(snr, gamma=5.0),
        )


# ---------------------------------------------------------------------------
# debiased_estimation_weights
# ---------------------------------------------------------------------------

class TestDebiasedEstimationWeights:
    def test_formula_snr_one(self):
        """At SNR=1, weight = 1/sqrt(1) = 1."""
        snr = torch.tensor([1.0])
        w = debiased_estimation_weights(snr)
        assert torch.isclose(w, torch.tensor([1.0]), atol=1e-6)

    def test_formula_snr_four(self):
        """At SNR=4, weight = 1/sqrt(4) = 0.5."""
        snr = torch.tensor([4.0])
        w = debiased_estimation_weights(snr)
        assert torch.isclose(w, torch.tensor([0.5]), atol=1e-6)

    def test_formula_snr_point25(self):
        """At SNR=0.25, weight = 1/sqrt(0.25) = 2."""
        snr = torch.tensor([0.25])
        w = debiased_estimation_weights(snr)
        assert torch.isclose(w, torch.tensor([2.0]), atol=1e-5)

    def test_high_snr_low_weight(self):
        """Higher SNR gives lower weight (debiases toward high-noise steps)."""
        snr_high = torch.tensor([100.0])
        snr_low = torch.tensor([1.0])
        assert debiased_estimation_weights(snr_high) < debiased_estimation_weights(snr_low)

    def test_near_zero_snr_no_inf(self):
        """SNR near 0 is clamped; result must be finite."""
        snr = torch.tensor([0.0])
        w = debiased_estimation_weights(snr)
        assert torch.isfinite(w).all()

    def test_batch_shape_preserved(self):
        """Output shape matches input shape."""
        snr = torch.rand(16) + 0.1
        w = debiased_estimation_weights(snr)
        assert w.shape == snr.shape

    def test_all_positive(self):
        """All weights should be positive."""
        snr = torch.logspace(-1, 3, 40)
        w = debiased_estimation_weights(snr)
        assert (w > 0).all()

    def test_formula_matches_manual(self):
        """Verify 1/sqrt(SNR) formula across a batch."""
        snr = torch.tensor([0.5, 1.0, 2.0, 9.0, 16.0])
        w = debiased_estimation_weights(snr)
        expected = 1.0 / snr.sqrt()
        assert torch.allclose(w, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# get_weight_fn
# ---------------------------------------------------------------------------

class TestGetWeightFn:
    def test_none_scheme_returns_none(self):
        """'none' scheme returns None (no weighting)."""
        fn = get_weight_fn("none")
        assert fn is None

    def test_min_snr_gamma_returns_callable(self):
        """'min_snr_gamma' returns a callable."""
        fn = get_weight_fn("min_snr_gamma")
        assert callable(fn)

    def test_debiased_returns_callable(self):
        """'debiased' returns a callable."""
        fn = get_weight_fn("debiased")
        assert callable(fn)

    def test_invalid_scheme_raises_value_error(self):
        """Unknown scheme raises ValueError with informative message."""
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            get_weight_fn("unknown_scheme")

    def test_min_snr_gamma_fn_applies_correctly(self):
        """Returned min_snr_gamma function applies gamma from closure."""
        gamma = 3.0
        fn = get_weight_fn("min_snr_gamma", snr_gamma=gamma)
        snr = torch.tensor([10.0])  # SNR > gamma, so weight = gamma / SNR
        w = fn(snr)
        expected = gamma / 10.0
        assert torch.isclose(w, torch.tensor([expected]), atol=1e-6)

    def test_min_snr_gamma_default_gamma(self):
        """Default gamma=5.0 is used when not specified."""
        fn = get_weight_fn("min_snr_gamma")
        snr = torch.tensor([10.0])
        w = fn(snr)
        expected = 5.0 / 10.0
        assert torch.isclose(w, torch.tensor([expected]), atol=1e-6)

    def test_debiased_fn_matches_direct_call(self):
        """Returned debiased function matches direct debiased_estimation_weights call."""
        fn = get_weight_fn("debiased")
        snr = torch.tensor([4.0, 1.0, 0.25])
        assert torch.allclose(fn(snr), debiased_estimation_weights(snr))

    def test_min_snr_gamma_fn_matches_direct_call(self):
        """Returned min_snr_gamma function matches direct min_snr_gamma_weights call."""
        gamma = 7.0
        fn = get_weight_fn("min_snr_gamma", snr_gamma=gamma)
        snr = torch.tensor([2.0, 7.0, 50.0])
        assert torch.allclose(fn(snr), min_snr_gamma_weights(snr, gamma=gamma))

    def test_different_gamma_values_produce_different_results(self):
        """Different snr_gamma values produce different weight functions."""
        fn5 = get_weight_fn("min_snr_gamma", snr_gamma=5.0)
        fn10 = get_weight_fn("min_snr_gamma", snr_gamma=10.0)
        snr = torch.tensor([7.0])
        assert not torch.isclose(fn5(snr), fn10(snr))

    def test_error_message_includes_supported_schemes(self):
        """Error message for unknown scheme lists supported options."""
        with pytest.raises(ValueError, match="min_snr_gamma"):
            get_weight_fn("bad_scheme")

    def test_empty_string_raises(self):
        """Empty string is not a valid scheme."""
        with pytest.raises(ValueError):
            get_weight_fn("")

    def test_case_sensitive(self):
        """Scheme matching is case-sensitive."""
        with pytest.raises(ValueError):
            get_weight_fn("Min_SNR_Gamma")

    def test_p2_scheme_returns_callable(self):
        """'p2' scheme returns a callable."""
        fn = get_weight_fn("p2")
        assert callable(fn)

    def test_p2_fn_applies_gamma_from_closure(self):
        """Returned p2 function uses p2_gamma from closure."""
        gamma = 2.0
        fn = get_weight_fn("p2", p2_gamma=gamma)
        snr = torch.tensor([3.0])
        w = fn(snr)
        expected = 1.0 / (1.0 + 3.0) ** 2
        assert torch.isclose(w, torch.tensor([expected]), atol=1e-6)

    def test_p2_fn_default_gamma(self):
        """Default p2_gamma=1.0 is used when not specified."""
        fn = get_weight_fn("p2")
        snr = torch.tensor([4.0])
        w = fn(snr)
        expected = p2_weights(snr, gamma=1.0)
        assert torch.allclose(w, expected)

    def test_error_message_includes_p2(self):
        """Error message for unknown scheme mentions 'p2'."""
        with pytest.raises(ValueError, match="p2"):
            get_weight_fn("unknown_scheme")


# ---------------------------------------------------------------------------
# p2_weights
# ---------------------------------------------------------------------------

class TestP2Weights:
    def test_formula_snr_zero_gamma_one(self):
        """At SNR=0, weight = 1/(1+0)^1 = 1."""
        snr = torch.tensor([0.0])
        w = p2_weights(snr, gamma=1.0)
        assert torch.isclose(w, torch.tensor([1.0]), atol=1e-6)

    def test_formula_snr_one_gamma_one(self):
        """At SNR=1, weight = 1/(1+1)^1 = 0.5."""
        snr = torch.tensor([1.0])
        w = p2_weights(snr, gamma=1.0)
        assert torch.isclose(w, torch.tensor([0.5]), atol=1e-6)

    def test_formula_snr_one_gamma_two(self):
        """At SNR=1, gamma=2: weight = 1/(1+1)^2 = 0.25."""
        snr = torch.tensor([1.0])
        w = p2_weights(snr, gamma=2.0)
        assert torch.isclose(w, torch.tensor([0.25]), atol=1e-6)

    def test_gamma_zero_equals_uniform(self):
        """gamma=0 gives weight = 1/(1+snr)^0 = 1 for all SNR."""
        snr = torch.logspace(-2, 3, 20)
        w = p2_weights(snr, gamma=0.0)
        assert torch.allclose(w, torch.ones_like(snr), atol=1e-6)

    def test_high_snr_lower_weight(self):
        """Higher SNR yields lower weight (downweights easy timesteps)."""
        snr_low = torch.tensor([1.0])
        snr_high = torch.tensor([100.0])
        assert p2_weights(snr_high) < p2_weights(snr_low)

    def test_higher_gamma_stronger_downweighting(self):
        """Larger gamma means even lower weight at the same high SNR."""
        snr = torch.tensor([10.0])
        w_gamma1 = p2_weights(snr, gamma=1.0)
        w_gamma2 = p2_weights(snr, gamma=2.0)
        assert w_gamma2 < w_gamma1

    def test_all_positive(self):
        """All weights are strictly positive."""
        snr = torch.logspace(-2, 4, 50)
        w = p2_weights(snr)
        assert (w > 0).all()

    def test_no_nan_or_inf(self):
        """No NaN or Inf in output, including at SNR=0."""
        snr = torch.cat([torch.tensor([0.0]), torch.logspace(-2, 4, 50)])
        w = p2_weights(snr)
        assert torch.isfinite(w).all()

    def test_batch_shape_preserved(self):
        """Output shape matches input shape."""
        snr = torch.rand(16) * 10
        w = p2_weights(snr)
        assert w.shape == snr.shape

    def test_custom_k(self):
        """Custom k shifts the denominator offset."""
        snr = torch.tensor([1.0])
        k = 2.0
        w = p2_weights(snr, gamma=1.0, k=k)
        expected = 1.0 / (k + 1.0)
        assert torch.isclose(w, torch.tensor([expected]), atol=1e-6)

    def test_formula_matches_manual_batch(self):
        """Verify 1/(1+snr)^gamma across a batch."""
        snr = torch.tensor([0.5, 1.0, 4.0, 9.0, 25.0])
        gamma = 1.5
        w = p2_weights(snr, gamma=gamma)
        expected = 1.0 / (1.0 + snr).pow(gamma)
        assert torch.allclose(w, expected, atol=1e-6)

    def test_monotonically_decreasing_with_snr(self):
        """Weights should decrease monotonically as SNR increases."""
        snr = torch.linspace(0.1, 50.0, 20)
        w = p2_weights(snr)
        diffs = w[1:] - w[:-1]
        assert (diffs < 0).all()
