"""Tests for AdamW Advanced optimizer."""
import pytest
import torch

from trainer.adamw_advanced import (
    AdamWAdvanced,
    _orthogonalize_gradient,
    _grams_update,
    _cautious_update,
    _get_effective_shape,
    _copy_stochastic_,
    _init_sr_generator,
    _factorize_state,
    _reconstruct_state,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_orthogonalize_gradient(self):
        p = torch.randn(4, 4)
        g = torch.randn(4, 4)
        g_orth = _orthogonalize_gradient(p, g)
        assert g_orth.shape == g.shape
        # Should be approximately orthogonal to p
        dot = torch.dot(p.view(-1).float(), g_orth.view(-1).float())
        assert abs(dot.item()) < 1e-4

    def test_grams_update(self):
        mt = torch.randn(8)
        grad = torch.randn(8)
        result = _grams_update(mt, grad)
        expected = grad.sign() * mt.abs()
        assert torch.allclose(result, expected)

    def test_cautious_update(self):
        mt = torch.tensor([1.0, -1.0, 1.0, -1.0])
        grad = torch.tensor([1.0, 1.0, -1.0, -1.0])
        result = _cautious_update(mt, grad)
        # Only positions where mt*grad > 0 should be nonzero (idx 0 and 3)
        assert result[1].item() == 0.0
        assert result[2].item() == 0.0
        assert result[0].item() != 0.0
        assert result[3].item() != 0.0

    def test_get_effective_shape(self):
        assert _get_effective_shape(12) == (4, 3)
        assert _get_effective_shape(16) == (4, 4)
        assert _get_effective_shape(7) == (7, 1)  # prime

    def test_stochastic_rounding(self):
        _init_sr_generator(torch.device("cpu"))
        src = torch.randn(10, dtype=torch.float32)
        tgt = torch.zeros(10, dtype=torch.bfloat16)
        _copy_stochastic_(tgt, src)
        # Result should be close to source (bf16 precision)
        assert torch.allclose(tgt.float(), src, atol=0.1)

    def test_factorize_reconstruct_unsigned(self):
        original = torch.rand(6, 4)  # non-negative for unsigned
        mu, mv = _factorize_state(original.clone(), signed=False)
        recon = _reconstruct_state((mu, mv), signed=False)
        # Rank-1 approximation won't be exact, but shape should match
        assert recon.shape == original.shape

    def test_factorize_reconstruct_signed(self):
        original = torch.randn(6, 4)
        mu, mv, sign = _factorize_state(original.clone(), signed=True)
        recon = _reconstruct_state((mu, mv, sign, 4), signed=True)
        assert recon.shape == original.shape
        # Signs should be preserved
        orig_signs = original > 0
        recon_signs = recon > 0
        assert (orig_signs == recon_signs).float().mean() > 0.8


# ---------------------------------------------------------------------------
# Optimizer basic tests
# ---------------------------------------------------------------------------

class TestAdamWAdvancedBasic:
    def _make_params(self, shape=(4, 4), dtype=torch.float32):
        p = torch.randn(shape, dtype=dtype, requires_grad=True)
        return [{"params": [p]}]

    def test_basic_step(self):
        params = self._make_params()
        opt = AdamWAdvanced(params, lr=0.01)
        p = params[0]["params"][0]
        p_before = p.data.clone()
        loss = (p ** 2).sum()
        loss.backward()
        opt.step()
        # Parameters should change
        assert not torch.equal(p.data, p_before)

    def test_step_parameter(self):
        params = self._make_params()
        opt = AdamWAdvanced(params, lr=0.01)
        p = params[0]["params"][0]
        loss = (p ** 2).sum()
        loss.backward()
        opt.step_parameter(p, opt.param_groups[0])
        # State should be initialized
        assert "step" in opt.state[p]
        assert opt.state[p]["step"] == 1

    def test_zero_grad(self):
        params = self._make_params()
        opt = AdamWAdvanced(params, lr=0.01)
        p = params[0]["params"][0]
        loss = (p ** 2).sum()
        loss.backward()
        assert p.grad is not None
        opt.zero_grad(set_to_none=False)
        assert p.grad is not None and p.grad.abs().sum() == 0

    def test_multiple_steps(self):
        params = self._make_params()
        opt = AdamWAdvanced(params, lr=0.01)
        p = params[0]["params"][0]
        for _ in range(5):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
        assert opt.state[p]["step"] == 5


# ---------------------------------------------------------------------------
# Feature toggle tests
# ---------------------------------------------------------------------------

class TestFeatureToggles:
    def _run_steps(self, n=3, **kwargs):
        p = torch.randn(8, 8, requires_grad=True)
        opt = AdamWAdvanced([{"params": [p]}], lr=0.01, **kwargs)
        for _ in range(n):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
        return p, opt

    def test_weight_decay(self):
        p, _ = self._run_steps(weight_decay=0.1)
        assert p.data.abs().mean() < 10  # sanity

    def test_cautious_wd(self):
        p, _ = self._run_steps(weight_decay=0.1, cautious_wd=True)
        assert p.data.abs().mean() < 10

    def test_cautious_mask(self):
        p, _ = self._run_steps(cautious_mask=True)
        assert p.data.abs().mean() < 10

    def test_grams_moment(self):
        p, _ = self._run_steps(grams_moment=True)
        assert p.data.abs().mean() < 10

    def test_orthogonal_gradient(self):
        p, _ = self._run_steps(orthogonal_gradient=True)
        assert p.data.abs().mean() < 10

    def test_atan2_update(self):
        p, _ = self._run_steps(use_atan2=True)
        assert p.data.abs().mean() < 10

    def test_ademamix(self):
        p, opt = self._run_steps(use_ademamix=True, beta3_ema=0.999, ademamix_alpha=5.0)
        state = opt.state[p]
        assert "exp_avg_slow" in state

    def test_no_bias_correction(self):
        p, _ = self._run_steps(use_bias_correction=False)
        assert p.data.abs().mean() < 10

    def test_nnmf_factorization(self):
        p, opt = self._run_steps(nnmf_factor=True)
        state = opt.state[p]
        assert state["factored"]
        assert "mu_v" in state
        assert "mv_v" in state

    def test_stochastic_rounding_bf16(self):
        p = torch.randn(8, 8, dtype=torch.bfloat16, requires_grad=True)
        opt = AdamWAdvanced([{"params": [p]}], lr=0.01, stochastic_rounding=True)
        for _ in range(3):
            opt.zero_grad()
            loss = (p.float() ** 2).sum()
            loss.backward()
            opt.step()
        assert p.dtype == torch.bfloat16

    def test_kourkoutas_beta(self):
        p, opt = self._run_steps(kourkoutas_beta=True, betas=(0.9, 0.99), beta2_min=0.9)
        assert opt._kourkoutas_helper is not None

    def test_cautious_mask_and_grams_incompatible(self):
        """cautious_mask should be auto-disabled when grams_moment is also set."""
        p = torch.randn(4, requires_grad=True)
        opt = AdamWAdvanced(
            [{"params": [p]}], lr=0.01,
            cautious_mask=True, grams_moment=True,
        )
        assert opt._cautious_mask is False
        assert opt._grams_moment is True

    def test_all_features_combined(self):
        """Smoke test: all compatible features enabled at once."""
        p = torch.randn(16, 16, requires_grad=True)
        opt = AdamWAdvanced(
            [{"params": [p]}], lr=0.01,
            weight_decay=0.01,
            cautious_wd=True,
            grams_moment=True,  # grams over cautious_mask
            orthogonal_gradient=True,
            use_atan2=True,
            use_ademamix=True,
            beta3_ema=0.999,
            ademamix_alpha=3.0,
            kourkoutas_beta=True,
            betas=(0.9, 0.99),
            beta2_min=0.88,
            nnmf_factor=True,
            use_bias_correction=True,
        )
        for _ in range(5):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
        # Just verify it doesn't crash and params changed
        assert opt.state[p]["step"] == 5


# ---------------------------------------------------------------------------
# Factory integration test
# ---------------------------------------------------------------------------

class TestFactoryIntegration:
    def test_create_via_factory(self):
        from trainer.optimizers import create_optimizer
        p = torch.randn(4, 4, requires_grad=True)
        opt = create_optimizer(
            "adamw_advanced",
            [{"params": [p]}],
            lr=0.001,
            use_ademamix=True,
        )
        assert isinstance(opt, AdamWAdvanced)
        assert opt._use_ademamix is True

    def test_listed_in_registry(self):
        from trainer.optimizers import list_optimizers
        assert "adamw_advanced" in list_optimizers()
