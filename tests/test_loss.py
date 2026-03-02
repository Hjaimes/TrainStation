"""Tests for trainer/loss.py — configurable loss functions."""
from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

from trainer.loss import LossFn, compute_loss, get_loss_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pred_target() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a pair of random tensors with fixed seed for reproducibility."""
    rng = torch.Generator()
    rng.manual_seed(42)
    pred = torch.randn(4, 8, generator=rng)
    target = torch.randn(4, 8, generator=rng)
    return pred, target


# ---------------------------------------------------------------------------
# get_loss_fn — callable resolution
# ---------------------------------------------------------------------------

class TestGetLossFn:
    def test_mse_returns_callable(self):
        fn = get_loss_fn("mse")
        assert callable(fn)

    def test_l1_returns_callable(self):
        fn = get_loss_fn("l1")
        assert callable(fn)

    def test_mae_returns_callable(self):
        fn = get_loss_fn("mae")
        assert callable(fn)

    def test_huber_returns_callable(self):
        fn = get_loss_fn("huber")
        assert callable(fn)

    def test_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_fn("cross_entropy")

    def test_invalid_type_message_mentions_supported(self):
        with pytest.raises(ValueError, match="mse"):
            get_loss_fn("bad_loss")

    def test_l1_and_mae_return_same_function(self, pred_target):
        pred, target = pred_target
        fn_l1 = get_loss_fn("l1")
        fn_mae = get_loss_fn("mae")
        assert torch.equal(fn_l1(pred, target), fn_mae(pred, target))


# ---------------------------------------------------------------------------
# Loss correctness — results match torch reference implementations
# ---------------------------------------------------------------------------

class TestLossCorrectness:
    def test_mse_matches_reference(self, pred_target):
        pred, target = pred_target
        fn = get_loss_fn("mse")
        result = fn(pred, target)
        expected = F.mse_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_l1_matches_reference(self, pred_target):
        pred, target = pred_target
        fn = get_loss_fn("l1")
        result = fn(pred, target)
        expected = F.l1_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_mae_matches_l1_reference(self, pred_target):
        pred, target = pred_target
        fn = get_loss_fn("mae")
        result = fn(pred, target)
        expected = F.l1_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_huber_default_delta_matches_reference(self, pred_target):
        pred, target = pred_target
        fn = get_loss_fn("huber")
        result = fn(pred, target)
        expected = F.huber_loss(pred, target, reduction="mean", delta=1.0)
        assert torch.allclose(result, expected)

    def test_huber_custom_delta_matches_reference(self, pred_target):
        pred, target = pred_target
        delta = 0.5
        fn = get_loss_fn("huber", delta=delta)
        result = fn(pred, target)
        expected = F.huber_loss(pred, target, reduction="mean", delta=delta)
        assert torch.allclose(result, expected)

    def test_huber_delta_2_matches_reference(self, pred_target):
        pred, target = pred_target
        delta = 2.0
        fn = get_loss_fn("huber", delta=delta)
        result = fn(pred, target)
        expected = F.huber_loss(pred, target, reduction="mean", delta=delta)
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Scalar output — all loss functions must return 0-dim tensors
# ---------------------------------------------------------------------------

class TestScalarOutput:
    @pytest.mark.parametrize("loss_type", ["mse", "l1", "mae", "huber"])
    def test_returns_scalar_tensor(self, loss_type, pred_target):
        pred, target = pred_target
        fn = get_loss_fn(loss_type)
        result = fn(pred, target)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0, f"Expected scalar (0-dim), got shape {result.shape}"

    @pytest.mark.parametrize("loss_type", ["mse", "l1", "mae", "huber"])
    def test_result_is_finite(self, loss_type, pred_target):
        pred, target = pred_target
        fn = get_loss_fn(loss_type)
        result = fn(pred, target)
        assert torch.isfinite(result)

    @pytest.mark.parametrize("loss_type", ["mse", "l1", "mae", "huber"])
    def test_identical_inputs_give_zero_loss(self, loss_type):
        t = torch.ones(4, 8)
        fn = get_loss_fn(loss_type)
        result = fn(t, t)
        assert torch.allclose(result, torch.zeros_like(result))


# ---------------------------------------------------------------------------
# compute_loss — convenience wrapper
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_default_loss_type_is_mse(self, pred_target):
        pred, target = pred_target
        result = compute_loss(pred, target)
        expected = F.mse_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_explicit_mse(self, pred_target):
        pred, target = pred_target
        result = compute_loss(pred, target, loss_type="mse")
        expected = F.mse_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_explicit_l1(self, pred_target):
        pred, target = pred_target
        result = compute_loss(pred, target, loss_type="l1")
        expected = F.l1_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_explicit_mae(self, pred_target):
        pred, target = pred_target
        result = compute_loss(pred, target, loss_type="mae")
        expected = F.l1_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_explicit_huber_default_delta(self, pred_target):
        pred, target = pred_target
        result = compute_loss(pred, target, loss_type="huber")
        expected = F.huber_loss(pred, target, reduction="mean", delta=1.0)
        assert torch.allclose(result, expected)

    def test_explicit_huber_custom_delta(self, pred_target):
        pred, target = pred_target
        result = compute_loss(pred, target, loss_type="huber", delta=0.25)
        expected = F.huber_loss(pred, target, reduction="mean", delta=0.25)
        assert torch.allclose(result, expected)

    def test_invalid_type_raises_value_error(self, pred_target):
        pred, target = pred_target
        with pytest.raises(ValueError, match="Unknown loss type"):
            compute_loss(pred, target, loss_type="focal")

    def test_returns_scalar_tensor(self, pred_target):
        pred, target = pred_target
        result = compute_loss(pred, target)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
