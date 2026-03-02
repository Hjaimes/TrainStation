"""Tests for LR scaling by effective batch size."""
import math
import pytest

from trainer.training.trainer import _scale_lr
from trainer.errors import ConfigError


class TestScaleLr:
    def test_none_returns_base_lr_unchanged(self):
        lr = _scale_lr(1e-4, batch_size=4, grad_accum=2, method="none")
        assert lr == 1e-4

    def test_none_ignores_batch_and_accum(self):
        # batch_size and grad_accum should have no effect with "none"
        assert _scale_lr(5e-5, batch_size=1, grad_accum=1, method="none") == 5e-5
        assert _scale_lr(5e-5, batch_size=8, grad_accum=4, method="none") == 5e-5

    def test_linear_multiplies_by_effective_batch(self):
        # effective = 4 * 2 = 8
        lr = _scale_lr(1e-4, batch_size=4, grad_accum=2, method="linear")
        assert math.isclose(lr, 1e-4 * 8, rel_tol=1e-9)

    def test_linear_batch1_accum1(self):
        # effective = 1, so lr unchanged
        lr = _scale_lr(3e-4, batch_size=1, grad_accum=1, method="linear")
        assert math.isclose(lr, 3e-4, rel_tol=1e-9)

    def test_linear_large_batch(self):
        # effective = 16 * 4 = 64
        lr = _scale_lr(1e-5, batch_size=16, grad_accum=4, method="linear")
        assert math.isclose(lr, 1e-5 * 64, rel_tol=1e-9)

    def test_sqrt_multiplies_by_sqrt_effective_batch(self):
        # effective = 4 * 4 = 16, sqrt(16) = 4.0
        lr = _scale_lr(1e-4, batch_size=4, grad_accum=4, method="sqrt")
        assert math.isclose(lr, 1e-4 * 4.0, rel_tol=1e-9)

    def test_sqrt_batch1_accum1(self):
        # effective = 1, sqrt(1) = 1.0, so lr unchanged
        lr = _scale_lr(2e-4, batch_size=1, grad_accum=1, method="sqrt")
        assert math.isclose(lr, 2e-4, rel_tol=1e-9)

    def test_sqrt_non_perfect_square(self):
        # effective = 2 * 1 = 2, sqrt(2) ≈ 1.41421356
        lr = _scale_lr(1e-4, batch_size=2, grad_accum=1, method="sqrt")
        assert math.isclose(lr, 1e-4 * math.sqrt(2), rel_tol=1e-9)

    def test_unknown_method_raises_config_error(self):
        with pytest.raises(ConfigError, match="Unknown lr_scaling method"):
            _scale_lr(1e-4, batch_size=1, grad_accum=1, method="invalid_mode")

    def test_unknown_method_message_includes_method_name(self):
        with pytest.raises(ConfigError, match="bogus"):
            _scale_lr(1e-4, batch_size=1, grad_accum=1, method="bogus")

    def test_linear_vs_sqrt_ordering(self):
        """For effective batch > 1, linear > sqrt > none."""
        base = 1e-4
        b, a = 4, 4  # effective = 16
        lr_none = _scale_lr(base, b, a, "none")
        lr_sqrt = _scale_lr(base, b, a, "sqrt")
        lr_linear = _scale_lr(base, b, a, "linear")
        assert lr_none < lr_sqrt < lr_linear
