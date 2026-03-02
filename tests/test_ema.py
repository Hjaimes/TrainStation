"""Tests for EMATracker in trainer/ema.py."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from trainer.ema import EMATracker


def _make_params(values: list[float]) -> list[nn.Parameter]:
    """Create a list of scalar Parameters with given initial values."""
    return [nn.Parameter(torch.tensor([v])) for v in values]


def _make_model(values: list[float]) -> nn.Linear:
    """Create a simple linear model whose parameters are seeded deterministically."""
    model = nn.Linear(len(values), 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([values], dtype=torch.float32))
    return model


class TestShadowInitialization:
    def test_shadow_initialized_to_params(self):
        params = _make_params([1.0, 2.0, 3.0])
        ema = EMATracker(params, decay=0.9999)

        for shadow, param in zip(ema.shadow_params, params):
            assert torch.allclose(shadow, param.data.float()), (
                f"Shadow {shadow} does not match param {param.data}"
            )

    def test_shadow_on_cpu(self):
        params = _make_params([1.0, -2.5, 0.0])
        ema = EMATracker(params, decay=0.9999, device="cpu")

        for shadow in ema.shadow_params:
            assert shadow.device.type == "cpu", (
                f"Shadow param is on {shadow.device}, expected cpu"
            )

    def test_shadow_dtype_is_fp32(self):
        # Create half-precision params to verify upcasting.
        params = [nn.Parameter(torch.tensor([1.0], dtype=torch.float16))]
        ema = EMATracker(params, decay=0.9999)

        for shadow in ema.shadow_params:
            assert shadow.dtype == torch.float32, (
                f"Shadow dtype is {shadow.dtype}, expected float32"
            )


class TestDecaySchedule:
    def test_warm_start_at_step_0(self):
        params = _make_params([0.0])
        ema = EMATracker(params, decay=0.9999)
        # (1+0) / (10+0) = 0.1
        assert abs(ema.get_decay(0) - 0.1) < 1e-6

    def test_warm_start_at_step_90(self):
        params = _make_params([0.0])
        ema = EMATracker(params, decay=0.9999)
        # (1+90) / (10+90) = 91/100 = 0.91
        assert abs(ema.get_decay(90) - 0.91) < 1e-6

    def test_decay_capped_at_max(self):
        params = _make_params([0.0])
        max_decay = 0.9999
        ema = EMATracker(params, decay=max_decay)
        # At a very large step the formula would exceed max_decay; must be capped.
        assert ema.get_decay(10_000_000) == max_decay

    def test_decay_monotonically_increasing(self):
        params = _make_params([0.0])
        ema = EMATracker(params, decay=0.9999)
        steps = [0, 1, 5, 10, 50, 100, 500, 1000]
        decays = [ema.get_decay(s) for s in steps]
        for a, b in zip(decays, decays[1:]):
            assert a <= b, f"Decay not monotone: {a} > {b}"


class TestStep:
    def test_step_moves_shadow_toward_params(self):
        """After changing params and calling step, shadow should move toward new value."""
        params = _make_params([0.0])
        ema = EMATracker(params, decay=0.9999)

        # Abruptly change the parameter.
        with torch.no_grad():
            params[0].data.fill_(10.0)

        initial_shadow = ema.shadow_params[0].item()
        ema.step(params, global_step=0)
        updated_shadow = ema.shadow_params[0].item()

        assert updated_shadow > initial_shadow, (
            "Shadow did not move toward the updated param value"
        )
        assert updated_shadow < 10.0, (
            "Shadow should not immediately equal the param"
        )

    def test_step_does_not_mutate_params(self):
        """step() must never modify the model parameters themselves."""
        params = _make_params([5.0, -3.0])
        ema = EMATracker(params, decay=0.9)
        original = [p.data.clone() for p in params]

        ema.step(params, global_step=10)

        for orig, param in zip(original, params):
            assert torch.allclose(orig, param.data), "step() mutated model params"

    def test_multiple_steps_converge(self):
        """After many steps with fixed params, shadow should be very close to params."""
        target = 7.0
        params = _make_params([target])
        # Start shadow far from target.
        ema = EMATracker(params, decay=0.99)
        with torch.no_grad():
            ema.shadow_params[0].fill_(0.0)

        for step in range(2000):
            ema.step(params, global_step=step)

        diff = abs(ema.shadow_params[0].item() - target)
        assert diff < 0.05, f"Shadow did not converge: diff={diff:.4f}"


class TestCopyToAndRestore:
    def test_copy_to_replaces_model_weights_with_shadow(self):
        params = _make_params([1.0, 2.0])
        ema = EMATracker(params, decay=0.9999)

        # Set shadow to a known distinct value.
        with torch.no_grad():
            for s in ema.shadow_params:
                s.fill_(99.0)

        ema.copy_to(params)

        for param in params:
            assert torch.allclose(param.data, torch.tensor([99.0])), (
                f"copy_to did not overwrite param: {param.data}"
            )

    def test_restore_brings_back_originals(self):
        params = _make_params([1.0, 2.0])
        original_values = [p.data.clone() for p in params]
        ema = EMATracker(params, decay=0.9999)

        # Overwrite shadow with something different.
        with torch.no_grad():
            for s in ema.shadow_params:
                s.fill_(99.0)

        ema.copy_to(params)
        ema.restore(params)

        for orig, param in zip(original_values, params):
            assert torch.allclose(orig, param.data), (
                f"restore() did not recover original: expected {orig}, got {param.data}"
            )

    def test_restore_clears_backup(self):
        params = _make_params([3.0])
        ema = EMATracker(params, decay=0.9)
        ema.copy_to(params)
        ema.restore(params)
        assert ema._backup == [], "backup should be empty after restore()"

    def test_copy_to_preserves_param_dtype(self):
        """copy_to must cast shadow (fp32) to the original param dtype."""
        params = [nn.Parameter(torch.tensor([1.0], dtype=torch.float32))]
        ema = EMATracker(params, decay=0.9)
        original_dtype = params[0].data.dtype

        ema.copy_to(params)

        assert params[0].data.dtype == original_dtype, (
            f"copy_to changed dtype from {original_dtype} to {params[0].data.dtype}"
        )


class TestStateDictRoundtrip:
    def test_state_dict_roundtrip(self):
        params = _make_params([1.0, -2.0, 3.5])
        ema = EMATracker(params, decay=0.999)

        # Run a few steps so shadow diverges from initial.
        with torch.no_grad():
            for p in params:
                p.data.fill_(10.0)
        for step in range(5):
            ema.step(params, global_step=step)

        state = ema.state_dict()

        # Load into a fresh tracker initialised with different params.
        params2 = _make_params([0.0, 0.0, 0.0])
        ema2 = EMATracker(params2, decay=0.5)
        ema2.load_state_dict(state)

        assert ema2.max_decay == 0.999, "max_decay not restored"
        for s1, s2 in zip(ema.shadow_params, ema2.shadow_params):
            assert torch.allclose(s1, s2), (
                f"Shadow mismatch after load_state_dict: {s1} vs {s2}"
            )

    def test_state_dict_shadow_params_are_clones(self):
        """Mutating shadow after state_dict() must not corrupt the saved state."""
        params = _make_params([1.0])
        ema = EMATracker(params, decay=0.9)
        state = ema.state_dict()

        with torch.no_grad():
            ema.shadow_params[0].fill_(999.0)

        assert not torch.allclose(
            state["shadow_params"][0], ema.shadow_params[0]
        ), "state_dict returned a view, not a clone"
