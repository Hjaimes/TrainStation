"""Optimizer and scheduler factory tests."""
import sys
import importlib
import pytest
import torch
import torch.nn as nn

from trainer.optimizers import (
    create_optimizer, list_optimizers, OPTIMIZERS,
    _get_lion, _get_came, _get_schedule_free_adamw,
)
from trainer.schedulers import create_scheduler, list_schedulers, SCHEDULERS
from trainer.errors import TrainerError


@pytest.fixture
def simple_params():
    model = nn.Linear(10, 10)
    return [{"params": list(model.parameters()), "lr": 1e-4}]


class TestOptimizerFactory:
    def test_adamw(self, simple_params):
        opt = create_optimizer("adamw", simple_params, lr=1e-4)
        assert type(opt).__name__ == "AdamW"

    def test_adam(self, simple_params):
        opt = create_optimizer("adam", simple_params, lr=1e-4)
        assert type(opt).__name__ == "Adam"

    def test_sgd(self, simple_params):
        opt = create_optimizer("sgd", simple_params, lr=1e-4)
        assert type(opt).__name__ == "SGD"

    def test_case_insensitive(self, simple_params):
        opt = create_optimizer("AdamW", simple_params, lr=1e-4)
        assert type(opt).__name__ == "AdamW"

    def test_unknown_raises(self, simple_params):
        with pytest.raises(Exception):
            create_optimizer("nonexistent_optimizer_xyz", simple_params, lr=1e-4)

    def test_list_optimizers(self):
        opts = list_optimizers()
        assert "adamw" in opts
        assert "adam" in opts
        assert "sgd" in opts

    def test_list_optimizers_includes_new(self):
        opts = list_optimizers()
        assert "lion" in opts
        assert "came" in opts
        assert "schedule_free_adamw" in opts

    def test_optimizers_dict(self):
        assert "adamw" in OPTIMIZERS
        assert callable(OPTIMIZERS["adamw"])

    def test_lion_missing_package(self, monkeypatch):
        """_get_lion() raises TrainerError with package name when lion-pytorch is absent."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "lion_pytorch":
                raise ImportError("No module named 'lion_pytorch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(TrainerError, match="lion-pytorch"):
            _get_lion()

    def test_came_missing_package(self, monkeypatch):
        """_get_came() raises TrainerError with package name when came-pytorch is absent."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "came_pytorch":
                raise ImportError("No module named 'came_pytorch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(TrainerError, match="came-pytorch"):
            _get_came()

    def test_schedule_free_adamw_missing_package(self, monkeypatch):
        """_get_schedule_free_adamw() raises TrainerError with package name when schedulefree is absent."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "schedulefree":
                raise ImportError("No module named 'schedulefree'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(TrainerError, match="schedulefree"):
            _get_schedule_free_adamw()

    def test_lion_installed(self, simple_params):
        """If lion-pytorch is installed, create_optimizer('lion') works."""
        lion_pytorch = pytest.importorskip("lion_pytorch")
        opt = create_optimizer("lion", simple_params, lr=1e-4)
        assert type(opt).__name__ == "Lion"

    def test_came_installed(self, simple_params):
        """If came-pytorch is installed, create_optimizer('came') works."""
        came_pytorch = pytest.importorskip("came_pytorch")
        opt = create_optimizer("came", simple_params, lr=1e-4)
        assert type(opt).__name__ == "CAME"

    def test_schedule_free_adamw_installed(self, simple_params):
        """If schedulefree is installed, create_optimizer('schedule_free_adamw') works."""
        schedulefree = pytest.importorskip("schedulefree")
        opt = create_optimizer("schedule_free_adamw", simple_params, lr=1e-4)
        assert type(opt).__name__ == "AdamWScheduleFree"


class TestSchedulerFactory:
    def test_rex(self, simple_params):
        opt = create_optimizer("adamw", simple_params, lr=1e-4)
        sched = create_scheduler("rex", opt, num_training_steps=100, warmup_steps=10)
        assert type(sched).__name__ == "RexLR"

    def test_cosine(self, simple_params):
        opt = create_optimizer("adamw", simple_params, lr=1e-4)
        sched = create_scheduler("cosine", opt, num_training_steps=100, warmup_steps=10)
        # transformers schedulers are LambdaLR under the hood
        assert sched is not None

    def test_constant(self, simple_params):
        opt = create_optimizer("adamw", simple_params, lr=1e-4)
        sched = create_scheduler("constant", opt, num_training_steps=100)
        assert sched is not None

    def test_linear(self, simple_params):
        opt = create_optimizer("adamw", simple_params, lr=1e-4)
        sched = create_scheduler("linear", opt, num_training_steps=100, warmup_steps=10)
        assert sched is not None

    def test_unknown_raises(self, simple_params):
        opt = create_optimizer("adamw", simple_params, lr=1e-4)
        with pytest.raises(Exception):
            create_scheduler("nonexistent_scheduler_xyz", opt, num_training_steps=100)

    def test_list_schedulers(self):
        scheds = list_schedulers()
        assert "rex" in scheds
        assert "cosine" in scheds
        assert "constant" in scheds

    def test_rex_lr_decreases(self, simple_params):
        opt = create_optimizer("adamw", simple_params, lr=1e-4)
        sched = create_scheduler("rex", opt, num_training_steps=100, warmup_steps=0)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(50):
            opt.step()
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_list_schedulers_includes_new(self):
        scheds = list_schedulers()
        assert "exponential" in scheds
        assert "inverse_sqrt" in scheds

    def test_exponential_warmup_then_decay(self, simple_params):
        """Exponential: LR increases during warmup, then decays geometrically."""
        opt = create_optimizer("adamw", simple_params, lr=1e-3)
        warmup = 10
        sched = create_scheduler(
            "exponential", opt, num_training_steps=200, warmup_steps=warmup, gamma=0.999,
        )
        assert sched is not None

        # Collect LRs across warmup and post-warmup
        lrs = []
        for _ in range(50):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])

        # LR at end of warmup should be higher than at step 1
        assert lrs[warmup - 1] > lrs[0]
        # Post-warmup LR should decrease
        assert lrs[-1] < lrs[warmup]

    def test_exponential_custom_gamma(self, simple_params):
        """Exponential with gamma=1.0 produces no decay after warmup."""
        opt = create_optimizer("adamw", simple_params, lr=1e-3)
        sched = create_scheduler(
            "exponential", opt, num_training_steps=50, warmup_steps=0, gamma=1.0,
        )
        base_lr = opt.param_groups[0]["lr"]
        for _ in range(20):
            sched.step()
        # gamma=1.0: lr_lambda(step) = 1.0**step = 1.0, so LR stays constant
        assert abs(opt.param_groups[0]["lr"] - base_lr) < 1e-12

    def test_inverse_sqrt_warmup_then_decay(self, simple_params):
        """Inverse sqrt: LR increases during warmup, then decays as 1/sqrt(step)."""
        opt = create_optimizer("adamw", simple_params, lr=1e-3)
        warmup = 10
        sched = create_scheduler(
            "inverse_sqrt", opt, num_training_steps=200, warmup_steps=warmup,
        )
        assert sched is not None

        lrs = []
        for _ in range(50):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])

        # Warmup should increase LR
        assert lrs[warmup - 1] > lrs[0]
        # Post-warmup should decrease
        assert lrs[-1] < lrs[warmup]

    def test_inverse_sqrt_no_warmup(self, simple_params):
        """Inverse sqrt with warmup=0 decays from the first step."""
        opt = create_optimizer("adamw", simple_params, lr=1e-3)
        sched = create_scheduler(
            "inverse_sqrt", opt, num_training_steps=100, warmup_steps=0,
        )
        lr_step1 = None
        lr_step20 = None
        for i in range(20):
            sched.step()
            lr = opt.param_groups[0]["lr"]
            if i == 0:
                lr_step1 = lr
            lr_step20 = lr
        assert lr_step20 < lr_step1
