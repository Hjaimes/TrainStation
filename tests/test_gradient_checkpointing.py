"""Tests that gradient checkpointing config is respected by strategies."""
import inspect
import pytest
from trainer.config.schema import TrainConfig


def _make_config(**model_overrides):
    model = {"architecture": "wan", "base_model_path": "/fake", **model_overrides}
    return TrainConfig(
        model=model,
        training={"method": "full_finetune"},
        data={"dataset_config_path": "/fake.toml"},
    )


class TestGradientCheckpointingConfig:
    def test_gradient_checkpointing_default_true(self):
        cfg = _make_config()
        assert cfg.model.gradient_checkpointing is True

    def test_gradient_checkpointing_can_disable(self):
        cfg = _make_config(gradient_checkpointing=False)
        assert cfg.model.gradient_checkpointing is False


class TestGradientCheckpointingInStrategy:
    def test_wan_strategy_enables_checkpointing(self):
        from trainer.arch.wan.strategy import WanStrategy
        source = inspect.getsource(WanStrategy.setup)
        assert "enable_gradient_checkpointing" in source

    def test_flux2_strategy_enables_checkpointing(self):
        from trainer.arch.flux_2.strategy import Flux2Strategy
        source = inspect.getsource(Flux2Strategy.setup)
        assert "enable_gradient_checkpointing" in source

    def test_flux2_no_dead_batch_key_checkpointing(self):
        """Verify dead code (batch.get('gradient_checkpointing')) was removed."""
        from trainer.arch.flux_2.strategy import Flux2Strategy
        source = inspect.getsource(Flux2Strategy.training_step)
        assert "gradient_checkpointing" not in source
