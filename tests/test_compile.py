"""Test torch.compile config field."""
import pytest
from trainer.config.schema import TrainConfig


def _make_config(**overrides):
    model = {"architecture": "wan", "base_model_path": "/fake", **overrides}
    return TrainConfig(
        model=model,
        training={"method": "full_finetune"},
        data={"dataset_config_path": "/fake.toml"},
    )


class TestCompileConfig:
    def test_compile_default_false(self):
        cfg = _make_config()
        assert cfg.model.compile_model is False

    def test_compile_can_enable(self):
        cfg = _make_config(compile_model=True)
        assert cfg.model.compile_model is True


class TestCompileHelper:
    def test_helper_exists_on_strategy(self):
        from trainer.arch.base import ModelStrategy
        assert hasattr(ModelStrategy, '_maybe_compile_model')

    def test_compile_disabled_returns_same_model(self):
        import torch.nn as nn
        from trainer.arch.base import ModelStrategy
        cfg = _make_config()
        strategy = ModelStrategy(cfg)
        model = nn.Linear(4, 4)
        result = strategy._maybe_compile_model(model, cfg)
        assert result is model  # same object, not compiled
