"""Registry: discovery, registration, resolution."""
import pytest

from trainer.registry import register_model, get_model_strategy, list_models, _model_strategies, _discovered
from trainer.arch.base import ModelStrategy
from trainer.config.schema import TrainConfig


def _make_config() -> TrainConfig:
    return TrainConfig.model_validate({
        "model": {"architecture": "test", "base_model_path": "/fake"},
        "data": {"dataset_config_path": "d.toml"},
        "training": {"method": "lora"},
        "network": {"rank": 16},
    })


class TestRegistry:
    def test_list_models_returns_list(self):
        result = list_models()
        assert isinstance(result, list)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_model_strategy("nonexistent_model_xyz")

    def test_register_and_resolve(self):
        @register_model("_test_dummy")
        class DummyStrategy(ModelStrategy):
            @property
            def architecture(self):
                return "_test_dummy"

            def setup(self):
                pass

            def training_step(self, components, batch, step):
                pass

        cls = get_model_strategy("_test_dummy")
        assert cls is DummyStrategy

        # Cleanup
        _model_strategies.pop("_test_dummy", None)

    def test_register_overwrite_warns(self, caplog):
        @register_model("_test_overwrite")
        class First(ModelStrategy):
            pass

        import logging
        with caplog.at_level(logging.WARNING):
            @register_model("_test_overwrite")
            class Second(ModelStrategy):
                pass

        assert "overwriting" in caplog.text

        # Cleanup
        _model_strategies.pop("_test_overwrite", None)
