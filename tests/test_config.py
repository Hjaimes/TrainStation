"""Config creation, round-trip serialization, validation, and overrides."""
import json
import pytest
from pathlib import Path

from trainer.config.schema import TrainConfig, ModelConfig, NetworkConfig
from trainer.config.io import load_config, save_config, apply_overrides, load_config_from_dict
from trainer.config.validation import validate_config


def _make_config(**overrides) -> TrainConfig:
    """Create a minimal valid TrainConfig for testing."""
    defaults = {
        "model": {"architecture": "wan", "base_model_path": "/fake/model.safetensors"},
        "training": {"method": "lora"},
        "network": {"network_type": "lora", "rank": 16, "alpha": 16.0},
        "data": {"dataset_config_path": "dataset.toml"},
    }
    defaults.update(overrides)
    return TrainConfig.model_validate(defaults)


class TestConfigCreation:
    def test_minimal_config(self):
        cfg = _make_config()
        assert cfg.model.architecture == "wan"
        assert cfg.training.method == "lora"
        assert cfg.network is not None
        assert cfg.network.rank == 16

    def test_full_finetune_no_network(self):
        cfg = _make_config(
            training={"method": "full_finetune"},
            network=None,
        )
        assert cfg.network is None
        assert cfg.training.method == "full_finetune"

    def test_lora_requires_network(self):
        with pytest.raises(ValueError, match="requires a .network. section"):
            _make_config(network=None)

    def test_full_finetune_rejects_network(self):
        with pytest.raises(ValueError, match="should not have a .network. section"):
            _make_config(
                training={"method": "full_finetune"},
                network={"network_type": "lora", "rank": 16},
            )

    def test_empty_model_path_rejected(self):
        with pytest.raises(ValueError, match="base_model_path must not be empty"):
            _make_config(model={"architecture": "wan", "base_model_path": ""})

    def test_no_dataset_rejected(self):
        with pytest.raises(ValueError, match="dataset_config_path or data.datasets"):
            _make_config(data={})

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            _make_config(model={
                "architecture": "wan",
                "base_model_path": "/fake/model",
                "nonexistent_field": True,
            })

    def test_defaults_applied(self):
        cfg = _make_config()
        assert cfg.training.batch_size == 1
        assert cfg.optimizer.learning_rate == 1e-4
        assert cfg.saving.output_dir == "./output"
        assert cfg.sampling.enabled is False


class TestConfigRoundTrip:
    def test_dict_round_trip(self):
        cfg = _make_config()
        d = cfg.to_dict()
        cfg2 = TrainConfig.from_dict(d)
        assert cfg2.model.architecture == cfg.model.architecture
        assert cfg2.training.method == cfg.training.method
        assert cfg2.network.rank == cfg.network.rank

    def test_json_round_trip(self, tmp_path):
        cfg = _make_config()
        path = tmp_path / "test_config.json"
        save_config(cfg, path)
        cfg2 = load_config(path)
        assert cfg2.model.architecture == cfg.model.architecture
        assert cfg2.optimizer.learning_rate == cfg.optimizer.learning_rate

    def test_yaml_round_trip(self, tmp_path):
        cfg = _make_config()
        path = tmp_path / "test_config.yaml"
        save_config(cfg, path)
        cfg2 = load_config(path)
        assert cfg2.model.architecture == cfg.model.architecture
        assert cfg2.network.rank == cfg.network.rank

    def test_freeze_returns_copy(self):
        cfg = _make_config()
        frozen = cfg.freeze()
        frozen.training.batch_size = 999
        assert cfg.training.batch_size == 1

    def test_load_from_dict(self):
        d = _make_config().to_dict()
        cfg = load_config_from_dict(d)
        assert cfg.model.architecture == "wan"


class TestConfigOverrides:
    def test_simple_override(self):
        cfg = _make_config()
        cfg2 = apply_overrides(cfg, ["training.batch_size=4"])
        assert cfg2.training.batch_size == 4

    def test_float_override(self):
        cfg = _make_config()
        cfg2 = apply_overrides(cfg, ["optimizer.learning_rate=5e-5"])
        assert cfg2.optimizer.learning_rate == 5e-5

    def test_bool_override(self):
        cfg = _make_config()
        cfg2 = apply_overrides(cfg, ["model.gradient_checkpointing=false"])
        assert cfg2.model.gradient_checkpointing is False

    def test_unknown_field_rejected(self):
        cfg = _make_config()
        with pytest.raises(ValueError, match="Unknown config field"):
            apply_overrides(cfg, ["training.nonexistent=42"])

    def test_bad_format_rejected(self):
        cfg = _make_config()
        with pytest.raises(ValueError, match="must be 'key=value'"):
            apply_overrides(cfg, ["no_equals_sign"])


class TestConfigValidation:
    def test_valid_config_has_no_errors(self, tmp_path):
        # Create a fake model file for validation
        model_path = tmp_path / "model.safetensors"
        model_path.touch()
        cfg = _make_config(model={
            "architecture": "wan",
            "base_model_path": str(model_path),
        })
        result = validate_config(cfg)
        assert not result.has_errors
        assert result.can_train

    def test_missing_model_path_is_error(self):
        cfg = _make_config()
        result = validate_config(cfg)
        assert result.has_errors
        assert any("does not exist" in e.message for e in result.errors)

    def test_high_batch_size_warning(self, tmp_path):
        model_path = tmp_path / "model.safetensors"
        model_path.touch()
        cfg = _make_config(
            model={"architecture": "wan", "base_model_path": str(model_path)},
            training={"method": "lora", "batch_size": 4, "gradient_accumulation_steps": 4},
        )
        result = validate_config(cfg)
        assert any("batch size" in w.message.lower() for w in result.warnings)

    def test_info_summary_present(self, tmp_path):
        model_path = tmp_path / "model.safetensors"
        model_path.touch()
        cfg = _make_config(model={
            "architecture": "wan",
            "base_model_path": str(model_path),
        })
        result = validate_config(cfg)
        assert len(result.info) > 0
        assert "wan" in result.info[0].message


class TestConfigLoadFromFile:
    def test_load_yaml(self, minimal_config_path):
        if minimal_config_path.exists():
            cfg = load_config(minimal_config_path)
            assert cfg.model.architecture == "wan"

    def test_unknown_format_rejected(self, tmp_path):
        bad_path = tmp_path / "config.xml"
        bad_path.write_text("<config/>")
        with pytest.raises(ValueError, match="Unknown config format"):
            load_config(bad_path)
