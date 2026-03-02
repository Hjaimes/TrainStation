"""Tests for validation loss system (Task 21).

Covers:
- ValidationConfig defaults and field assignment
- ValidationRunner.run returns dict with val_loss
- ValidationRunner cycles the dataloader when exhausted
- torch.no_grad() is active during run()
- on_validation_end callback method exists on TrainingCallback
- TrainConfig has a validation field of the correct type
- Config round-trip with validation settings
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, call

from trainer.config.schema import ValidationConfig, TrainConfig, ModelConfig, NetworkConfig
from trainer.callbacks import TrainingCallback
from trainer.training.validation import ValidationRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_strategy(loss_value: float = 0.25):
    """Return a mock ModelStrategy whose training_step yields a fixed loss."""
    output = MagicMock()
    output.loss = torch.tensor(loss_value, requires_grad=False)

    strategy = MagicMock()
    strategy.training_step.return_value = output
    return strategy


def _make_mock_dataloader(num_batches: int = 3):
    """Return a mock DataLoader with a fixed number of batches."""
    batches = [{"latents": torch.zeros(1, 4)} for _ in range(num_batches)]
    dataloader = MagicMock()
    dataloader.__iter__ = MagicMock(side_effect=lambda: iter(batches))
    return dataloader


def _make_minimal_train_config(**overrides) -> TrainConfig:
    """Return a minimal TrainConfig that passes cross-validation."""
    base = dict(
        model=ModelConfig(architecture="wan", base_model_path="/fake/model"),
        network=NetworkConfig(rank=4, alpha=4.0),
    )
    base.update(overrides)
    # Provide a dataset path to satisfy the data validation rule
    from trainer.config.schema import DataConfig, DatasetEntry
    if "data" not in base:
        base["data"] = DataConfig(datasets=[DatasetEntry(path="/fake/data")])
    return TrainConfig(**base)


# ---------------------------------------------------------------------------
# ValidationConfig tests
# ---------------------------------------------------------------------------

class TestValidationConfig:
    def test_defaults(self):
        cfg = ValidationConfig()
        assert cfg.enabled is False
        assert cfg.data_path is None
        assert cfg.interval_steps == 500
        assert cfg.num_steps == 10
        assert cfg.fixed_timestep == 0.5

    def test_enabled_can_be_set(self):
        cfg = ValidationConfig(enabled=True, data_path="/val/data.toml")
        assert cfg.enabled is True
        assert cfg.data_path == "/val/data.toml"

    def test_interval_steps_can_be_set(self):
        cfg = ValidationConfig(interval_steps=100)
        assert cfg.interval_steps == 100

    def test_num_steps_can_be_set(self):
        cfg = ValidationConfig(num_steps=50)
        assert cfg.num_steps == 50

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ValidationConfig(nonexistent_field=True)


# ---------------------------------------------------------------------------
# ValidationRunner tests
# ---------------------------------------------------------------------------

class TestValidationRunnerRun:
    def test_returns_dict_with_val_loss(self):
        strategy = _make_mock_strategy(loss_value=0.5)
        components = MagicMock()
        dataloader = _make_mock_dataloader(num_batches=5)

        runner = ValidationRunner(
            strategy=strategy, components=components,
            dataloader=dataloader, num_steps=3,
        )
        metrics = runner.run(step=100)

        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert isinstance(metrics["val_loss"], float)

    def test_val_loss_is_average_of_steps(self):
        """val_loss must equal the mean of per-step losses."""
        strategy = _make_mock_strategy(loss_value=0.4)
        components = MagicMock()
        dataloader = _make_mock_dataloader(num_batches=10)

        runner = ValidationRunner(
            strategy=strategy, components=components,
            dataloader=dataloader, num_steps=5,
        )
        metrics = runner.run(step=0)

        assert abs(metrics["val_loss"] - 0.4) < 1e-6

    def test_calls_on_before_and_after_sampling(self):
        """Strategy eval/train mode toggle must be called exactly once each."""
        strategy = _make_mock_strategy()
        components = MagicMock()
        dataloader = _make_mock_dataloader(num_batches=3)

        runner = ValidationRunner(
            strategy=strategy, components=components,
            dataloader=dataloader, num_steps=2,
        )
        runner.run(step=10)

        strategy.on_before_sampling.assert_called_once_with(components)
        strategy.on_after_sampling.assert_called_once_with(components)

    def test_training_step_called_num_steps_times(self):
        strategy = _make_mock_strategy()
        components = MagicMock()
        dataloader = _make_mock_dataloader(num_batches=10)

        runner = ValidationRunner(
            strategy=strategy, components=components,
            dataloader=dataloader, num_steps=4,
        )
        runner.run(step=50)

        assert strategy.training_step.call_count == 4


class TestValidationRunnerCycling:
    def test_cycles_dataloader_when_exhausted(self):
        """ValidationRunner must restart iteration when the dataloader runs out."""
        strategy = _make_mock_strategy()
        components = MagicMock()

        # Dataloader has only 2 batches but we request 5 steps.
        batches = [{"latents": torch.zeros(1, 4)}, {"latents": torch.zeros(1, 4)}]
        call_count = 0

        def make_iter():
            nonlocal call_count
            call_count += 1
            return iter(batches)

        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(side_effect=make_iter)

        runner = ValidationRunner(
            strategy=strategy, components=components,
            dataloader=dataloader, num_steps=5,
        )
        runner.run(step=0)

        # training_step must have been called 5 times despite only 2 batches
        assert strategy.training_step.call_count == 5
        # iter() must have been called more than once to cycle
        assert call_count > 1


class TestValidationRunnerNoGrad:
    def test_no_grad_active_during_run(self):
        """Gradients must not be computed during validation."""
        grad_enabled_during_step = []

        def mock_training_step(components, batch, step):
            grad_enabled_during_step.append(torch.is_grad_enabled())
            output = MagicMock()
            output.loss = torch.tensor(0.1)
            return output

        strategy = MagicMock()
        strategy.training_step.side_effect = mock_training_step
        components = MagicMock()
        dataloader = _make_mock_dataloader(num_batches=3)

        runner = ValidationRunner(
            strategy=strategy, components=components,
            dataloader=dataloader, num_steps=3,
        )
        runner.run(step=0)

        assert all(not enabled for enabled in grad_enabled_during_step), (
            "torch.no_grad() must be active during all validation steps"
        )

    def test_on_after_sampling_called_even_on_error(self):
        """on_after_sampling (train mode restore) must run even if a step raises."""
        def bad_step(components, batch, step):
            raise RuntimeError("simulated step failure")

        strategy = MagicMock()
        strategy.training_step.side_effect = bad_step
        components = MagicMock()
        dataloader = _make_mock_dataloader(num_batches=3)

        runner = ValidationRunner(
            strategy=strategy, components=components,
            dataloader=dataloader, num_steps=2,
        )
        with pytest.raises(RuntimeError, match="simulated step failure"):
            runner.run(step=0)

        # on_after_sampling must still have been called (finally block)
        strategy.on_after_sampling.assert_called_once_with(components)


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------

class TestOnValidationEndCallback:
    def test_method_exists_on_base_class(self):
        cb = TrainingCallback()
        assert hasattr(cb, "on_validation_end"), (
            "TrainingCallback must have an on_validation_end method"
        )
        assert callable(cb.on_validation_end)

    def test_method_accepts_step_and_metrics(self):
        cb = TrainingCallback()
        # Must not raise
        cb.on_validation_end(step=100, metrics={"val_loss": 0.42})

    def test_method_returns_none(self):
        cb = TrainingCallback()
        result = cb.on_validation_end(step=0, metrics={"val_loss": 0.0})
        assert result is None


# ---------------------------------------------------------------------------
# TrainConfig integration tests
# ---------------------------------------------------------------------------

class TestTrainConfigValidation:
    def test_train_config_has_validation_field(self):
        cfg = _make_minimal_train_config()
        assert hasattr(cfg, "validation")
        assert isinstance(cfg.validation, ValidationConfig)

    def test_validation_defaults_in_train_config(self):
        cfg = _make_minimal_train_config()
        assert cfg.validation.enabled is False
        assert cfg.validation.data_path is None
        assert cfg.validation.interval_steps == 500

    def test_train_config_round_trip_with_validation(self):
        cfg = _make_minimal_train_config()
        cfg.validation.enabled = True
        cfg.validation.data_path = "/val/dataset.toml"
        cfg.validation.interval_steps = 250
        cfg.validation.num_steps = 20

        d = cfg.to_dict()
        cfg2 = TrainConfig.from_dict(d)

        assert cfg2.validation.enabled is True
        assert cfg2.validation.data_path == "/val/dataset.toml"
        assert cfg2.validation.interval_steps == 250
        assert cfg2.validation.num_steps == 20

    def test_train_config_with_explicit_validation_block(self):
        from trainer.config.schema import DataConfig, DatasetEntry
        cfg = TrainConfig(
            model=ModelConfig(architecture="wan", base_model_path="/fake/model"),
            network=NetworkConfig(rank=4, alpha=4.0),
            data=DataConfig(datasets=[DatasetEntry(path="/fake/data")]),
            validation=ValidationConfig(
                enabled=True,
                data_path="/val/data.toml",
                interval_steps=100,
                num_steps=5,
            ),
        )
        assert cfg.validation.enabled is True
        assert cfg.validation.interval_steps == 100
