"""Tests for training method implementations: LoRA and FullFinetune.

Covers Task 16 (per-component LR overrides) and general method correctness.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trainer.training.methods import (
    FullFinetuneMethod,
    LoRAMethod,
    TrainingMethodResult,
    create_training_method,
)
from trainer.config.schema import (
    TrainConfig,
    ModelConfig,
    TrainingConfig,
    OptimizerConfig,
    NetworkConfig,
    DataConfig,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _model_config() -> ModelConfig:
    return ModelConfig(architecture="wan", base_model_path="/fake/model")


def _data_config() -> DataConfig:
    return DataConfig(datasets=[{"path": "/fake/data"}])


def _make_full_finetune_config(
    component_lr_overrides: dict[str, float] | None = None,
) -> TrainConfig:
    return TrainConfig(
        model=_model_config(),
        training=TrainingConfig(method="full_finetune"),
        optimizer=OptimizerConfig(
            learning_rate=1e-4,
            component_lr_overrides=component_lr_overrides,
        ),
        data=_data_config(),
    )


# ---------------------------------------------------------------------------
# Small test model with named sub-components
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        return self.norm(self.proj(x))


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 16)

    def forward(self, x):
        return self.proj(x)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# TestFullFinetuneMethod
# ---------------------------------------------------------------------------

class TestFullFinetuneMethod:
    def test_all_params_trainable(self):
        model = _TinyModel()
        method = FullFinetuneMethod(_make_full_finetune_config())
        result = method.prepare(model, arch="wan", learning_rate=1e-4)

        assert isinstance(result, TrainingMethodResult)
        assert result.network is None
        flat = result.get_trainable_params_flat()
        assert len(flat) == len(list(model.parameters()))
        for p in flat:
            assert p.requires_grad

    def test_single_param_group_without_overrides(self):
        model = _TinyModel()
        method = FullFinetuneMethod(_make_full_finetune_config())
        result = method.prepare(model, arch="wan", learning_rate=2e-4)

        assert len(result.trainable_params) == 1
        assert result.trainable_params[0]["lr"] == pytest.approx(2e-4)

    def test_component_lr_overrides_creates_multiple_groups(self):
        model = _TinyModel()
        overrides = {"encoder": 1e-5, "decoder": 5e-5}
        method = FullFinetuneMethod(_make_full_finetune_config(component_lr_overrides=overrides))
        result = method.prepare(model, arch="wan", learning_rate=1e-4)

        # Expect: encoder group + decoder group + (optionally) remainder group
        # All encoder params match "encoder", all decoder params match "decoder"
        lrs = {g["lr"] for g in result.trainable_params}
        assert 1e-5 in lrs
        assert 5e-5 in lrs

    def test_component_lr_encoder_params_isolated(self):
        model = _TinyModel()
        overrides = {"encoder": 1e-5}
        method = FullFinetuneMethod(_make_full_finetune_config(component_lr_overrides=overrides))
        result = method.prepare(model, arch="wan", learning_rate=1e-4)

        # Locate the encoder group
        encoder_group = next(g for g in result.trainable_params if g["lr"] == pytest.approx(1e-5))
        encoder_param_ids = {id(p) for g in result.trainable_params
                             if g["lr"] == pytest.approx(1e-5) for p in g["params"]}

        # All encoder params should be in that group
        expected_encoder_ids = {id(p) for p in model.encoder.parameters()}
        assert expected_encoder_ids == encoder_param_ids

    def test_component_lr_remaining_params_get_base_lr(self):
        model = _TinyModel()
        overrides = {"encoder": 1e-5}
        method = FullFinetuneMethod(_make_full_finetune_config(component_lr_overrides=overrides))
        result = method.prepare(model, arch="wan", learning_rate=1e-4)

        # Remaining (decoder) params should be at base lr
        base_group = next(g for g in result.trainable_params if g["lr"] == pytest.approx(1e-4))
        base_ids = {id(p) for p in base_group["params"]}
        decoder_ids = {id(p) for p in model.decoder.parameters()}
        assert decoder_ids == base_ids

    def test_no_duplicate_params_across_groups(self):
        """A parameter must appear in exactly one group."""
        model = _TinyModel()
        overrides = {"encoder": 1e-5, "decoder": 5e-5}
        method = FullFinetuneMethod(_make_full_finetune_config(component_lr_overrides=overrides))
        result = method.prepare(model, arch="wan", learning_rate=1e-4)

        all_ids: list[int] = []
        for g in result.trainable_params:
            all_ids.extend(id(p) for p in g["params"])
        # No duplicates
        assert len(all_ids) == len(set(all_ids))

    def test_total_params_unchanged_with_overrides(self):
        """All parameters must be covered regardless of overrides."""
        model = _TinyModel()
        overrides = {"encoder": 1e-5, "decoder": 5e-5}
        method = FullFinetuneMethod(_make_full_finetune_config(component_lr_overrides=overrides))
        result = method.prepare(model, arch="wan", learning_rate=1e-4)

        total = sum(sum(p.numel() for p in g["params"]) for g in result.trainable_params)
        expected = sum(p.numel() for p in model.parameters())
        assert total == expected

    def test_component_lr_no_remainder_when_fully_covered(self):
        """When every param is matched, there should be no remainder group."""
        model = _TinyModel()
        # encoder and decoder together cover all parameters
        overrides = {"encoder": 1e-5, "decoder": 5e-5}
        method = FullFinetuneMethod(_make_full_finetune_config(component_lr_overrides=overrides))
        result = method.prepare(model, arch="wan", learning_rate=1e-4)

        # No group at base lr should exist (or if it does it should have 0 params)
        base_groups = [g for g in result.trainable_params if g["lr"] == pytest.approx(1e-4)]
        assert all(len(g["params"]) == 0 for g in base_groups)


# ---------------------------------------------------------------------------
# TestCreateTrainingMethod
# ---------------------------------------------------------------------------

class TestCreateTrainingMethod:
    def test_full_finetune(self):
        cfg = _make_full_finetune_config()
        method = create_training_method(cfg)
        assert isinstance(method, FullFinetuneMethod)

    def test_lora(self):
        cfg = TrainConfig(
            model=_model_config(),
            training=TrainingConfig(method="lora"),
            optimizer=OptimizerConfig(),
            data=_data_config(),
            network=NetworkConfig(rank=4, alpha=4.0),
        )
        method = create_training_method(cfg)
        assert isinstance(method, LoRAMethod)

    def test_unknown_method_raises(self):
        cfg = TrainConfig(
            model=_model_config(),
            training=TrainingConfig(method="full_finetune"),
            optimizer=OptimizerConfig(),
            data=_data_config(),
        )
        # Bypass pydantic by patching after construction
        cfg.training.method = "unknown_method"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown training method"):
            create_training_method(cfg)


# ---------------------------------------------------------------------------
# TestTrainingMethodResult
# ---------------------------------------------------------------------------

class TestTrainingMethodResult:
    def test_get_trainable_params_flat(self):
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(8))
        result = TrainingMethodResult(
            trainable_params=[{"params": [p1]}, {"params": [p2]}],
            network=None,
            save_fn=lambda *a: None,
            cleanup_fn=lambda: None,
        )
        flat = result.get_trainable_params_flat()
        assert flat == [p1, p2]

    def test_get_trainable_params_flat_cached(self):
        p1 = nn.Parameter(torch.randn(4))
        result = TrainingMethodResult(
            trainable_params=[{"params": [p1]}],
            network=None,
            save_fn=lambda *a: None,
            cleanup_fn=lambda: None,
        )
        flat1 = result.get_trainable_params_flat()
        flat2 = result.get_trainable_params_flat()
        assert flat1 is flat2  # Cache was used
