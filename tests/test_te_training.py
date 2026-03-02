"""Tests for text encoder training infrastructure (Tasks 26-28).

Covers:
- Config fields for TE training
- ItemInfo.caption_path field
- Caption loading in BucketBatchManager
- Base class encode_text_for_training / _setup_text_encoder_training
- LoRAMethod.prepare() text_encoders parameter
- FullFinetuneMethod.prepare() text_encoders parameter
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import fields as dc_fields

import pytest
import torch
import torch.nn as nn

from trainer.config.schema import (
    DataConfig,
    ModelConfig,
    NetworkConfig,
    OptimizerConfig,
    TrainConfig,
    TrainingConfig,
)
from trainer.data.dataset import BucketBatchManager, ItemInfo
from trainer.arch.base import ModelComponents, ModelStrategy
from trainer.training.methods import (
    FullFinetuneMethod,
    LoRAMethod,
    TrainingMethodResult,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _model_config() -> ModelConfig:
    return ModelConfig(architecture="wan", base_model_path="/fake/model")


def _data_config() -> DataConfig:
    return DataConfig(datasets=[{"path": "/fake/data"}])


def _lora_config(**training_kwargs) -> TrainConfig:
    return TrainConfig(
        model=_model_config(),
        training=TrainingConfig(method="lora", **training_kwargs),
        optimizer=OptimizerConfig(learning_rate=1e-4),
        data=_data_config(),
        network=NetworkConfig(rank=4, alpha=4.0),
    )


def _full_finetune_config(**training_kwargs) -> TrainConfig:
    return TrainConfig(
        model=_model_config(),
        training=TrainingConfig(method="full_finetune", **training_kwargs),
        optimizer=OptimizerConfig(learning_rate=1e-4),
        data=_data_config(),
    )


# ---------------------------------------------------------------------------
# Task 26: Config fields
# ---------------------------------------------------------------------------

class TestTETrainingConfig:
    def test_train_text_encoder_default_false(self):
        cfg = TrainingConfig()
        assert cfg.train_text_encoder is False

    def test_text_encoder_lr_default_none(self):
        cfg = TrainingConfig()
        assert cfg.text_encoder_lr is None

    def test_text_encoder_gradient_checkpointing_default_true(self):
        cfg = TrainingConfig()
        assert cfg.text_encoder_gradient_checkpointing is True

    def test_train_text_encoder_set_true(self):
        cfg = TrainingConfig(train_text_encoder=True)
        assert cfg.train_text_encoder is True

    def test_text_encoder_lr_set(self):
        cfg = TrainingConfig(text_encoder_lr=5e-5)
        assert cfg.text_encoder_lr == pytest.approx(5e-5)

    def test_text_encoder_gradient_checkpointing_set_false(self):
        cfg = TrainingConfig(text_encoder_gradient_checkpointing=False)
        assert cfg.text_encoder_gradient_checkpointing is False

    def test_config_round_trip(self):
        """Config serializes and deserializes with TE training fields."""
        cfg = TrainConfig(
            model=_model_config(),
            training=TrainingConfig(
                method="full_finetune",
                train_text_encoder=True,
                text_encoder_lr=1e-5,
                text_encoder_gradient_checkpointing=False,
            ),
            data=_data_config(),
        )
        d = cfg.to_dict()
        restored = TrainConfig.from_dict(d)
        assert restored.training.train_text_encoder is True
        assert restored.training.text_encoder_lr == pytest.approx(1e-5)
        assert restored.training.text_encoder_gradient_checkpointing is False

    def test_full_config_with_te_training_validates(self):
        """Full config with TE training fields passes cross-validation."""
        cfg = _lora_config(train_text_encoder=True, text_encoder_lr=5e-5)
        assert cfg.training.train_text_encoder is True
        assert cfg.training.text_encoder_lr == pytest.approx(5e-5)


# ---------------------------------------------------------------------------
# Task 26: ItemInfo.caption_path
# ---------------------------------------------------------------------------

class TestItemInfoCaptionPath:
    def test_caption_path_field_exists(self):
        field_names = {f.name for f in dc_fields(ItemInfo)}
        assert "caption_path" in field_names

    def test_caption_path_default_none(self):
        item = ItemInfo(
            item_key="test",
            original_size=(512, 512),
            bucket_reso=(512, 512),
        )
        assert item.caption_path is None

    def test_caption_path_set(self):
        item = ItemInfo(
            item_key="test",
            original_size=(512, 512),
            bucket_reso=(512, 512),
            caption_path="/some/path.txt",
        )
        assert item.caption_path == "/some/path.txt"


# ---------------------------------------------------------------------------
# Task 26: Caption loading in BucketBatchManager
# ---------------------------------------------------------------------------

class TestCaptionLoading:
    def _make_cache_files(self, tmp_dir: str, item_key: str, arch: str = "wan"):
        """Create minimal safetensors files for testing."""
        from safetensors.torch import save_file

        # Latent cache
        latent_path = os.path.join(tmp_dir, f"{item_key}_0512x0512_{arch}.safetensors")
        save_file({"latents_0512x0512_bf16": torch.randn(4, 8, 8)}, latent_path)

        # TE cache
        te_path = os.path.join(tmp_dir, f"{item_key}_{arch}_te.safetensors")
        save_file({"prompt_embeds_bf16": torch.randn(77, 768)}, te_path)

        return latent_path, te_path

    def test_captions_loaded_when_caption_path_exists(self):
        """When items have caption_path pointing to existing files, captions are in the batch."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            latent_path, te_path = self._make_cache_files(tmp_dir, "img001")

            # Write a caption file
            caption_path = os.path.join(tmp_dir, "img001.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write("a photo of a cat")

            items = [ItemInfo(
                item_key="img001",
                original_size=(512, 512),
                bucket_reso=(512, 512),
                latent_cache_path=latent_path,
                text_encoder_output_cache_path=te_path,
                caption_path=caption_path,
            )]

            mgr = BucketBatchManager(
                bucketed_items={(512, 512): items},
                batch_size=1,
            )
            batch = mgr[0]
            assert "captions" in batch
            assert batch["captions"] == ["a photo of a cat"]

    def test_no_captions_key_when_no_caption_path(self):
        """When items have no caption_path, batch should not contain 'captions'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            latent_path, te_path = self._make_cache_files(tmp_dir, "img002")

            items = [ItemInfo(
                item_key="img002",
                original_size=(512, 512),
                bucket_reso=(512, 512),
                latent_cache_path=latent_path,
                text_encoder_output_cache_path=te_path,
            )]

            mgr = BucketBatchManager(
                bucketed_items={(512, 512): items},
                batch_size=1,
            )
            batch = mgr[0]
            assert "captions" not in batch

    def test_empty_caption_file_not_included(self):
        """An empty caption file results in an empty string, which should not trigger inclusion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            latent_path, te_path = self._make_cache_files(tmp_dir, "img003")

            caption_path = os.path.join(tmp_dir, "img003.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write("")  # empty

            items = [ItemInfo(
                item_key="img003",
                original_size=(512, 512),
                bucket_reso=(512, 512),
                latent_cache_path=latent_path,
                text_encoder_output_cache_path=te_path,
                caption_path=caption_path,
            )]

            mgr = BucketBatchManager(
                bucketed_items={(512, 512): items},
                batch_size=1,
            )
            batch = mgr[0]
            # Empty captions means 'any(c for c in captions)' is False
            assert "captions" not in batch

    def test_mixed_caption_batch(self):
        """When some items have captions and some don't, captions list has empty strings for missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            lp1, tp1 = self._make_cache_files(tmp_dir, "imgA")
            lp2, tp2 = self._make_cache_files(tmp_dir, "imgB")

            caption_path = os.path.join(tmp_dir, "imgA.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write("a dog running")

            items = [
                ItemInfo(
                    item_key="imgA",
                    original_size=(512, 512),
                    bucket_reso=(512, 512),
                    latent_cache_path=lp1,
                    text_encoder_output_cache_path=tp1,
                    caption_path=caption_path,
                ),
                ItemInfo(
                    item_key="imgB",
                    original_size=(512, 512),
                    bucket_reso=(512, 512),
                    latent_cache_path=lp2,
                    text_encoder_output_cache_path=tp2,
                    # No caption_path
                ),
            ]

            mgr = BucketBatchManager(
                bucketed_items={(512, 512): items},
                batch_size=2,
            )
            batch = mgr[0]
            assert "captions" in batch
            assert batch["captions"] == ["a dog running", ""]


# ---------------------------------------------------------------------------
# Task 27: Base class methods
# ---------------------------------------------------------------------------

class TestBaseClassTEMethods:
    def test_encode_text_for_training_raises_not_implemented(self):
        """Base class encode_text_for_training raises NotImplementedError."""

        class _StubStrategy(ModelStrategy):
            @property
            def architecture(self) -> str:
                return "stub_arch"

        cfg = _lora_config()
        strategy = _StubStrategy(cfg)
        components = ModelComponents(model=nn.Linear(4, 4))
        with pytest.raises(NotImplementedError, match="Text encoder training not implemented for stub_arch"):
            strategy.encode_text_for_training(components, ["hello"], torch.device("cpu"))

    def test_setup_text_encoder_training_caches_values(self):
        """_setup_text_encoder_training stores config values on the strategy."""
        cfg = _lora_config(
            train_text_encoder=True,
            text_encoder_lr=2e-5,
            text_encoder_gradient_checkpointing=False,
        )
        strategy = ModelStrategy(cfg)
        strategy._setup_text_encoder_training(cfg)

        assert strategy._train_text_encoder is True
        assert strategy._text_encoder_lr == pytest.approx(2e-5)
        assert strategy._te_gradient_checkpointing is False

    def test_setup_text_encoder_training_defaults(self):
        """_setup_text_encoder_training with default config values."""
        cfg = _lora_config()
        strategy = ModelStrategy(cfg)
        strategy._setup_text_encoder_training(cfg)

        assert strategy._train_text_encoder is False
        assert strategy._text_encoder_lr is None
        assert strategy._te_gradient_checkpointing is True

    def test_default_class_attribute(self):
        """ModelStrategy._train_text_encoder defaults to False."""
        cfg = _lora_config()
        strategy = ModelStrategy(cfg)
        assert strategy._train_text_encoder is False


# ---------------------------------------------------------------------------
# Task 28: LoRAMethod text_encoders parameter
# ---------------------------------------------------------------------------

class TestLoRAMethodTextEncoders:
    def test_prepare_accepts_text_encoders_param(self):
        """LoRAMethod.prepare() accepts text_encoders without error."""
        cfg = _lora_config()
        method = LoRAMethod(cfg)

        # Need a model with WanAttentionBlock to satisfy arch config
        from tests.test_methods import _TinyModel
        model = _make_wan_compatible_model()
        result = method.prepare(
            model=model, arch="wan", learning_rate=1e-4, text_encoders=None,
        )
        assert isinstance(result, TrainingMethodResult)

    def test_prepare_backward_compatible_without_text_encoders(self):
        """LoRAMethod.prepare() works without text_encoders (backward compat)."""
        cfg = _lora_config()
        method = LoRAMethod(cfg)
        model = _make_wan_compatible_model()
        # Call without text_encoders argument at all
        result = method.prepare(model=model, arch="wan", learning_rate=1e-4)
        assert isinstance(result, TrainingMethodResult)


class TestFullFinetuneMethodTextEncoders:
    def test_prepare_accepts_text_encoders_param(self):
        """FullFinetuneMethod.prepare() accepts text_encoders without error."""
        cfg = _full_finetune_config()
        method = FullFinetuneMethod(cfg)
        model = nn.Linear(32, 32)
        result = method.prepare(
            model=model, arch="wan", learning_rate=1e-4, text_encoders=None,
        )
        assert isinstance(result, TrainingMethodResult)

    def test_prepare_backward_compatible_without_text_encoders(self):
        """FullFinetuneMethod.prepare() works without text_encoders (backward compat)."""
        cfg = _full_finetune_config()
        method = FullFinetuneMethod(cfg)
        model = nn.Linear(32, 32)
        result = method.prepare(model=model, arch="wan", learning_rate=1e-4)
        assert isinstance(result, TrainingMethodResult)

    def test_te_params_added_when_train_text_encoder(self):
        """When train_text_encoder=True, TE params are included in trainable_params."""
        cfg = _full_finetune_config(train_text_encoder=True, text_encoder_lr=5e-5)
        method = FullFinetuneMethod(cfg)
        model = nn.Linear(32, 32)
        te = nn.Linear(64, 64)

        result = method.prepare(
            model=model, arch="wan", learning_rate=1e-4, text_encoders=[te],
        )

        # Should have multiple param groups: model + TE
        total_params = sum(sum(p.numel() for p in g["params"]) for g in result.trainable_params)
        model_params = sum(p.numel() for p in model.parameters())
        te_params = sum(p.numel() for p in te.parameters())
        assert total_params == model_params + te_params

    def test_te_lr_uses_separate_lr(self):
        """TE param group should use text_encoder_lr when specified."""
        cfg = _full_finetune_config(train_text_encoder=True, text_encoder_lr=5e-5)
        method = FullFinetuneMethod(cfg)
        model = nn.Linear(32, 32)
        te = nn.Linear(64, 64)

        result = method.prepare(
            model=model, arch="wan", learning_rate=1e-4, text_encoders=[te],
        )

        # Last group should be the TE group with lr=5e-5
        lrs = {g["lr"] for g in result.trainable_params}
        assert 5e-5 in lrs

    def test_te_lr_falls_back_to_base_lr(self):
        """When text_encoder_lr is None, TE uses the base learning_rate."""
        cfg = _full_finetune_config(train_text_encoder=True)
        method = FullFinetuneMethod(cfg)
        model = nn.Linear(32, 32)
        te = nn.Linear(64, 64)

        result = method.prepare(
            model=model, arch="wan", learning_rate=1e-4, text_encoders=[te],
        )

        # All groups should be at 1e-4
        for g in result.trainable_params:
            assert g["lr"] == pytest.approx(1e-4)

    def test_no_te_params_when_disabled(self):
        """When train_text_encoder=False, TE params are not included."""
        cfg = _full_finetune_config(train_text_encoder=False)
        method = FullFinetuneMethod(cfg)
        model = nn.Linear(32, 32)
        te = nn.Linear(64, 64)

        result = method.prepare(
            model=model, arch="wan", learning_rate=1e-4, text_encoders=[te],
        )

        total_params = sum(sum(p.numel() for p in g["params"]) for g in result.trainable_params)
        model_params = sum(p.numel() for p in model.parameters())
        assert total_params == model_params


# ---------------------------------------------------------------------------
# Task 28: ArchNetworkConfig te_target_modules field
# ---------------------------------------------------------------------------

class TestArchConfigTEField:
    def test_te_target_modules_in_typed_dict(self):
        """ArchNetworkConfig TypedDict accepts te_target_modules."""
        from trainer.networks.arch_configs import ArchNetworkConfig
        config: ArchNetworkConfig = {
            "target_modules": ["SomeBlock"],
            "te_target_modules": ["T5EncoderLayer"],
        }
        assert config["te_target_modules"] == ["T5EncoderLayer"]


# ---------------------------------------------------------------------------
# Helper: create a model compatible with "wan" arch_config target_modules
# ---------------------------------------------------------------------------

class WanAttentionBlock(nn.Module):
    """Minimal mock of WanAttentionBlock for testing LoRA application."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32)
        self.k_proj = nn.Linear(32, 32)
        self.v_proj = nn.Linear(32, 32)
        self.out_proj = nn.Linear(32, 32)

    def forward(self, x):
        return self.out_proj(self.v_proj(x))


class _WanModel(nn.Module):
    """Minimal model with WanAttentionBlock children for LoRA testing."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([WanAttentionBlock() for _ in range(2)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _make_wan_compatible_model() -> nn.Module:
    return _WanModel()
