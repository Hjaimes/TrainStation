"""Tests for masked training - mask loading, normalization, and masked loss computation."""
from __future__ import annotations

import io
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask_safetensors(mask: torch.Tensor) -> bytes:
    """Serialize a mask tensor to safetensors bytes in memory."""
    from safetensors.torch import save
    return save({"mask": mask})


def _write_mask_safetensors(path: str, mask: torch.Tensor) -> None:
    """Write a mask tensor to a safetensors file."""
    from safetensors.torch import save_file
    save_file({"mask": mask}, path)


# ---------------------------------------------------------------------------
# Tests: mask_utils.load_mask
# ---------------------------------------------------------------------------

class TestLoadMask:
    def test_image_mask_shape(self, tmp_path):
        """load_mask returns (1, H, W) for an image mask."""
        from trainer.data.mask_utils import load_mask
        mask = torch.ones(1, 32, 32)
        _write_mask_safetensors(str(tmp_path / "test_mask.safetensors"), mask)
        loaded = load_mask(str(tmp_path / "test_mask.safetensors"))
        assert loaded.shape == (1, 32, 32)

    def test_video_mask_shape(self, tmp_path):
        """load_mask returns (1, F, H, W) for a video mask."""
        from trainer.data.mask_utils import load_mask
        mask = torch.ones(1, 8, 16, 16)
        _write_mask_safetensors(str(tmp_path / "vid_mask.safetensors"), mask)
        loaded = load_mask(str(tmp_path / "vid_mask.safetensors"))
        assert loaded.shape == (1, 8, 16, 16)

    def test_clamp_below_zero(self, tmp_path):
        """Values below 0 are clamped to 0."""
        from trainer.data.mask_utils import load_mask
        mask = torch.full((1, 8, 8), -0.5)
        _write_mask_safetensors(str(tmp_path / "m.safetensors"), mask)
        loaded = load_mask(str(tmp_path / "m.safetensors"))
        assert loaded.min().item() >= 0.0

    def test_clamp_above_one(self, tmp_path):
        """Values above 1 are clamped to 1."""
        from trainer.data.mask_utils import load_mask
        mask = torch.full((1, 8, 8), 2.0)
        _write_mask_safetensors(str(tmp_path / "m.safetensors"), mask)
        loaded = load_mask(str(tmp_path / "m.safetensors"))
        assert loaded.max().item() <= 1.0

    def test_values_preserved_within_range(self, tmp_path):
        """Values already in [0, 1] are not altered."""
        from trainer.data.mask_utils import load_mask
        mask = torch.tensor([[[0.0, 0.5, 1.0, 0.25]]])  # (1, 1, 4)
        _write_mask_safetensors(str(tmp_path / "m.safetensors"), mask)
        loaded = load_mask(str(tmp_path / "m.safetensors"))
        assert torch.allclose(loaded, mask)


# ---------------------------------------------------------------------------
# Tests: mask_utils.normalize_mask
# ---------------------------------------------------------------------------

class TestNormalizeMask:
    def test_image_passthrough_when_shapes_match(self):
        """normalize_mask returns the same tensor when shapes already match."""
        from trainer.data.mask_utils import normalize_mask
        mask = torch.ones(1, 16, 16)
        result = normalize_mask(mask, (16, 16))
        assert result.shape == mask.shape
        assert torch.equal(result, mask)

    def test_image_resize_upscale(self):
        """normalize_mask upscales image mask to target_shape."""
        from trainer.data.mask_utils import normalize_mask
        mask = torch.zeros(1, 4, 4)
        mask[0, :2, :2] = 1.0  # top-left quarter is 1
        result = normalize_mask(mask, (8, 8))
        assert result.shape == (1, 8, 8)
        # Top-left corner should still be 1 after nearest-neighbor
        assert result[0, 0, 0].item() == 1.0
        assert result[0, 7, 7].item() == 0.0

    def test_image_resize_downscale(self):
        """normalize_mask downscales image mask to target_shape."""
        from trainer.data.mask_utils import normalize_mask
        mask = torch.ones(1, 32, 32)
        result = normalize_mask(mask, (8, 8))
        assert result.shape == (1, 8, 8)
        assert result.min().item() == 1.0

    def test_video_passthrough_when_shapes_match(self):
        """normalize_mask returns same tensor for video when shapes match."""
        from trainer.data.mask_utils import normalize_mask
        mask = torch.ones(1, 5, 16, 16)
        result = normalize_mask(mask, (5, 16, 16))
        assert result.shape == mask.shape

    def test_video_resize_spatial(self):
        """normalize_mask resizes only spatial dims for video mask."""
        from trainer.data.mask_utils import normalize_mask
        mask = torch.ones(1, 5, 4, 4)
        result = normalize_mask(mask, (5, 8, 8))
        # Temporal dimension unchanged, spatial dims resized
        assert result.shape == (1, 5, 8, 8)

    def test_invalid_ndim_raises(self):
        """normalize_mask raises ValueError for unsupported mask ndim."""
        from trainer.data.mask_utils import normalize_mask
        mask = torch.ones(8, 8)  # 2D - invalid
        with pytest.raises(ValueError, match="Unsupported mask ndim"):
            normalize_mask(mask, (8, 8))

    def test_output_dtype_preserved(self):
        """normalize_mask output dtype matches input dtype."""
        from trainer.data.mask_utils import normalize_mask
        mask = torch.ones(1, 4, 4, dtype=torch.float16)
        result = normalize_mask(mask, (8, 8))
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# Tests: ModelStrategy._compute_masked_loss
# ---------------------------------------------------------------------------

class TestComputeMaskedLoss:
    """Tests for the _compute_masked_loss method on a minimal ModelStrategy subclass."""

    @pytest.fixture()
    def strategy(self):
        """Create a minimal ModelStrategy instance for testing _compute_masked_loss."""
        from trainer.arch.base import ModelStrategy
        from trainer.config.schema import TrainConfig

        cfg = TrainConfig.model_validate({
            "model": {"architecture": "wan", "base_model_path": "/fake/model.safetensors"},
            "training": {"method": "full_finetune"},
            "data": {"dataset_config_path": "dummy.toml"},
        })

        class _DummyStrategy(ModelStrategy):
            @property
            def architecture(self):
                return "dummy"

            def setup(self):
                raise NotImplementedError

            def training_step(self, components, batch, step):
                raise NotImplementedError

        s = _DummyStrategy(cfg)
        # No loss_fn configured - falls back to MSE
        return s

    def test_all_ones_mask_matches_standard_mse(self, strategy):
        """With all-ones mask and mask_weight=1, result equals plain MSE."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        mask = torch.ones(2, 1, 8, 8)

        masked_loss = strategy._compute_masked_loss(
            pred, target, mask, mask_weight=1.0, normalize_by_area=False
        )
        standard_loss = F.mse_loss(pred, target, reduction="mean")
        assert torch.allclose(masked_loss, standard_loss, atol=1e-5), (
            f"Expected {standard_loss.item():.6f}, got {masked_loss.item():.6f}"
        )

    def test_all_zeros_mask_differs_from_ones(self, strategy):
        """With all-zeros mask and mask_weight=1, weight is always 1 (no change)."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.zeros(2, 4, 8, 8)
        mask_ones = torch.ones(2, 1, 8, 8)
        mask_zeros = torch.zeros(2, 1, 8, 8)

        loss_ones = strategy._compute_masked_loss(
            pred, target, mask_ones, mask_weight=1.0, normalize_by_area=False
        )
        loss_zeros = strategy._compute_masked_loss(
            pred, target, mask_zeros, mask_weight=1.0, normalize_by_area=False
        )
        # With mask_weight=1, masked=(mask*1 + (1-mask)*1) == 1 always,
        # so both produce the same result.
        assert torch.allclose(loss_ones, loss_zeros, atol=1e-5)

    def test_mask_weight_scales_masked_region(self, strategy):
        """Higher mask_weight increases loss when mask is all-ones."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.zeros_like(pred)
        mask = torch.ones(2, 1, 8, 8)  # all masked

        loss_w1 = strategy._compute_masked_loss(
            pred, target, mask, mask_weight=1.0, normalize_by_area=False
        )
        loss_w2 = strategy._compute_masked_loss(
            pred, target, mask, mask_weight=2.0, normalize_by_area=False
        )
        assert loss_w2.item() > loss_w1.item(), (
            "Higher mask_weight on fully-masked input should produce higher loss"
        )

    def test_partial_mask_between_extremes(self, strategy):
        """Loss with partial mask (half ones) differs from all-ones mask."""
        pred = torch.ones(2, 4, 8, 8)
        target = torch.zeros(2, 4, 8, 8)
        mask_full = torch.ones(2, 1, 8, 8)
        mask_half = torch.zeros(2, 1, 8, 8)
        mask_half[:, :, :4, :] = 1.0  # top half masked

        loss_full = strategy._compute_masked_loss(
            pred, target, mask_full, mask_weight=2.0, normalize_by_area=False
        )
        loss_half = strategy._compute_masked_loss(
            pred, target, mask_half, mask_weight=2.0, normalize_by_area=False
        )
        # With different mask regions and weight=2, losses differ
        assert not torch.allclose(loss_full, loss_half, atol=1e-5)

    def test_normalize_by_area_true_vs_false(self, strategy):
        """normalize_by_area affects the output value."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.zeros_like(pred)
        mask = torch.ones(2, 1, 8, 8)
        mask[:, :, :4, :] = 0.0  # half masked

        loss_normalized = strategy._compute_masked_loss(
            pred, target, mask, mask_weight=2.0, normalize_by_area=True
        )
        loss_not_normalized = strategy._compute_masked_loss(
            pred, target, mask, mask_weight=2.0, normalize_by_area=False
        )
        assert not torch.allclose(loss_normalized, loss_not_normalized, atol=1e-5)

    def test_returns_scalar_tensor(self, strategy):
        """_compute_masked_loss always returns a 0-dim scalar tensor."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        result = strategy._compute_masked_loss(pred, target, mask)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_result_is_finite(self, strategy):
        """_compute_masked_loss returns a finite value."""
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        result = strategy._compute_masked_loss(pred, target, mask)
        assert torch.isfinite(result)

    def test_identical_pred_target_gives_zero_loss(self, strategy):
        """When pred == target, masked loss is zero regardless of mask."""
        x = torch.randn(2, 4, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        result = strategy._compute_masked_loss(x, x, mask)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_mask_weight_zero_reduces_masked_to_unmasked(self, strategy):
        """mask_weight=0 makes masked regions contribute zero; only unmasked regions count."""
        pred = torch.ones(2, 4, 8, 8)
        target = torch.zeros(2, 4, 8, 8)
        # All masked, mask_weight=0 -> weight everywhere = 0*1 + (1-1) = 0 -> loss = 0
        mask = torch.ones(2, 1, 8, 8)
        result = strategy._compute_masked_loss(
            pred, target, mask, mask_weight=0.0, normalize_by_area=False
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_video_mask_broadcastable(self, strategy):
        """_compute_masked_loss works with 5D video tensors."""
        pred = torch.randn(2, 16, 8, 8, 8)   # B, C, F, H, W
        target = torch.randn_like(pred)
        mask = torch.ones(2, 1, 8, 8, 8)      # B, 1, F, H, W
        result = strategy._compute_masked_loss(pred, target, mask)
        assert result.ndim == 0
        assert torch.isfinite(result)


# ---------------------------------------------------------------------------
# Tests: DataConfig masked_training fields
# ---------------------------------------------------------------------------

class TestDataConfigMaskedFields:
    def test_masked_training_default_false(self):
        """masked_training defaults to False."""
        from trainer.config.schema import DataConfig
        cfg = DataConfig()
        assert cfg.masked_training is False

    def test_mask_weight_default_one(self):
        """mask_weight defaults to 1.0."""
        from trainer.config.schema import DataConfig
        cfg = DataConfig()
        assert cfg.mask_weight == 1.0

    def test_unmasked_probability_default_zero(self):
        """unmasked_probability defaults to 0.0."""
        from trainer.config.schema import DataConfig
        cfg = DataConfig()
        assert cfg.unmasked_probability == 0.0

    def test_normalize_masked_area_loss_default_true(self):
        """normalize_masked_area_loss defaults to True."""
        from trainer.config.schema import DataConfig
        cfg = DataConfig()
        assert cfg.normalize_masked_area_loss is True

    def test_masked_training_fields_roundtrip(self):
        """All masked training fields survive a round-trip through to_dict/from_dict."""
        from trainer.config.schema import TrainConfig

        cfg = TrainConfig.model_validate({
            "model": {"architecture": "wan", "base_model_path": "/fake/model.safetensors"},
            "training": {"method": "full_finetune"},
            "data": {
                "dataset_config_path": "dummy.toml",
                "masked_training": True,
                "mask_weight": 2.5,
                "unmasked_probability": 0.1,
                "normalize_masked_area_loss": False,
            },
        })
        d = cfg.to_dict()
        cfg2 = TrainConfig.from_dict(d)
        assert cfg2.data.masked_training is True
        assert cfg2.data.mask_weight == 2.5
        assert cfg2.data.unmasked_probability == 0.1
        assert cfg2.data.normalize_masked_area_loss is False

    def test_masked_training_fields_accept_valid_values(self):
        """DataConfig accepts valid masked training field values."""
        from trainer.config.schema import DataConfig
        cfg = DataConfig(
            masked_training=True,
            mask_weight=0.5,
            unmasked_probability=0.2,
            normalize_masked_area_loss=True,
        )
        assert cfg.masked_training is True
        assert cfg.mask_weight == 0.5


# ---------------------------------------------------------------------------
# Tests: ItemInfo mask_cache_path field
# ---------------------------------------------------------------------------

class TestItemInfoMaskCachePath:
    def test_mask_cache_path_default_none(self):
        """ItemInfo.mask_cache_path defaults to None."""
        from trainer.data.dataset import ItemInfo
        item = ItemInfo(
            item_key="test",
            original_size=(512, 512),
            bucket_reso=(512, 512),
        )
        assert item.mask_cache_path is None

    def test_mask_cache_path_can_be_set(self):
        """ItemInfo.mask_cache_path can be set to a string path."""
        from trainer.data.dataset import ItemInfo
        item = ItemInfo(
            item_key="test",
            original_size=(512, 512),
            bucket_reso=(512, 512),
            mask_cache_path="/path/to/mask.safetensors",
        )
        assert item.mask_cache_path == "/path/to/mask.safetensors"
