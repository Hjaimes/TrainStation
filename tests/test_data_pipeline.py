"""
Tests for the data pipeline: dataset, TOML config, loader, caching.

Uses synthetic .safetensors cache files to validate the full pipeline
without requiring real model outputs.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from trainer.data.dataset import (
    ItemInfo,
    BucketBatchManager,
    CachedDataset,
    CachedDatasetGroup,
)
from trainer.data.loader import check_cache_exists


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_latent_cache(
    directory: str,
    name: str,
    w: int,
    h: int,
    arch: str,
    frame_count: int = 1,
    frame_pos: int = 0,
) -> str:
    """Create a synthetic latent cache .safetensors file and return its path."""
    if frame_count > 1:
        # Video: {name}_{pos:05d}-{frames:03d}_{W:04d}x{H:04d}_{arch}.safetensors
        filename = f"{name}_{frame_pos:05d}-{frame_count:03d}_{w:04d}x{h:04d}_{arch}.safetensors"
    else:
        # Image: {name}_{W:04d}x{H:04d}_{arch}.safetensors
        filename = f"{name}_{w:04d}x{h:04d}_{arch}.safetensors"

    filepath = os.path.join(directory, filename)

    # Create synthetic tensors matching Musubi's cache key format.
    # Latent key: latents_{FxHxW}_{dtype} e.g. "latents_1x64x64_bfloat16"
    lat_h, lat_w = h // 8, w // 8
    tensors: dict[str, torch.Tensor] = {
        f"latents_{frame_count}x{lat_h}x{lat_w}_bfloat16": torch.randn(
            16, frame_count, lat_h, lat_w, dtype=torch.bfloat16
        ),
    }
    save_file(tensors, filepath)
    return filepath


def _create_te_cache(
    directory: str,
    name: str,
    arch: str,
    seq_len: int = 512,
) -> str:
    """Create a synthetic text-encoder cache .safetensors file and return its path."""
    filename = f"{name}_{arch}_te.safetensors"
    filepath = os.path.join(directory, filename)

    tensors: dict[str, torch.Tensor] = {
        "t5_bfloat16": torch.randn(seq_len, 4096, dtype=torch.bfloat16),
        "t5_mask": torch.ones(seq_len, dtype=torch.bfloat16),
    }
    save_file(tensors, filepath)
    return filepath


def _create_cache_pair(
    directory: str,
    name: str,
    w: int,
    h: int,
    arch: str = "wan",
    frame_count: int = 1,
    frame_pos: int = 0,
    seq_len: int = 77,
) -> tuple[str, str]:
    """Create both latent and TE cache files. Returns (latent_path, te_path)."""
    lat_path = _create_latent_cache(directory, name, w, h, arch, frame_count, frame_pos)
    te_path = _create_te_cache(directory, name, arch, seq_len)
    return lat_path, te_path


# ---------------------------------------------------------------------------
# ItemInfo
# ---------------------------------------------------------------------------

class TestItemInfo:
    def test_basic_creation(self):
        item = ItemInfo(
            item_key="test_image",
            original_size=(512, 512),
            bucket_reso=(512, 512),
        )
        assert item.item_key == "test_image"
        assert item.frame_count == 1

    def test_default_loss_weight(self):
        """loss_weight defaults to 1.0 (no-op multiplier)."""
        item = ItemInfo(
            item_key="test_image",
            original_size=(512, 512),
            bucket_reso=(512, 512),
        )
        assert item.loss_weight == 1.0

    def test_custom_loss_weight(self):
        """loss_weight can be set to any float."""
        item = ItemInfo(
            item_key="test_image",
            original_size=(512, 512),
            bucket_reso=(512, 512),
            loss_weight=2.5,
        )
        assert item.loss_weight == 2.5

    def test_video_item(self):
        item = ItemInfo(
            item_key="test_video",
            original_size=(960, 544),
            bucket_reso=(960, 544, 25),
            frame_count=25,
        )
        assert item.frame_count == 25
        assert len(item.bucket_reso) == 3

    def test_repr(self):
        item = ItemInfo(
            item_key="img01",
            original_size=(256, 256),
            bucket_reso=(256, 256),
        )
        r = repr(item)
        assert "img01" in r
        assert "(256, 256)" in r


# ---------------------------------------------------------------------------
# Filename parsing (CachedDataset helpers)
# ---------------------------------------------------------------------------

class TestFilenameParsing:
    def setup_method(self):
        self.ds = CachedDataset(
            cache_directory="/tmp/fake",
            architecture="wan",
            batch_size=1,
        )

    def test_parse_image_latent(self):
        """Image: my_image_0512x0512_wan → (my_image, (512, 512), 1)"""
        result = self.ds._parse_latent_filename("my_image_0512x0512_wan")
        assert result is not None
        key, size, frames = result
        assert key == "my_image"
        assert size == (512, 512)
        assert frames == 1

    def test_parse_video_latent(self):
        """Video: clip_00000-025_0960x0544_wan → (clip, (960, 544), 25)"""
        result = self.ds._parse_latent_filename("clip_00000-025_0960x0544_wan")
        assert result is not None
        key, size, frames = result
        assert key == "clip"
        assert size == (960, 544)
        assert frames == 25

    def test_parse_compound_name(self):
        """Names with underscores: my_cool_image_0512x0512_wan"""
        result = self.ds._parse_latent_filename("my_cool_image_0512x0512_wan")
        assert result is not None
        key, size, frames = result
        assert key == "my_cool_image"
        assert size == (512, 512)
        assert frames == 1

    def test_parse_rejects_wrong_arch(self):
        """Should reject if arch suffix doesn't match."""
        result = self.ds._parse_latent_filename("my_image_0512x0512_flux")
        assert result is None

    def test_parse_rejects_no_size(self):
        """Should reject if no valid size token."""
        result = self.ds._parse_latent_filename("my_image_wan")
        assert result is None

    def test_parse_rejects_te_files(self):
        """TE cache files end with _wan_te, the stem without .safetensors
        would be my_image_wan_te - parser sees _te as arch suffix != wan."""
        # The glob pattern *_wan.safetensors won't match *_wan_te.safetensors,
        # so this case doesn't arise in practice. But verify parsing rejects it.
        result = self.ds._parse_latent_filename("my_image_wan_te")
        assert result is None

    def test_derive_te_cache_path(self):
        path = self.ds._derive_te_cache_path("my_image", is_video=False)
        expected = os.path.join("/tmp/fake", "my_image_wan_te.safetensors")
        assert path == expected


# ---------------------------------------------------------------------------
# BucketBatchManager
# ---------------------------------------------------------------------------

class TestBucketBatchManager:
    def test_basic_batching(self):
        items = [
            ItemInfo("img1", (512, 512), (512, 512)),
            ItemInfo("img2", (512, 512), (512, 512)),
            ItemInfo("img3", (512, 512), (512, 512)),
        ]
        bucketed = {(512, 512): items}
        mgr = BucketBatchManager(bucketed, batch_size=2)
        # 3 items / batch_size 2 = 2 batches (ceil)
        assert len(mgr) == 2

    def test_multiple_buckets(self):
        items_a = [ItemInfo(f"a{i}", (512, 512), (512, 512)) for i in range(4)]
        items_b = [ItemInfo(f"b{i}", (768, 768), (768, 768)) for i in range(3)]
        bucketed = {(512, 512): items_a, (768, 768): items_b}
        mgr = BucketBatchManager(bucketed, batch_size=2)
        # 4/2=2 batches from A, 3/2=2 batches from B → 4 total
        assert len(mgr) == 4

    def test_shuffle_changes_order(self):
        items = [ItemInfo(f"img{i}", (512, 512), (512, 512)) for i in range(10)]
        bucketed = {(512, 512): items}
        mgr = BucketBatchManager(bucketed, batch_size=2)
        order_before = list(mgr.bucket_batch_indices)
        mgr.shuffle(seed=42)
        order_after = list(mgr.bucket_batch_indices)
        # With 5 batches and seed=42, order should change
        assert order_before != order_after

    def test_shuffle_deterministic(self):
        """Two separate managers with same data+seed produce same batch order."""
        items1 = [ItemInfo(f"img{i}", (512, 512), (512, 512)) for i in range(10)]
        items2 = [ItemInfo(f"img{i}", (512, 512), (512, 512)) for i in range(10)]
        mgr1 = BucketBatchManager({(512, 512): items1}, batch_size=2)
        mgr2 = BucketBatchManager({(512, 512): items2}, batch_size=2)
        mgr1.shuffle(seed=123)
        mgr2.shuffle(seed=123)
        assert mgr1.bucket_batch_indices == mgr2.bucket_batch_indices

    def test_getitem_dataset_weight_default(self, tmp_path):
        """Batch always contains a 'dataset_weight' tensor with default 1.0."""
        # Need real cache files because __getitem__ loads safetensors
        _create_cache_pair(str(tmp_path), "img1", 512, 512)
        _create_cache_pair(str(tmp_path), "img2", 512, 512)

        ds = CachedDataset(str(tmp_path), "wan", batch_size=2)
        ds.prepare_for_training()
        batch = ds[0]

        assert "dataset_weight" in batch
        dw = batch["dataset_weight"]
        assert isinstance(dw, torch.Tensor)
        assert dw.shape == (2,)
        assert torch.all(dw == 1.0)

    def test_getitem_dataset_weight_custom(self, tmp_path):
        """Batch 'dataset_weight' tensor reflects the CachedDataset loss_weight."""
        _create_cache_pair(str(tmp_path), "img1", 512, 512)
        _create_cache_pair(str(tmp_path), "img2", 512, 512)

        ds = CachedDataset(str(tmp_path), "wan", batch_size=2, loss_weight=2.0)
        ds.prepare_for_training()
        batch = ds[0]

        dw = batch["dataset_weight"]
        assert dw.shape == (2,)
        assert torch.all(dw == 2.0)


# ---------------------------------------------------------------------------
# CachedDataset (with real files)
# ---------------------------------------------------------------------------

class TestCachedDataset:
    def test_prepare_discovers_files(self, tmp_path):
        """Create synthetic cache files and verify prepare_for_training finds them."""
        _create_cache_pair(str(tmp_path), "img1", 512, 512)
        _create_cache_pair(str(tmp_path), "img2", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
        )
        ds.prepare_for_training()
        assert ds.num_train_items == 2
        assert len(ds) == 2  # 2 items / batch_size 1 = 2 batches

    def test_prepare_with_repeats(self, tmp_path):
        _create_cache_pair(str(tmp_path), "img1", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
            num_repeats=3,
        )
        ds.prepare_for_training()
        assert ds.num_train_items == 1  # 1 unique item
        assert len(ds) == 3  # 3 repeats / batch_size 1 = 3 batches

    def test_prepare_no_files_raises(self, tmp_path):
        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
        )
        with pytest.raises(FileNotFoundError, match="No latent cache files"):
            ds.prepare_for_training()

    def test_prepare_missing_te_skips(self, tmp_path):
        """If latent exists but TE cache is missing, item is skipped."""
        _create_latent_cache(str(tmp_path), "lonely", 512, 512, "wan")
        # Don't create TE cache
        # But we need at least one valid pair to avoid the "none could be paired" error
        _create_cache_pair(str(tmp_path), "valid", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
        )
        ds.prepare_for_training()
        assert ds.num_train_items == 1  # Only "valid", not "lonely"

    def test_getitem_returns_batch_dict(self, tmp_path):
        """Verify __getitem__ returns correctly structured batch dict."""
        _create_cache_pair(str(tmp_path), "img1", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
        )
        ds.prepare_for_training()
        batch = ds[0]

        assert isinstance(batch, dict)
        assert "latents" in batch
        assert "t5" in batch
        assert "t5_mask" in batch
        assert "timesteps" in batch
        assert batch["timesteps"] is None

    def test_getitem_key_stripping(self, tmp_path):
        """Verify key stripping: latents_1x64x64_bfloat16 → latents, t5_bfloat16 → t5."""
        _create_cache_pair(str(tmp_path), "img1", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
        )
        ds.prepare_for_training()
        batch = ds[0]

        # "latents_1x64x64_bfloat16" → strip dtype → "latents_1x64x64" → strip FxHxW → "latents"
        assert "latents" in batch
        assert isinstance(batch["latents"], torch.Tensor)

        # "t5_bfloat16" → strip dtype → "t5"
        assert "t5" in batch

        # "t5_mask" → kept as-is (mask key)
        assert "t5_mask" in batch

    def test_getitem_batched_shape(self, tmp_path):
        """When batch_size > 1, tensors should be stacked along dim 0."""
        _create_cache_pair(str(tmp_path), "img1", 512, 512)
        _create_cache_pair(str(tmp_path), "img2", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=2,
        )
        ds.prepare_for_training()
        batch = ds[0]

        # Should be stacked: [2, C, F, H, W]
        assert batch["latents"].shape[0] == 2

    def test_video_bucket_includes_frames(self, tmp_path):
        """Video items should bucket by (W, H, F)."""
        _create_cache_pair(str(tmp_path), "clip", 960, 544, frame_count=25, frame_pos=0)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
        )
        ds.prepare_for_training()
        # Bucket should be (960, 544, 25)
        assert ds.batch_manager is not None
        bucket_resos = ds.batch_manager.bucket_resos
        assert (960, 544, 25) in bucket_resos

    def test_epoch_and_seed(self, tmp_path):
        _create_cache_pair(str(tmp_path), "img1", 512, 512)
        _create_cache_pair(str(tmp_path), "img2", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
        )
        ds.prepare_for_training()
        ds.set_seed(42)
        ds.set_current_epoch(1)
        ds.shuffle_buckets()  # Should not raise

    def test_loss_weight_stored_on_dataset(self):
        """CachedDataset stores loss_weight attribute correctly."""
        ds = CachedDataset(
            cache_directory="/tmp/fake",
            architecture="wan",
            batch_size=1,
            loss_weight=3.0,
        )
        assert ds.loss_weight == 3.0

    def test_loss_weight_default_is_one(self):
        """CachedDataset default loss_weight is 1.0."""
        ds = CachedDataset(
            cache_directory="/tmp/fake",
            architecture="wan",
            batch_size=1,
        )
        assert ds.loss_weight == 1.0

    def test_loss_weight_wired_to_item_info(self, tmp_path):
        """ItemInfo objects in prepare_for_training get the dataset's loss_weight."""
        _create_cache_pair(str(tmp_path), "img1", 512, 512)

        ds = CachedDataset(
            cache_directory=str(tmp_path),
            architecture="wan",
            batch_size=1,
            loss_weight=4.0,
        )
        ds.prepare_for_training()

        # Check that all items in all buckets have the correct loss_weight
        assert ds.batch_manager is not None
        for items in ds.batch_manager.buckets.values():
            for item in items:
                assert item.loss_weight == 4.0


# ---------------------------------------------------------------------------
# CachedDatasetGroup
# ---------------------------------------------------------------------------

class TestCachedDatasetGroup:
    def test_group_combines_datasets(self, tmp_path):
        dir1 = tmp_path / "ds1"
        dir2 = tmp_path / "ds2"
        dir1.mkdir()
        dir2.mkdir()

        _create_cache_pair(str(dir1), "img1", 512, 512)
        _create_cache_pair(str(dir2), "img2", 512, 512)

        ds1 = CachedDataset(str(dir1), "wan", batch_size=1)
        ds2 = CachedDataset(str(dir2), "wan", batch_size=1)
        ds1.prepare_for_training()
        ds2.prepare_for_training()

        group = CachedDatasetGroup([ds1, ds2])
        assert group.num_train_items == 2
        assert len(group) == 2

    def test_group_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CachedDatasetGroup([])


# ---------------------------------------------------------------------------
# TOML config parsing
# ---------------------------------------------------------------------------

class TestTomlConfig:
    def test_parse_simple_toml(self, tmp_path):
        from trainer.data.toml_config import parse_toml_config

        # Create cache files
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _create_cache_pair(str(cache_dir), "img1", 512, 512)

        # Write TOML config
        toml_path = tmp_path / "dataset.toml"
        toml_path.write_text(
            f'[general]\nbatch_size = 2\n\n'
            f'[[datasets]]\ncache_directory = "{cache_dir.as_posix()}"\n'
            f'num_repeats = 1\n'
        )

        datasets = parse_toml_config(str(toml_path), architecture="wan")
        assert len(datasets) == 1
        assert datasets[0].batch_size == 2

    def test_parse_batch_size_override(self, tmp_path):
        from trainer.data.toml_config import parse_toml_config

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _create_cache_pair(str(cache_dir), "img1", 512, 512)

        toml_path = tmp_path / "dataset.toml"
        toml_path.write_text(
            f'[general]\nbatch_size = 2\n\n'
            f'[[datasets]]\ncache_directory = "{cache_dir.as_posix()}"\n'
        )

        datasets = parse_toml_config(str(toml_path), architecture="wan", batch_size_override=4)
        assert datasets[0].batch_size == 4

    def test_parse_missing_file_raises(self):
        from trainer.data.toml_config import parse_toml_config

        with pytest.raises(FileNotFoundError):
            parse_toml_config("/nonexistent/config.toml", architecture="wan")

    def test_parse_no_datasets_raises(self, tmp_path):
        from trainer.data.toml_config import parse_toml_config

        toml_path = tmp_path / "empty.toml"
        toml_path.write_text("[general]\nbatch_size = 1\n")

        with pytest.raises(ValueError, match="No \\[\\[datasets\\]\\]"):
            parse_toml_config(str(toml_path), architecture="wan")

    def test_parse_multiple_datasets(self, tmp_path):
        from trainer.data.toml_config import parse_toml_config

        dir1 = tmp_path / "cache1"
        dir2 = tmp_path / "cache2"
        dir1.mkdir()
        dir2.mkdir()
        _create_cache_pair(str(dir1), "img1", 512, 512)
        _create_cache_pair(str(dir2), "img2", 512, 512)

        toml_path = tmp_path / "multi.toml"
        toml_path.write_text(
            f'[general]\nbatch_size = 1\n\n'
            f'[[datasets]]\ncache_directory = "{dir1.as_posix()}"\nnum_repeats = 2\n\n'
            f'[[datasets]]\ncache_directory = "{dir2.as_posix()}"\nnum_repeats = 3\n'
        )

        datasets = parse_toml_config(str(toml_path), architecture="wan")
        assert len(datasets) == 2
        assert datasets[0].num_repeats == 2
        assert datasets[1].num_repeats == 3


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

class TestCaching:
    def test_check_cache_exists_true(self, tmp_path):
        _create_latent_cache(str(tmp_path), "img1", 512, 512, "wan")
        assert check_cache_exists(str(tmp_path), "wan") is True

    def test_check_cache_exists_false(self, tmp_path):
        assert check_cache_exists(str(tmp_path), "wan") is False


# ---------------------------------------------------------------------------
# create_dataloader (integration)
# ---------------------------------------------------------------------------

class TestCreateDataloader:
    def test_inline_datasets(self, tmp_path):
        from trainer.data.loader import create_dataloader
        from trainer.config.schema import DataConfig, DatasetEntry

        _create_cache_pair(str(tmp_path), "img1", 512, 512)
        _create_cache_pair(str(tmp_path), "img2", 512, 512)

        data_config = DataConfig(
            datasets=[DatasetEntry(path=str(tmp_path), repeats=1)],
            num_workers=0,
            persistent_workers=False,
        )

        strategy = MagicMock()
        strategy.architecture = "wan"
        components = MagicMock()

        dl = create_dataloader(
            config=data_config,
            strategy=strategy,
            components=components,
            batch_size=2,
        )

        assert dl is not None
        batch = next(iter(dl))
        assert isinstance(batch, dict)
        assert "latents" in batch
        assert batch["latents"].shape[0] == 2  # batch of 2

    def test_dataset_weight_in_batch(self, tmp_path):
        """Batches produced by create_dataloader always contain 'dataset_weight'."""
        from trainer.data.loader import create_dataloader
        from trainer.config.schema import DataConfig, DatasetEntry

        _create_cache_pair(str(tmp_path), "img1", 512, 512)

        data_config = DataConfig(
            datasets=[DatasetEntry(path=str(tmp_path), repeats=1, weight=1.0)],
            num_workers=0,
            persistent_workers=False,
        )

        strategy = MagicMock()
        strategy.architecture = "wan"
        components = MagicMock()

        dl = create_dataloader(
            config=data_config,
            strategy=strategy,
            components=components,
            batch_size=1,
        )

        batch = next(iter(dl))
        assert "dataset_weight" in batch
        assert isinstance(batch["dataset_weight"], torch.Tensor)
        assert batch["dataset_weight"].item() == pytest.approx(1.0)

    def test_custom_weight_propagates_to_batch(self, tmp_path):
        """DatasetEntry.weight != 1.0 is propagated through to batch['dataset_weight']."""
        from trainer.data.loader import create_dataloader
        from trainer.config.schema import DataConfig, DatasetEntry

        _create_cache_pair(str(tmp_path), "img1", 512, 512)

        data_config = DataConfig(
            datasets=[DatasetEntry(path=str(tmp_path), repeats=1, weight=0.5)],
            num_workers=0,
            persistent_workers=False,
        )

        strategy = MagicMock()
        strategy.architecture = "wan"
        components = MagicMock()

        dl = create_dataloader(
            config=data_config,
            strategy=strategy,
            components=components,
            batch_size=1,
        )

        batch = next(iter(dl))
        assert batch["dataset_weight"].item() == pytest.approx(0.5)

    def test_toml_config_path(self, tmp_path):
        from trainer.data.loader import create_dataloader
        from trainer.config.schema import DataConfig

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        _create_cache_pair(str(cache_dir), "img1", 512, 512)

        toml_path = tmp_path / "dataset.toml"
        toml_path.write_text(
            f'[general]\nbatch_size = 1\n\n'
            f'[[datasets]]\ncache_directory = "{cache_dir.as_posix()}"\n'
        )

        data_config = DataConfig(
            dataset_config_path=str(toml_path),
            num_workers=0,
            persistent_workers=False,
        )

        strategy = MagicMock()
        strategy.architecture = "wan"
        components = MagicMock()

        dl = create_dataloader(
            config=data_config,
            strategy=strategy,
            components=components,
            batch_size=1,
        )

        batch = next(iter(dl))
        assert "latents" in batch

    def test_no_data_source_raises(self):
        from trainer.data.loader import create_dataloader
        from trainer.config.schema import DataConfig

        data_config = DataConfig()
        strategy = MagicMock()
        strategy.architecture = "wan"
        components = MagicMock()

        with pytest.raises(ValueError, match="No data source"):
            create_dataloader(
                config=data_config,
                strategy=strategy,
                components=components,
                batch_size=1,
            )

    def test_missing_cache_raises(self, tmp_path):
        from trainer.data.loader import create_dataloader
        from trainer.config.schema import DataConfig, DatasetEntry

        data_config = DataConfig(
            datasets=[DatasetEntry(path=str(tmp_path / "nonexistent"), repeats=1)],
            num_workers=0,
        )

        strategy = MagicMock()
        strategy.architecture = "wan"
        components = MagicMock()

        with pytest.raises(FileNotFoundError):
            create_dataloader(
                config=data_config,
                strategy=strategy,
                components=components,
                batch_size=1,
            )


# ---------------------------------------------------------------------------
# _compute_loss and _compute_weighted_loss with loss_weight (base.py)
# ---------------------------------------------------------------------------

class TestComputeLossWeight:
    """Unit tests for the loss_weight parameter on ModelStrategy._compute_loss
    and _compute_weighted_loss."""

    def _make_strategy(self):
        """Return a minimal ModelStrategy with _loss_fn=None (uses MSE default)."""
        from trainer.arch.base import ModelStrategy
        from unittest.mock import MagicMock
        cfg = MagicMock()
        strategy = ModelStrategy.__new__(ModelStrategy)
        strategy.config = cfg
        strategy._loss_fn = None
        strategy._unreduced_loss_fn = None
        strategy._weight_fn = None
        return strategy

    def test_compute_loss_no_weight_is_mse(self):
        """With loss_weight=None, _compute_loss returns standard MSE."""
        from trainer.arch.base import ModelStrategy
        import torch.nn.functional as F

        strategy = self._make_strategy()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)

        result = strategy._compute_loss(pred, target, loss_weight=None)
        expected = F.mse_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_compute_loss_weight_one_is_identity(self):
        """loss_weight of all-ones is a no-op (multiplies by 1.0)."""
        strategy = self._make_strategy()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)

        weight = torch.ones(2)
        unweighted = strategy._compute_loss(pred, target, loss_weight=None)
        weighted = strategy._compute_loss(pred, target, loss_weight=weight)
        assert torch.allclose(unweighted, weighted)

    def test_compute_loss_weight_scales_loss(self):
        """loss_weight of 2.0 doubles the loss value."""
        strategy = self._make_strategy()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)

        weight = torch.full((2,), 2.0)
        unweighted = strategy._compute_loss(pred, target, loss_weight=None)
        weighted = strategy._compute_loss(pred, target, loss_weight=weight)
        assert torch.allclose(weighted, unweighted * 2.0)

    def test_compute_loss_weight_zero_gives_zero(self):
        """loss_weight of 0.0 produces a zero loss regardless of pred/target."""
        strategy = self._make_strategy()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)

        weight = torch.zeros(2)
        result = strategy._compute_loss(pred, target, loss_weight=weight)
        assert result.item() == pytest.approx(0.0)

    def test_compute_loss_weight_partial(self):
        """loss_weight.mean() is applied: [1.0, 3.0] → mean=2.0 → doubles loss."""
        strategy = self._make_strategy()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)

        weight = torch.tensor([1.0, 3.0])  # mean = 2.0
        unweighted = strategy._compute_loss(pred, target, loss_weight=None)
        weighted = strategy._compute_loss(pred, target, loss_weight=weight)
        assert torch.allclose(weighted, unweighted * 2.0)

    def test_compute_weighted_loss_no_weight_fn_no_loss_weight(self):
        """Without weight_fn or loss_weight, _compute_weighted_loss == _compute_loss."""
        import torch.nn.functional as F
        strategy = self._make_strategy()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        timesteps = torch.rand(2)

        result = strategy._compute_weighted_loss(pred, target, timesteps, loss_weight=None)
        expected = F.mse_loss(pred, target, reduction="mean")
        assert torch.allclose(result, expected)

    def test_compute_weighted_loss_no_weight_fn_with_loss_weight(self):
        """Without weight_fn, loss_weight is still applied via _compute_loss fallback."""
        strategy = self._make_strategy()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        timesteps = torch.rand(2)
        weight = torch.full((2,), 0.5)

        unweighted = strategy._compute_weighted_loss(pred, target, timesteps, loss_weight=None)
        weighted = strategy._compute_weighted_loss(pred, target, timesteps, loss_weight=weight)
        assert torch.allclose(weighted, unweighted * 0.5)
