"""
Core data pipeline for loading pre-cached .safetensors files.

Cache files are produced by Musubi_Tuner's caching scripts (cache_latents.py and
cache_text_encoder_outputs.py). This module reads those files and serves batches
to the training loop without any raw image/video loading.

Filename conventions expected on disk
--------------------------------------
Latent cache (image):
    {name}_{W:04d}x{H:04d}_{arch}.safetensors

Latent cache (video):
    {name}_{pos:05d}-{frames:03d}_{W:04d}x{H:04d}_{arch}.safetensors

Text-encoder cache (both):
    {name}_{arch}_te.safetensors

where {name} for video strips the frame-position+size tokens so that all clips
from the same source share a single TE cache file.
"""

from __future__ import annotations

import glob
import itertools
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.utils.data
from safetensors.torch import load_file

from trainer.data.mask_utils import load_mask as _load_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ItemInfo
# ---------------------------------------------------------------------------

@dataclass
class ItemInfo:
    """Metadata for a single cached training item (image or video clip)."""

    item_key: str
    original_size: tuple[int, int]                  # (W, H)
    bucket_reso: tuple[int, ...]                     # (W, H) for images, (W, H, F) for video
    frame_count: int = 1
    latent_cache_path: str = ""
    text_encoder_output_cache_path: str = ""
    loss_weight: float = 1.0                         # Per-dataset loss multiplier from DatasetEntry.weight
    mask_cache_path: str | None = None               # Optional path to {name}_mask.safetensors
    caption_path: str | None = None                  # Optional path to {item_key}.txt caption file (for TE training)

    def __repr__(self) -> str:
        return (
            f"ItemInfo(item_key={self.item_key!r}, "
            f"original_size={self.original_size}, "
            f"bucket_reso={self.bucket_reso}, "
            f"frame_count={self.frame_count})"
        )


# ---------------------------------------------------------------------------
# BucketBatchManager
# ---------------------------------------------------------------------------

class BucketBatchManager:
    """
    Manages pre-batched items grouped by bucket resolution.

    Items are grouped so that every batch contains tensors of identical spatial
    (and temporal for video) shape, enabling torch.stack across the batch.

    The dataset self-batches: DataLoader should be used with batch_size=1 and
    collate_fn that unwraps the outer list (or the default collate with squeeze).
    """

    def __init__(
        self,
        bucketed_items: dict[tuple[int, ...], list[ItemInfo]],
        batch_size: int,
    ) -> None:
        self.batch_size = batch_size
        self.buckets: dict[tuple[int, ...], list[ItemInfo]] = bucketed_items

        # Sorted for deterministic order before the first shuffle.
        self.bucket_resos: list[tuple[int, ...]] = sorted(self.buckets.keys())

        # List of (bucket_reso, batch_start_index) - one entry per batch.
        self.bucket_batch_indices: list[tuple[tuple[int, ...], int]] = []
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            num_batches = math.ceil(len(bucket) / self.batch_size)
            for i in range(num_batches):
                self.bucket_batch_indices.append((bucket_reso, i * self.batch_size))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def shuffle(self, seed: int | None = None) -> None:
        """Shuffle items within each bucket, then shuffle the batch order."""
        if seed is not None:
            random.seed(seed)

        for bucket in self.buckets.values():
            random.shuffle(bucket)

        random.shuffle(self.bucket_batch_indices)

    def __len__(self) -> int:
        return len(self.bucket_batch_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load safetensors cache files for one batch and return a collated dict.

        Key-stripping logic (improved from Musubi_Tuner's BucketBatchManager):
          - "varlen_*"  → strip prefix via slice (not .replace() - avoids
                          stripping from middle of key); tensor stays as list
          - "*_mask"    → keep key as-is
          - otherwise   → rsplit("_", 1)[0]  (strip dtype suffix)
          - if result starts with "latents_" → rsplit("_", 1)[0] again
                         (strip the FxHxW shape token)

        Non-varlen keys are stacked with torch.stack.
        "timesteps" is always set to None (trainer generates these).
        """
        bucket_reso, batch_start = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        batch_end = min(batch_start + self.batch_size, len(bucket))

        batch_tensor_data: dict[str, list[torch.Tensor]] = {}
        varlen_keys: set[str] = set()
        loss_weights: list[float] = []
        mask_tensors: list[torch.Tensor | None] = []

        for item_info in bucket[batch_start:batch_end]:
            loss_weights.append(item_info.loss_weight)

            # Load optional spatial mask for masked training
            if item_info.mask_cache_path is not None and os.path.isfile(item_info.mask_cache_path):
                mask_tensors.append(_load_mask(item_info.mask_cache_path))
            else:
                mask_tensors.append(None)

            sd_latent = load_file(item_info.latent_cache_path)
            sd_te = load_file(item_info.text_encoder_output_cache_path)

            # Iterate both dicts without merging into a third dict
            for raw_key, tensor in itertools.chain(sd_latent.items(), sd_te.items()):
                is_varlen = raw_key.startswith("varlen_")
                content_key = raw_key

                if is_varlen:
                    # Strip the "varlen_" prefix to get the logical key.
                    # Use slice (not .replace) to only strip the prefix, not
                    # occurrences elsewhere in the key.
                    content_key = content_key[7:]  # len("varlen_") == 7

                if content_key.endswith("_mask"):
                    # Mask tensors: keep the key exactly as-is.
                    pass
                else:
                    # Strip trailing dtype suffix (e.g. "_bf16", "_fp32").
                    content_key = content_key.rsplit("_", 1)[0]
                    # For latent tensors, also strip the FxHxW shape token.
                    if content_key.startswith("latents_"):
                        content_key = content_key.rsplit("_", 1)[0]

                batch_tensor_data.setdefault(content_key, []).append(tensor)

                if is_varlen:
                    varlen_keys.add(content_key)

        # Stack fixed-shape tensors; leave varlen keys as list[Tensor].
        result: dict[str, Any] = {}
        for key, tensors in batch_tensor_data.items():
            if key in varlen_keys:
                result[key] = tensors          # variable-length: list[Tensor]
            else:
                result[key] = torch.stack(tensors)  # [B, ...]

        # Trainer is responsible for sampling timesteps each step.
        result["timesteps"] = None

        # Per-sample dataset loss multiplier, shape (B,), dtype float32.
        # Scalar 1.0 for all items is the no-op default (DatasetEntry.weight default).
        result["dataset_weight"] = torch.tensor(loss_weights, dtype=torch.float32)

        # Spatial loss masks: only included when all items in the batch have a mask.
        # Shape: (B, 1, H, W) for images or (B, 1, F, H, W) for video.
        # Strategies resize to match latent spatial dims before loss computation.
        if all(m is not None for m in mask_tensors):
            result["loss_mask"] = torch.stack(mask_tensors)  # type: ignore[arg-type]

        # Raw captions for text encoder training (loaded from .txt files alongside cached data).
        # Only included when at least one item has a caption_path.
        captions: list[str] = []
        for item_info in bucket[batch_start:batch_end]:
            if item_info.caption_path is not None and os.path.isfile(item_info.caption_path):
                with open(item_info.caption_path, "r", encoding="utf-8") as f:
                    captions.append(f.read().strip())
            else:
                captions.append("")

        if any(c for c in captions):  # At least one non-empty caption
            result["captions"] = captions

        return result


# ---------------------------------------------------------------------------
# CachedDataset
# ---------------------------------------------------------------------------

class CachedDataset(torch.utils.data.Dataset):
    """
    Dataset that reads exclusively from pre-cached .safetensors files.

    No raw image or video data is ever loaded.  The dataset self-batches,
    so DataLoader should be called with batch_size=1.

    Parameters
    ----------
    cache_directory:
        Directory containing ``*_{architecture}.safetensors`` latent caches
        and the corresponding ``*_{architecture}_te.safetensors`` TE caches.
    architecture:
        Architecture short-name used as the file-name suffix, e.g. ``"wan"``.
    batch_size:
        Number of items per returned batch.
    num_repeats:
        How many times each item is repeated in the epoch (data augmentation
        via repetition at the dataset level).
    enable_bucket:
        Reserved for future use; currently all items are bucketed by their
        exact cache resolution.
    """

    def __init__(
        self,
        cache_directory: str,
        architecture: str,
        batch_size: int,
        num_repeats: int = 1,
        enable_bucket: bool = True,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.cache_directory = cache_directory
        self.architecture = architecture
        self.batch_size = batch_size
        self.num_repeats = max(1, num_repeats)
        self.enable_bucket = enable_bucket
        self.loss_weight = loss_weight

        self.batch_manager: BucketBatchManager | None = None
        self.num_train_items: int = 0
        self._seed: int = 0
        self._current_epoch: int = 0

    # ------------------------------------------------------------------
    # Filename parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_size_token(token: str) -> tuple[int, int] | None:
        """Parse a ``WWWWxHHHH`` token into (W, H), or return None."""
        if "x" not in token:
            return None
        parts = token.split("x")
        if len(parts) != 2:
            return None
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return None

    @staticmethod
    def _parse_frame_token(token: str) -> int | None:
        """Parse a ``PPPPP-FFF`` token and return the frame count, or None."""
        if "-" not in token:
            return None
        pos, _, frames = token.partition("-")
        try:
            int(pos)
            return int(frames)
        except ValueError:
            return None

    def _parse_latent_filename(
        self, filename: str
    ) -> tuple[str, tuple[int, int], int] | None:
        """
        Parse a latent cache filename into (item_key, (W, H), frame_count).

        Returns None if the filename does not match the expected pattern.

        Image pattern (tokens from the right):
            [-1] = "{arch}.safetensors"   (already stripped by the caller)
            [-2] = "WWWWxHHHH"
            [-3..] = item_key parts

        Video pattern:
            [-1] = "{arch}.safetensors"
            [-2] = "WWWWxHHHH"
            [-3] = "PPPPP-FFF"
            [-4..] = item_key parts
        """
        # Strip the arch suffix from the extension-free stem.
        stem = filename
        arch_suffix = f"_{self.architecture}"
        if not stem.endswith(arch_suffix):
            return None
        stem = stem[: -len(arch_suffix)]   # e.g.  "clip_00000-025_0960x0544"

        tokens = stem.split("_")
        if len(tokens) < 2:
            return None

        # tokens[-1] must be the size token "WWWWxHHHH"
        size = self._parse_size_token(tokens[-1])
        if size is None:
            return None
        w, h = size

        # Check whether tokens[-2] looks like a frame token "PPPPP-FFF"
        frame_count_parsed: int | None = None
        if len(tokens) >= 3:
            frame_count_parsed = self._parse_frame_token(tokens[-2])

        if frame_count_parsed is not None:
            # Video: item_key = everything before the frame+size tokens.
            frame_count = frame_count_parsed
            item_key = "_".join(tokens[:-2])
        else:
            # Image: item_key = everything before the size token.
            frame_count = 1
            item_key = "_".join(tokens[:-1])

        if not item_key:
            return None

        return item_key, (w, h), frame_count

    @staticmethod
    def _derive_te_filename(item_key: str, architecture: str) -> str:
        """Return just the filename for a TE cache (no directory)."""
        return f"{item_key}_{architecture}_te.safetensors"

    def _derive_te_cache_path(
        self, item_key: str, is_video: bool
    ) -> str:
        """
        Derive the text-encoder cache path from an item_key.

        For both images and videos the TE cache is keyed by the bare item
        name (without size or frame tokens), so all clips of a video share
        a single TE cache file.

        Path: ``{cache_directory}/{item_key}_{arch}_te.safetensors``
        """
        return os.path.join(
            self.cache_directory,
            self._derive_te_filename(item_key, self.architecture),
        )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_for_training(self) -> None:
        """
        Discover cache files, parse their filenames, pair them with TE caches,
        group by bucket resolution, and build the BucketBatchManager.

        Raises
        ------
        FileNotFoundError
            If no latent cache files are found in ``cache_directory``.
        """
        pattern = os.path.join(
            self.cache_directory, f"*_{self.architecture}.safetensors"
        )
        latent_files = sorted(glob.glob(pattern))

        if not latent_files:
            raise FileNotFoundError(
                f"No latent cache files found matching pattern: {pattern}\n"
                f"Run the caching scripts first to populate '{self.cache_directory}'."
            )

        # Build a set of existing TE cache filenames for O(1) lookup instead
        # of one os.path.exists syscall per latent file.
        te_suffix = f"_{self.architecture}_te.safetensors"
        existing_te_files: set[str] = {
            f for f in os.listdir(self.cache_directory)
            if f.endswith(te_suffix)
        }

        bucketed: dict[tuple[int, ...], list[ItemInfo]] = {}
        skipped = 0

        for cache_file in latent_files:
            # Pass the stem (no extension) to the parser.
            stem = os.path.splitext(os.path.basename(cache_file))[0]
            parsed = self._parse_latent_filename(stem)
            if parsed is None:
                logger.warning("Could not parse latent cache filename, skipping: %s", stem)
                skipped += 1
                continue

            item_key, original_size, frame_count = parsed
            is_video = frame_count > 1

            te_filename = self._derive_te_filename(item_key, self.architecture)
            if te_filename not in existing_te_files:
                logger.warning(
                    "Text-encoder cache not found for '%s', skipping. Expected: %s",
                    item_key,
                    te_filename,
                )
                skipped += 1
                continue

            te_path = os.path.join(self.cache_directory, te_filename)

            # Bucket key: (W, H) for images, (W, H, F) for video.
            if is_video:
                bucket_reso: tuple[int, ...] = (original_size[0], original_size[1], frame_count)
            else:
                bucket_reso = original_size  # (W, H)

            item = ItemInfo(
                item_key=item_key,
                original_size=original_size,
                bucket_reso=bucket_reso,
                frame_count=frame_count,
                latent_cache_path=cache_file,
                text_encoder_output_cache_path=te_path,
                loss_weight=self.loss_weight,
            )

            # Check for caption file (for text encoder training)
            caption_file = os.path.join(self.cache_directory, f"{item_key}.txt")
            if os.path.isfile(caption_file):
                item.caption_path = caption_file

            bucket = bucketed.setdefault(bucket_reso, [])
            bucket.extend([item] * self.num_repeats)

        if not bucketed:
            raise FileNotFoundError(
                f"Found {len(latent_files)} latent cache files but none could be paired "
                f"with a text-encoder cache in '{self.cache_directory}'. "
                f"Run cache_text_encoder_outputs first."
            )

        self.batch_manager = BucketBatchManager(bucketed, self.batch_size)
        self.num_train_items = sum(len(b) for b in bucketed.values()) // self.num_repeats

        total_unique = self.num_train_items
        total_with_repeats = total_unique * self.num_repeats
        logger.info(
            "CachedDataset [%s]: %d unique items (%d with repeats), "
            "%d buckets, %d batches. Skipped %d files.",
            self.architecture,
            total_unique,
            total_with_repeats,
            len(bucketed),
            len(self.batch_manager),
            skipped,
        )
        for reso, items in sorted(bucketed.items()):
            logger.info("  bucket %s: %d items", reso, len(items))

    # ------------------------------------------------------------------
    # Epoch / seed management
    # ------------------------------------------------------------------

    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def set_seed(self, seed: int, shared_epoch: int | None = None) -> None:
        self._seed = seed
        if shared_epoch is not None:
            self._current_epoch = shared_epoch

    def shuffle_buckets(self) -> None:
        """Shuffle bucket contents and batch ordering for the current epoch."""
        if self.batch_manager is None:
            raise RuntimeError("Call prepare_for_training() before shuffle_buckets().")
        effective_seed = self._seed + self._current_epoch
        self.batch_manager.shuffle(seed=effective_seed)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.batch_manager is None:
            return 0
        return len(self.batch_manager)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.batch_manager is None:
            raise RuntimeError("Call prepare_for_training() before accessing dataset items.")
        return self.batch_manager[idx]


# ---------------------------------------------------------------------------
# CachedDatasetGroup
# ---------------------------------------------------------------------------

class CachedDatasetGroup(torch.utils.data.ConcatDataset):
    """
    Wraps multiple CachedDataset instances into a single Dataset.

    Inherits length and item-access from ConcatDataset.  Adds epoch
    propagation and a convenience ``num_train_items`` property.
    """

    def __init__(self, datasets: list[CachedDataset]) -> None:
        if not datasets:
            raise ValueError("CachedDatasetGroup requires at least one CachedDataset.")
        super().__init__(datasets)
        self._datasets: list[CachedDataset] = datasets

    # ------------------------------------------------------------------
    # Epoch / seed management - propagate to all child datasets
    # ------------------------------------------------------------------

    def set_current_epoch(self, epoch: int) -> None:
        for ds in self._datasets:
            ds.set_current_epoch(epoch)

    def set_seed(self, seed: int, shared_epoch: int | None = None) -> None:
        for ds in self._datasets:
            ds.set_seed(seed, shared_epoch)

    def shuffle_buckets(self) -> None:
        for ds in self._datasets:
            ds.shuffle_buckets()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_train_items(self) -> int:
        """Total number of unique training items across all child datasets."""
        return sum(ds.num_train_items for ds in self._datasets)
