"""
Data loading, caching utilities, and regularization data iteration.

``create_dataloader()`` is imported by ``trainer.training.trainer`` to build
the DataLoader from config.  It supports two modes:

1. **TOML config** (Musubi compat): ``data.dataset_config_path`` points to a
   Musubi-format ``.toml`` file with ``[general]`` and ``[[datasets]]`` sections.

2. **Inline datasets**: ``data.datasets`` contains a list of ``DatasetEntry``
   objects specifying cache directories directly.

The returned DataLoader uses ``batch_size=1`` because the dataset self-batches
via ``BucketBatchManager``.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Iterator

import torch.utils.data

from trainer.config.schema import DataConfig
from trainer.arch.base import ModelStrategy, ModelComponents
from trainer.data.dataset import CachedDataset, CachedDatasetGroup

logger = logging.getLogger(__name__)


def _unwrap_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function that unwraps the outer list.

    Because the dataset self-batches (returns a pre-batched dict),
    DataLoader's default collate would wrap it in an extra list dimension.
    This function simply returns the single dict from the batch list.
    """
    if len(batch) != 1:
        raise RuntimeError(
            f"Expected batch_size=1 from DataLoader (dataset self-batches), got {len(batch)}"
        )
    return batch[0]


def create_dataloader(
    config: DataConfig,
    strategy: ModelStrategy,
    components: ModelComponents,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader from the data configuration.

    Parameters
    ----------
    config:
        The ``DataConfig`` from the training configuration.
    strategy:
        The model strategy (provides ``architecture`` name).
    components:
        Model components (reserved for future use, e.g. tokenizer access).
    batch_size:
        Training batch size (number of items per self-batched sample).

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader with ``batch_size=1`` and custom collate.

    Raises
    ------
    FileNotFoundError
        If no cache files are found.
    ValueError
        If neither ``dataset_config_path`` nor ``datasets`` is provided.
    """
    architecture = strategy.architecture
    datasets: list[CachedDataset] = []

    if config.dataset_config_path is not None:
        # Mode 1: Musubi-compatible TOML config
        from trainer.data.toml_config import parse_toml_config
        datasets = parse_toml_config(
            toml_path=config.dataset_config_path,
            architecture=architecture,
            batch_size_override=batch_size,
        )
    elif config.datasets:
        # Mode 2: Inline dataset entries
        for entry in config.datasets:
            ds = CachedDataset(
                cache_directory=entry.path,
                architecture=architecture,
                batch_size=batch_size,
                num_repeats=entry.repeats,
                enable_bucket=config.enable_bucket,
                loss_weight=entry.weight,
            )
            datasets.append(ds)
    else:
        raise ValueError(
            "No data source configured. Provide either 'dataset_config_path' "
            "(path to a Musubi TOML config) or 'datasets' (inline dataset entries) "
            "in the data configuration."
        )

    # Prepare all datasets (discovers files, builds bucket managers).
    # prepare_for_training() itself raises FileNotFoundError if no cache files found,
    # so we don't need a separate check_cache_exists pre-scan (avoids double glob).
    for ds in datasets:
        ds.prepare_for_training()

    # Wrap in a group if multiple datasets
    if len(datasets) == 1:
        dataset: torch.utils.data.Dataset = datasets[0]
    else:
        dataset = CachedDatasetGroup(datasets)

    # pin_memory is only beneficial with worker processes; when num_workers=0
    # (main process loads data), it just adds a redundant CPU-side copy.
    use_pin_memory = config.num_workers > 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Shuffling is handled by BucketBatchManager
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        collate_fn=_unwrap_collate,
        pin_memory=use_pin_memory,
    )

    logger.info(
        "Created dataloader: %d dataset(s), %d total batches, pin_memory=%s",
        len(datasets),
        len(dataset),
        use_pin_memory,
    )

    return dataloader


# ---------------------------------------------------------------------------
# Cache checking utilities (formerly data/caching.py)
# ---------------------------------------------------------------------------

def check_cache_exists(cache_directory: str, architecture: str) -> bool:
    """Quick sanity check: does the cache directory contain at least one
    file matching the expected latent cache pattern?

    Uses iglob to avoid materializing the full file list - returns as soon
    as one match is found.
    """
    pattern = os.path.join(cache_directory, f"*_{architecture}.safetensors")
    return next(glob.iglob(pattern), None) is not None


def log_caching_instructions(architecture: str) -> None:
    """Log instructions for running Musubi_Tuner's caching scripts."""
    logger.info(
        "=== Pre-caching Required ===\n"
        "This trainer requires pre-cached latent and text-encoder outputs.\n"
        "Run Musubi_Tuner's caching scripts before training:\n"
        "\n"
        "  1. Cache latents:\n"
        "     python wan_cache_latents.py \\\n"
        "       --dataset_config <your_dataset.toml> \\\n"
        "       --model_config <model_config.yaml> \\\n"
        "       --vae <path_to_vae>\n"
        "\n"
        "  2. Cache text-encoder outputs:\n"
        "     python wan_cache_text_encoder_outputs.py \\\n"
        "       --dataset_config <your_dataset.toml> \\\n"
        "       --model_config <model_config.yaml> \\\n"
        "       --t5 <path_to_t5>\n"
        "\n"
        "Architecture: %s",
        architecture,
    )


# ---------------------------------------------------------------------------
# Regularization data iterator (formerly data/reg_loader.py)
# ---------------------------------------------------------------------------

class RegDataIterator:
    """Infinite-cycling iterator over a regularization DataLoader.

    Automatically restarts when the underlying dataloader is exhausted.
    Thread-safe for single-consumer use (typical training loop).
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self._iter: Iterator | None = None

    def next_batch(self) -> dict[str, Any]:
        """Get the next regularization batch, cycling if needed."""
        if self._iter is None:
            self._iter = iter(self.dataloader)
        try:
            return next(self._iter)
        except StopIteration:
            logger.debug("Regularization dataloader exhausted - cycling.")
            self._iter = iter(self.dataloader)
            return next(self._iter)

    def __len__(self) -> int:
        """Number of batches per epoch (before cycling)."""
        return len(self.dataloader)
