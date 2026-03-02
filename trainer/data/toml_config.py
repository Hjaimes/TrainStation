"""
Simple TOML parser for Musubi-format dataset configuration files.

Converts a TOML file (with [general] and [[datasets]] sections) into
CachedDataset instances, without the full BlueprintGenerator/ConfigSanitizer
complexity.

Expected TOML format
--------------------
::

    [general]
    resolution = 512
    batch_size = 1
    enable_bucket = true

    [[datasets]]
    cache_directory = "/path/to/cache"
    num_repeats = 10

    [[datasets]]
    cache_directory = "/path/to/another_cache"
    num_repeats = 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from trainer.data.dataset import CachedDataset


def parse_toml_config(
    toml_path: str | Path,
    architecture: str,
    batch_size_override: int | None = None,
) -> list[CachedDataset]:
    """
    Parse a Musubi-format TOML config and return CachedDataset instances.

    Parameters
    ----------
    toml_path:
        Path to the TOML dataset configuration file.
    architecture:
        Architecture short-name (e.g. ``"wan"``).
    batch_size_override:
        If provided, overrides the batch_size from the TOML ``[general]`` section.

    Returns
    -------
    list[CachedDataset]
        One dataset per ``[[datasets]]`` entry in the TOML file.

    Raises
    ------
    FileNotFoundError
        If the TOML file does not exist.
    ValueError
        If the TOML is missing required fields.
    """
    toml_path = Path(toml_path)
    if not toml_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {toml_path}")

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    general: dict[str, Any] = config.get("general", {})
    dataset_entries: list[dict[str, Any]] = config.get("datasets", [])

    if not dataset_entries:
        raise ValueError(
            f"No [[datasets]] entries found in {toml_path}. "
            f"At least one dataset is required."
        )

    # General-level defaults.
    default_batch_size = general.get("batch_size", 1)
    default_enable_bucket = general.get("enable_bucket", True)

    batch_size = batch_size_override if batch_size_override is not None else default_batch_size

    datasets: list[CachedDataset] = []

    for i, entry in enumerate(dataset_entries):
        cache_dir = entry.get("cache_directory")
        if cache_dir is None:
            # Also accept "image_directory" / "video_directory" as aliases.
            cache_dir = entry.get("image_directory") or entry.get("video_directory")

        if cache_dir is None:
            raise ValueError(
                f"[[datasets]] entry {i} in {toml_path} is missing "
                f"'cache_directory' (or 'image_directory'/'video_directory')."
            )

        num_repeats = entry.get("num_repeats", 1)
        enable_bucket = entry.get("enable_bucket", default_enable_bucket)

        ds = CachedDataset(
            cache_directory=str(cache_dir),
            architecture=architecture,
            batch_size=batch_size,
            num_repeats=num_repeats,
            enable_bucket=enable_bucket,
        )
        datasets.append(ds)

    logger.info(
        "Parsed TOML config %s: %d dataset(s), batch_size=%d",
        toml_path,
        len(datasets),
        batch_size,
    )

    return datasets
