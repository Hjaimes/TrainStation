"""Pure text utilities for caption processing: tag shuffling and token dropout.

These functions are stateless and operate only on strings — no model or tensor
dependencies. They can be called safely from a DataLoader worker process.
"""
from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def shuffle_tags(
    caption: str,
    delimiter: str = ",",
    keep_first_n: int = 0,
    seed: int | None = None,
) -> str:
    """Shuffle tags in a caption string, keeping the first N tags fixed.

    Args:
        caption: Raw caption text.
        delimiter: Tag separator character.
        keep_first_n: Number of leading tags to preserve in-place.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Caption with shuffled tags, preserving the original delimiter and
        a single space after each delimiter to match common caption formats.
    """
    if not caption or not caption.strip():
        return caption

    tags = [t.strip() for t in caption.split(delimiter)]
    tags = [t for t in tags if t]  # drop empty strings from trailing delimiters

    if len(tags) <= 1:
        return tags[0] if tags else caption

    fixed = tags[:keep_first_n]
    to_shuffle = tags[keep_first_n:]

    rng = random.Random(seed)
    rng.shuffle(to_shuffle)

    sep = delimiter + " "
    return sep.join(fixed + to_shuffle)


def apply_token_dropout(
    caption: str,
    dropout_rate: float,
    delimiter: str = ",",
    keep_first_n: int = 0,
    seed: int | None = None,
) -> str:
    """Randomly drop tags from a caption string.

    Args:
        caption: Raw caption text.
        dropout_rate: Fraction of non-fixed tags to drop in [0, 1).
        delimiter: Tag separator character.
        keep_first_n: Number of leading tags guaranteed to be kept.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Caption with a random subset of tags removed. Fixed tags are never
        dropped. If all non-fixed tags are dropped the fixed tags are returned.
    """
    if not caption or not caption.strip() or dropout_rate <= 0.0:
        return caption

    tags = [t.strip() for t in caption.split(delimiter)]
    tags = [t for t in tags if t]

    if len(tags) <= 1:
        return tags[0] if tags else caption

    fixed = tags[:keep_first_n]
    candidates = tags[keep_first_n:]

    rng = random.Random(seed)
    kept = [t for t in candidates if rng.random() >= dropout_rate]

    sep = delimiter + " "
    return sep.join(fixed + kept)


def process_caption(
    caption: str,
    shuffle: bool = False,
    keep_first_n: int = 0,
    dropout_rate: float = 0.0,
    delimiter: str = ",",
    seed: int | None = None,
) -> str:
    """Apply shuffle then dropout. Combined entry-point for the data pipeline.

    Shuffle is applied before dropout so that which tags are dropped varies
    each step even when dropout_rate is constant.

    Args:
        caption: Raw caption text.
        shuffle: Whether to shuffle the non-fixed tags.
        keep_first_n: Number of leading tags to preserve in both operations.
        dropout_rate: Fraction of non-fixed tags to drop after shuffling.
        delimiter: Tag separator character.
        seed: Optional RNG seed. When provided both operations get the same
            seed so the result is fully reproducible.

    Returns:
        Processed caption string.
    """
    result = caption

    if shuffle:
        result = shuffle_tags(result, delimiter=delimiter, keep_first_n=keep_first_n, seed=seed)

    if dropout_rate > 0.0:
        # Use a derived seed for dropout so it differs from the shuffle seed
        # while still being deterministic when a seed is given.
        dropout_seed = (seed + 1) if seed is not None else None
        result = apply_token_dropout(
            result,
            dropout_rate=dropout_rate,
            delimiter=delimiter,
            keep_first_n=keep_first_n,
            seed=dropout_seed,
        )

    return result
