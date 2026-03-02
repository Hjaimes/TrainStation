"""Tests for trainer/data/text_processing.py — tag shuffle and token dropout."""
from __future__ import annotations

import pytest

from trainer.data.text_processing import (
    apply_token_dropout,
    process_caption,
    shuffle_tags,
)


# ---------------------------------------------------------------------------
# shuffle_tags
# ---------------------------------------------------------------------------

class TestShuffleTags:
    def test_empty_caption_returns_unchanged(self):
        assert shuffle_tags("") == ""

    def test_whitespace_only_returns_unchanged(self):
        assert shuffle_tags("   ") == "   "

    def test_single_tag_returns_unchanged(self):
        assert shuffle_tags("cat") == "cat"

    def test_output_contains_same_tags(self):
        caption = "a, b, c, d, e"
        result = shuffle_tags(caption, seed=0)
        original_tags = {t.strip() for t in caption.split(",")}
        result_tags = {t.strip() for t in result.split(",")}
        assert original_tags == result_tags

    def test_seed_reproducibility(self):
        caption = "tag1, tag2, tag3, tag4, tag5"
        r1 = shuffle_tags(caption, seed=42)
        r2 = shuffle_tags(caption, seed=42)
        assert r1 == r2

    def test_different_seeds_produce_different_order(self):
        # Very high probability — 5 tags have 120 permutations.
        caption = "a, b, c, d, e"
        results = {shuffle_tags(caption, seed=s) for s in range(20)}
        assert len(results) > 1, "All seeds produced the same order — shuffling may be broken"

    def test_keep_first_n_preserves_leading_tags(self):
        caption = "fixed1, fixed2, a, b, c, d"
        for seed in range(10):
            result = shuffle_tags(caption, keep_first_n=2, seed=seed)
            tags = [t.strip() for t in result.split(",")]
            assert tags[0] == "fixed1"
            assert tags[1] == "fixed2"

    def test_keep_first_n_greater_than_tag_count(self):
        # Should not crash — just return all tags in original order.
        caption = "a, b, c"
        result = shuffle_tags(caption, keep_first_n=10, seed=0)
        tags = [t.strip() for t in result.split(",")]
        assert tags == ["a", "b", "c"]

    def test_keep_first_n_zero_shuffles_all(self):
        caption = "a, b, c, d, e"
        # With keep_first_n=0 the first tag may change.
        shuffled_set = {shuffle_tags(caption, keep_first_n=0, seed=s) for s in range(30)}
        assert len(shuffled_set) > 1

    def test_custom_delimiter(self):
        caption = "a|b|c|d"
        result = shuffle_tags(caption, delimiter="|", seed=1)
        original_tags = set(caption.split("|"))
        result_tags = set(result.split("|"))
        # Strip potential spaces introduced by the function.
        result_tags = {t.strip() for t in result_tags}
        assert original_tags == result_tags

    def test_two_tags_always_returns_both(self):
        caption = "alpha, beta"
        result = shuffle_tags(caption, seed=99)
        result_tags = {t.strip() for t in result.split(",")}
        assert result_tags == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# apply_token_dropout
# ---------------------------------------------------------------------------

class TestApplyTokenDropout:
    def test_empty_caption_returns_unchanged(self):
        assert apply_token_dropout("", 0.5) == ""

    def test_zero_rate_returns_unchanged(self):
        caption = "a, b, c, d"
        assert apply_token_dropout(caption, 0.0) == caption

    def test_single_tag_returns_unchanged(self):
        assert apply_token_dropout("only", 0.9) == "only"

    def test_dropout_removes_some_tags(self):
        # With rate=0.8 and 20 tags we expect meaningful removal.
        caption = ", ".join(f"tag{i}" for i in range(20))
        result = apply_token_dropout(caption, dropout_rate=0.8, seed=7)
        result_tags = [t.strip() for t in result.split(",") if t.strip()]
        assert len(result_tags) < 20

    def test_kept_tags_are_subset_of_original(self):
        caption = "a, b, c, d, e, f"
        original = {t.strip() for t in caption.split(",")}
        result = apply_token_dropout(caption, dropout_rate=0.5, seed=3)
        result_tags = {t.strip() for t in result.split(",") if t.strip()}
        assert result_tags.issubset(original)

    def test_seed_reproducibility(self):
        caption = "a, b, c, d, e, f, g, h"
        r1 = apply_token_dropout(caption, 0.5, seed=42)
        r2 = apply_token_dropout(caption, 0.5, seed=42)
        assert r1 == r2

    def test_keep_first_n_never_dropped(self):
        caption = "must_keep1, must_keep2, " + ", ".join(f"tag{i}" for i in range(20))
        for seed in range(10):
            result = apply_token_dropout(caption, dropout_rate=0.99, keep_first_n=2, seed=seed)
            tags = [t.strip() for t in result.split(",") if t.strip()]
            assert tags[0] == "must_keep1"
            assert tags[1] == "must_keep2"

    def test_statistical_dropout_fraction(self):
        """Over many seeds, average fraction dropped should be near the rate."""
        caption = ", ".join(f"t{i}" for i in range(100))
        rate = 0.5
        total_original = 100
        total_kept = 0
        trials = 200
        for s in range(trials):
            result = apply_token_dropout(caption, dropout_rate=rate, seed=s)
            kept = len([t for t in result.split(",") if t.strip()])
            total_kept += kept
        avg_kept = total_kept / trials
        # Expect roughly 50 kept; allow ±10 for randomness.
        assert 35 <= avg_kept <= 65, f"Average kept={avg_kept:.1f} deviates too much from expected ~50"

    def test_rate_1_drops_all_non_fixed(self):
        caption = "fixed, a, b, c, d"
        result = apply_token_dropout(caption, dropout_rate=1.0, keep_first_n=1, seed=0)
        tags = [t.strip() for t in result.split(",") if t.strip()]
        assert tags == ["fixed"]

    def test_custom_delimiter(self):
        caption = "a|b|c|d"
        result = apply_token_dropout(caption, dropout_rate=0.0, delimiter="|")
        assert result == caption


# ---------------------------------------------------------------------------
# process_caption
# ---------------------------------------------------------------------------

class TestProcessCaption:
    def test_no_ops_returns_original(self):
        caption = "a, b, c"
        assert process_caption(caption) == caption

    def test_shuffle_only_same_tags(self):
        caption = "a, b, c, d, e"
        result = process_caption(caption, shuffle=True, seed=5)
        assert {t.strip() for t in result.split(",")} == {t.strip() for t in caption.split(",")}

    def test_dropout_only_subset(self):
        caption = "a, b, c, d, e, f"
        result = process_caption(caption, dropout_rate=0.8, seed=1)
        original = {t.strip() for t in caption.split(",")}
        result_tags = {t.strip() for t in result.split(",") if t.strip()}
        assert result_tags.issubset(original)

    def test_shuffle_and_dropout_combined(self):
        caption = ", ".join(f"tag{i}" for i in range(10))
        result = process_caption(caption, shuffle=True, dropout_rate=0.4, seed=99)
        original = {t.strip() for t in caption.split(",")}
        result_tags = {t.strip() for t in result.split(",") if t.strip()}
        assert result_tags.issubset(original)

    def test_seed_reproducibility(self):
        caption = "a, b, c, d, e, f, g"
        r1 = process_caption(caption, shuffle=True, dropout_rate=0.3, seed=7)
        r2 = process_caption(caption, shuffle=True, dropout_rate=0.3, seed=7)
        assert r1 == r2

    def test_keep_first_n_preserved_through_pipeline(self):
        caption = "keep_me, a, b, c, d, e, f"
        for seed in range(10):
            result = process_caption(caption, shuffle=True, keep_first_n=1, dropout_rate=0.9, seed=seed)
            first = result.split(",")[0].strip()
            assert first == "keep_me"

    def test_empty_caption_passthrough(self):
        assert process_caption("", shuffle=True, dropout_rate=0.5, seed=0) == ""
