"""Tests for trainer/data/augmentations.py - crop jitter and random flip."""
from __future__ import annotations

import torch
import pytest

from trainer.data.augmentations import apply_crop_jitter, apply_random_flip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def image_latents() -> torch.Tensor:
    """4-D batch of image latents [B=2, C=4, H=16, W=16]."""
    return torch.randn(2, 4, 16, 16)


@pytest.fixture()
def video_latents() -> torch.Tensor:
    """5-D batch of video latents [B=2, C=4, F=8, H=16, W=16]."""
    return torch.randn(2, 4, 8, 16, 16)


# ---------------------------------------------------------------------------
# apply_crop_jitter
# ---------------------------------------------------------------------------

class TestApplyCropJitter:
    def test_shape_preserved_4d(self, image_latents):
        result = apply_crop_jitter(image_latents, jitter_pixels=16)
        assert result.shape == image_latents.shape

    def test_shape_preserved_5d(self, video_latents):
        result = apply_crop_jitter(video_latents, jitter_pixels=16)
        assert result.shape == video_latents.shape

    def test_jitter_zero_is_identity(self, image_latents):
        result = apply_crop_jitter(image_latents, jitter_pixels=0)
        assert torch.equal(result, image_latents)

    def test_jitter_below_vae_scale_is_identity(self, image_latents):
        # jitter_pixels=7 with default vae_scale_factor=8 → max_shift=0
        result = apply_crop_jitter(image_latents, jitter_pixels=7, vae_scale_factor=8)
        assert torch.equal(result, image_latents)

    def test_jitter_can_change_content(self, image_latents):
        # With jitter_pixels=64 the tensor should shift at least some of the time.
        changed = False
        for _ in range(20):
            result = apply_crop_jitter(image_latents, jitter_pixels=64)
            if not torch.equal(result, image_latents):
                changed = True
                break
        assert changed, "apply_crop_jitter never changed the tensor - shifts may always be zero"

    def test_dtype_preserved(self, image_latents):
        result = apply_crop_jitter(image_latents.half(), jitter_pixels=16)
        assert result.dtype == torch.float16

    def test_custom_vae_scale_factor(self):
        latents = torch.arange(36, dtype=torch.float32).reshape(1, 1, 6, 6)
        # vae_scale_factor=4, jitter_pixels=4 → max_shift=1
        result = apply_crop_jitter(latents, jitter_pixels=4, vae_scale_factor=4)
        assert result.shape == latents.shape

    def test_roll_wraps_correctly(self):
        """Verify that torch.roll semantics hold: values wrap around, not clipped."""
        latents = torch.zeros(1, 1, 4, 4)
        latents[0, 0, 0, 0] = 1.0  # top-left corner
        # Force a shift of exactly 1 in height by using a large jitter and
        # checking that the sum is preserved (roll is lossless).
        result = apply_crop_jitter(latents, jitter_pixels=8, vae_scale_factor=8)
        assert result.sum().item() == pytest.approx(1.0), "Roll should preserve all values"


# ---------------------------------------------------------------------------
# apply_random_flip
# ---------------------------------------------------------------------------

class TestApplyRandomFlip:
    def test_shape_preserved_4d(self, image_latents):
        result = apply_random_flip(image_latents)
        assert result.shape == image_latents.shape

    def test_shape_preserved_5d(self, video_latents):
        result = apply_random_flip(video_latents)
        assert result.shape == video_latents.shape

    def test_prob_zero_is_identity(self, image_latents):
        result = apply_random_flip(image_latents, probability=0.0)
        assert torch.equal(result, image_latents)

    def test_prob_one_always_flips(self):
        latents = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        result = apply_random_flip(latents, probability=1.0)
        expected = torch.flip(latents, dims=[-1])
        assert torch.equal(result, expected)

    def test_flip_changes_content(self):
        """With prob=1.0 and non-symmetric content the result must differ."""
        latents = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        result = apply_random_flip(latents, probability=1.0)
        assert not torch.equal(result, latents)

    def test_flip_preserves_sum(self, image_latents):
        result = apply_random_flip(image_latents, probability=1.0)
        assert torch.allclose(result.sum(), image_latents.sum())

    def test_flip_is_per_sample(self):
        """Some samples should be flipped and some not when prob=0.5 and B is large."""
        torch.manual_seed(0)
        latents = torch.randn(32, 4, 8, 8)
        result = apply_random_flip(latents, probability=0.5)
        flipped_back = torch.flip(result, dims=[-1])
        # At least one sample should match the original (not flipped).
        matches_original = torch.all(result == latents, dim=(1, 2, 3))
        matches_flipped = torch.all(flipped_back == latents, dim=(1, 2, 3))
        combined = matches_original | matches_flipped
        assert combined.all(), "Every sample should be either original or flipped"
        # Ensure both outcomes occur.
        assert matches_original.any() and matches_flipped.any(), (
            "Expected a mix of flipped/non-flipped samples with prob=0.5"
        )

    def test_dtype_preserved(self, image_latents):
        result = apply_random_flip(image_latents.half(), probability=0.5)
        assert result.dtype == torch.float16
