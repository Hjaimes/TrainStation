"""
End-to-end tests with mock models and synthetic data.

These tests verify the full pipeline from TrainingSession.start() through
the training loop without requiring real Wan model weights or a GPU.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from trainer.arch.base import ModelComponents, TrainStepOutput
from trainer.arch.wan.strategy import WanStrategy


# ---------------------------------------------------------------------------
# Helpers — synthetic cache files
# ---------------------------------------------------------------------------

def _create_cache_pair(
    directory: str,
    name: str,
    w: int = 512,
    h: int = 512,
    arch: str = "wan",
    frame_count: int = 1,
    frame_pos: int = 0,
    seq_len: int = 77,
) -> None:
    """Create both latent and TE cache files for a synthetic training item."""
    lat_h, lat_w = h // 8, w // 8

    # Latent cache
    if frame_count > 1:
        lat_fname = f"{name}_{frame_pos:05d}-{frame_count:03d}_{w:04d}x{h:04d}_{arch}.safetensors"
    else:
        lat_fname = f"{name}_{w:04d}x{h:04d}_{arch}.safetensors"

    save_file(
        {f"latents_{frame_count}x{lat_h}x{lat_w}_bfloat16": torch.randn(
            16, frame_count, lat_h, lat_w, dtype=torch.bfloat16
        )},
        os.path.join(directory, lat_fname),
    )

    # TE cache
    te_fname = f"{name}_{arch}_te.safetensors"
    save_file(
        {
            "t5_bfloat16": torch.randn(seq_len, 4096, dtype=torch.bfloat16),
            "t5_mask": torch.ones(seq_len, dtype=torch.bfloat16),
        },
        os.path.join(directory, te_fname),
    )


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------

class TinyMockDiT(nn.Module):
    """Tiny model that mimics WanModel's forward signature.

    Returns list of tensors (one per batch item) like real WanModel.
    """

    def __init__(self, in_channels: int = 16, out_channels: int = 16):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None,
                skip_block_indices=None, f_indices=None):
        # x is either a batched tensor [B, C, F, H, W] or iterable of [C, F, H, W]
        if isinstance(x, torch.Tensor) and x.dim() == 5:
            results = []
            for i in range(x.shape[0]):
                item = x[i]  # [C, F, H, W]
                c, f, h, w = item.shape
                flat = item.reshape(c, -1).permute(1, 0)  # [F*H*W, C]
                out = self.linear(flat.float())  # [F*H*W, C_out]
                results.append(out.permute(1, 0).reshape(c, f, h, w))
            return results
        else:
            results = []
            for item in x:
                c, f, h, w = item.shape
                flat = item.reshape(c, -1).permute(1, 0)
                out = self.linear(flat.float())
                results.append(out.permute(1, 0).reshape(c, f, h, w))
            return results

    def enable_gradient_checkpointing(self):
        pass

    def prepare_block_swap_before_forward(self):
        pass


def _mock_setup(strategy: WanStrategy) -> ModelComponents:
    """Replace WanStrategy.setup() to return a TinyMockDiT.

    Must set all cached instance attributes that setup() would set,
    since training_step() reads them directly.
    """
    import math
    from trainer.arch.wan.components.configs import WAN_CONFIGS

    cfg = strategy.config
    wan_config = WAN_CONFIGS["t2v-14B"]
    patch_size = wan_config.patch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy._blocks_to_swap = 0
    strategy._device = device
    strategy._train_dtype = torch.bfloat16
    strategy._wan_config = wan_config
    strategy._task = "t2v-14B"
    strategy._patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
    strategy._noise_offset_val = cfg.training.noise_offset
    strategy._use_gradient_checkpointing = cfg.model.gradient_checkpointing
    dfs = cfg.training.discrete_flow_shift
    strategy._flow_shift = math.exp(dfs) if dfs != 0 else 1.0
    strategy._ts_method = cfg.training.timestep_sampling
    strategy._ts_min = cfg.training.min_timestep
    strategy._ts_max = cfg.training.max_timestep
    strategy._ts_sigmoid_scale = cfg.training.sigmoid_scale
    strategy._ts_logit_mean = cfg.training.logit_mean
    strategy._ts_logit_std = cfg.training.logit_std

    model = TinyMockDiT(in_channels=16, out_channels=16).to(device)

    return ModelComponents(
        model=model,
        extra={
            "wan_config": wan_config,
            "task": "t2v-14B",
            "patch_size": patch_size,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWanStrategyTrainingStep:
    """Test training_step with a mock model on CPU."""

    def test_training_step_shapes(self, tmp_path):
        """Feed synthetic batch through WanStrategy.training_step(), verify output."""
        from trainer.config.schema import TrainConfig

        config = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake/path",
                   "gradient_checkpointing": False},
            training={"method": "full_finetune", "timestep_sampling": "uniform"},
            data={"datasets": [{"path": str(tmp_path)}]},
        )
        strategy = WanStrategy(config)
        components = _mock_setup(strategy)

        # Create a synthetic batch (like what BucketBatchManager would return)
        batch = {
            "latents": torch.randn(2, 16, 1, 64, 64, dtype=torch.bfloat16),
            "t5": [
                torch.randn(77, 4096, dtype=torch.bfloat16),
                torch.randn(77, 4096, dtype=torch.bfloat16),
            ],
            "t5_mask": torch.ones(2, 77, dtype=torch.bfloat16),
            "timesteps": None,
        }

        output = strategy.training_step(components, batch, step=0)

        assert isinstance(output, TrainStepOutput)
        assert output.loss.dim() == 0  # scalar
        assert torch.isfinite(output.loss)
        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics

    def test_training_step_with_stacked_t5(self, tmp_path):
        """When T5 is stacked (fixed-length), should still work."""
        from trainer.config.schema import TrainConfig

        config = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake/path",
                   "gradient_checkpointing": False},
            training={"method": "full_finetune"},
            data={"datasets": [{"path": str(tmp_path)}]},
        )
        strategy = WanStrategy(config)
        components = _mock_setup(strategy)

        batch = {
            "latents": torch.randn(1, 16, 1, 64, 64, dtype=torch.bfloat16),
            "t5": torch.randn(1, 77, 4096, dtype=torch.bfloat16),  # stacked
            "t5_mask": torch.ones(1, 77, dtype=torch.bfloat16),
            "timesteps": None,
        }

        output = strategy.training_step(components, batch, step=0)
        assert torch.isfinite(output.loss)


class TestDataloaderToStrategyIntegration:
    """Create synthetic cache files, create dataloader, iterate one batch,
    verify it's compatible with strategy's expected batch format."""

    def test_dataloader_batch_feeds_strategy(self, tmp_path):
        from trainer.config.schema import TrainConfig, DataConfig, DatasetEntry
        from trainer.data.loader import create_dataloader

        # Create synthetic cache
        _create_cache_pair(str(tmp_path), "img1", 512, 512)
        _create_cache_pair(str(tmp_path), "img2", 512, 512)

        config = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake/path",
                   "gradient_checkpointing": False},
            training={"method": "full_finetune", "batch_size": 2},
            data={"datasets": [{"path": str(tmp_path), "repeats": 1}]},
        )
        strategy = WanStrategy(config)
        components = _mock_setup(strategy)

        data_config = DataConfig(
            datasets=[DatasetEntry(path=str(tmp_path), repeats=1)],
            num_workers=0,
            persistent_workers=False,
        )

        dl = create_dataloader(
            config=data_config,
            strategy=strategy,
            components=components,
            batch_size=2,
        )

        batch = next(iter(dl))

        # Verify batch has the right keys
        assert "latents" in batch
        assert "t5" in batch
        assert "timesteps" in batch

        # Feed it to training_step
        output = strategy.training_step(components, batch, step=0)
        assert torch.isfinite(output.loss)


class TestTimestepDistributions:
    """Statistical tests for timestep sampling methods."""

    @pytest.mark.parametrize("method", ["uniform", "sigmoid", "logit_normal", "shift"])
    def test_distribution_range(self, method):
        """All methods should produce t in [0, 1]."""
        from trainer.arch.base import ModelStrategy
        t = ModelStrategy._sample_t(10000, torch.device("cpu"), method=method)
        assert t.min() >= 0.0 - 1e-6
        assert t.max() <= 1.0 + 1e-6

    def test_uniform_is_uniform(self):
        """Uniform distribution: each quartile should have ~25% of samples."""
        from trainer.arch.base import ModelStrategy
        t = ModelStrategy._sample_t(40000, torch.device("cpu"), method="uniform")
        for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            frac = ((t >= lo) & (t < hi)).float().mean().item()
            assert 0.20 < frac < 0.30, f"Quartile [{lo}, {hi}): {frac:.3f} not ~0.25"

    def test_sigmoid_is_peaked(self):
        """Sigmoid with scale=1: should concentrate more around 0.5 than edges."""
        from trainer.arch.base import ModelStrategy
        t = ModelStrategy._sample_t(
            40000, torch.device("cpu"), method="sigmoid", sigmoid_scale=1.0,
        )
        mid_frac = ((t >= 0.3) & (t <= 0.7)).float().mean().item()
        assert mid_frac > 0.5, f"Expected >50% in [0.3, 0.7], got {mid_frac:.3f}"


class TestFullE2EWithMock:
    """Full TrainingSession.start() flow with mock model."""

    def test_wan_full_finetune_e2e(self, tmp_path):
        """Run a complete training session with mock model, 3 steps, verify completion."""
        from trainer.config.schema import TrainConfig
        from trainer.training.session import TrainingSession

        # Create cache data
        for i in range(4):
            _create_cache_pair(str(tmp_path), f"img{i}", 512, 512)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a fake model file to pass validation
        fake_model = tmp_path / "fake_model.safetensors"
        fake_model.touch()

        config = TrainConfig(
            model={
                "architecture": "wan",
                "base_model_path": str(fake_model),
                "gradient_checkpointing": False,
            },
            training={
                "method": "full_finetune",
                "max_steps": 3,
                "batch_size": 2,
                "mixed_precision": "no",
                "seed": 42,
            },
            optimizer={
                "optimizer_type": "adamw",
                "learning_rate": 1e-4,
                "scheduler_type": "constant",
            },
            data={
                "datasets": [{"path": str(tmp_path), "repeats": 1}],
                "num_workers": 0,
                "persistent_workers": False,
            },
            saving={
                "output_dir": str(output_dir),
                "save_every_n_epochs": None,
            },
        )

        # Patch setup() to return mock model
        def patched_setup(self):
            return _mock_setup(self)

        session = TrainingSession()

        with patch.object(WanStrategy, "setup", patched_setup):
            session.start(config, mode="train")

        # If we get here without exception, the full pipeline completed
