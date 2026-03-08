"""Tests for Flux 2 architecture components.

Following the pattern of tests/test_wan_components.py.
All tests run on CPU with tiny synthetic tensors - no real weights required.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_all_variants_present(self):
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS
        expected = {"dev", "klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b"}
        assert set(FLUX2_CONFIGS.keys()) == expected

    def test_dev_fields(self):
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS
        cfg = FLUX2_CONFIGS["dev"]
        assert cfg.in_channels == 128
        assert cfg.hidden_size == 6144
        assert cfg.num_heads == 48
        assert cfg.depth == 8
        assert cfg.depth_single_blocks == 48
        assert cfg.use_guidance_embed is True
        assert cfg.qwen_variant is None  # Mistral3

    def test_klein_4b_fields(self):
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS
        cfg = FLUX2_CONFIGS["klein-4b"]
        assert cfg.in_channels == 128
        assert cfg.hidden_size == 3072
        assert cfg.num_heads == 24
        assert cfg.depth == 5
        assert cfg.depth_single_blocks == 20
        assert cfg.use_guidance_embed is False
        assert cfg.qwen_variant == "4B"

    def test_klein_9b_fields(self):
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS
        cfg = FLUX2_CONFIGS["klein-9b"]
        assert cfg.hidden_size == 4096
        assert cfg.num_heads == 32
        assert cfg.depth == 8
        assert cfg.depth_single_blocks == 24
        assert cfg.use_guidance_embed is False
        assert cfg.qwen_variant == "8B"

    def test_axes_dim_consistency(self):
        """axes_dim must sum to hidden_size // num_heads for each variant."""
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS
        for name, cfg in FLUX2_CONFIGS.items():
            pe_dim = cfg.hidden_size // cfg.num_heads
            assert sum(cfg.axes_dim) == pe_dim, (
                f"Variant '{name}': axes_dim sum {sum(cfg.axes_dim)} != pe_dim {pe_dim}"
            )

    def test_configs_are_frozen(self):
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS
        cfg = FLUX2_CONFIGS["dev"]
        with pytest.raises((AttributeError, TypeError)):
            cfg.hidden_size = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_prc_img_single_sample(self):
        """Single (C, H, W) tensor packs to (HW, C) with correct IDs."""
        from trainer.arch.flux_2.components.utils import prc_img
        x = torch.randn(128, 4, 4)
        packed, ids = prc_img(x)
        assert packed.shape == (16, 128)   # HW=4*4, C=128
        assert ids.shape == (16, 4)        # (HW, 4)

    def test_prc_img_batched(self):
        """Batched (B, C, H, W) tensor packs to (B, HW, C) with correct IDs."""
        from trainer.arch.flux_2.components.utils import prc_img
        B, C, H, W = 2, 128, 8, 6
        x = torch.randn(B, C, H, W)
        packed, ids = prc_img(x)
        assert packed.shape == (B, H * W, C)
        assert ids.shape == (B, H * W, 4)

    def test_prc_txt_batched(self):
        """Batched (B, L, D) returns unchanged x and IDs of shape (B, L, 4)."""
        from trainer.arch.flux_2.components.utils import prc_txt
        B, L, D = 2, 512, 15360
        x = torch.randn(B, L, D)
        out, ids = prc_txt(x)
        assert out is x
        assert ids.shape == (B, L, 4)

    def test_unpack_latents(self):
        from trainer.arch.flux_2.components.utils import unpack_latents
        B, H, W, C = 2, 8, 6, 128
        x = torch.randn(B, H * W, C)
        out = unpack_latents(x, H, W)
        assert out.shape == (B, C, H, W)

    def test_prc_img_round_trip(self):
        """prc_img then unpack_latents should recover the original spatial layout."""
        from trainer.arch.flux_2.components.utils import prc_img, unpack_latents
        B, C, H, W = 1, 128, 4, 4
        x = torch.randn(B, C, H, W)
        packed, _ = prc_img(x)
        recovered = unpack_latents(packed, H, W)
        assert recovered.shape == x.shape
        # Values should match (prc_img rearranges, unpack_latents inverts)
        assert torch.allclose(x, recovered)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_discovers_flux2(self):
        from trainer.registry import list_models
        assert "flux_2" in list_models()

    def test_registry_resolves_flux2(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("flux_2")
        assert cls.__name__ == "Flux2Strategy"

    def test_flux2_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("flux_2")
        config = TrainConfig(
            model=ModelConfig(architecture="flux_2", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "flux_2"
        assert strategy.supports_video is False


# ---------------------------------------------------------------------------
# Tiny mock model
# ---------------------------------------------------------------------------

class TinyMockFlux2(nn.Module):
    """Minimal model with Flux 2 forward signature.

    Returns a single tensor (B, HW, C) - not a list like WanModel.
    """

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, in_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        x_ids: torch.Tensor,
        timesteps: torch.Tensor,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        # x: (B, HW, C) - apply a simple linear projection and return
        B, L, C = x.shape
        out = self.linear(x.float()).to(x.dtype)
        return out  # (B, HW, C)

    def enable_gradient_checkpointing(self) -> None:
        pass

    def prepare_block_swap_before_forward(self) -> None:
        pass


def _mock_setup(strategy) -> "ModelComponents":
    """Set all self._* attributes that setup() would set, return TinyMockFlux2."""
    from trainer.arch.base import ModelComponents
    from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS

    cfg = strategy.config
    flux2_config = FLUX2_CONFIGS["dev"]
    device = torch.device("cpu")

    strategy._blocks_to_swap = 0
    strategy._device = device
    strategy._train_dtype = torch.bfloat16
    strategy._flux2_config = flux2_config
    strategy._model_version = "dev"
    strategy._noise_offset_val = cfg.training.noise_offset
    dfs = cfg.training.discrete_flow_shift
    strategy._flow_shift = math.exp(dfs) if dfs != 0 else 1.0
    strategy._guidance_scale = cfg.training.guidance_scale
    strategy._ts_method = cfg.training.timestep_sampling
    strategy._ts_min = cfg.training.min_timestep
    strategy._ts_max = cfg.training.max_timestep
    strategy._ts_sigmoid_scale = cfg.training.sigmoid_scale
    strategy._ts_logit_mean = cfg.training.logit_mean
    strategy._ts_logit_std = cfg.training.logit_std

    model = TinyMockFlux2(in_channels=128).to(device)

    return ModelComponents(
        model=model,
        extra={"flux2_config": flux2_config, "model_version": "dev"},
    )


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def _make_strategy(self):
        from trainer.arch.flux_2.strategy import Flux2Strategy
        from trainer.config.schema import TrainConfig

        config = TrainConfig(
            model={"architecture": "flux_2", "base_model_path": "/fake",
                   "gradient_checkpointing": False},
            training={"method": "full_finetune", "timestep_sampling": "uniform"},
            data={"dataset_config_path": "/fake.toml"},
        )
        return Flux2Strategy(config)

    def test_training_step_produces_loss(self):
        """Feed a synthetic batch through Flux2Strategy.training_step, verify output."""
        from trainer.arch.base import TrainStepOutput

        strategy = self._make_strategy()
        components = _mock_setup(strategy)

        # Flux 2 batch: latents (B, 128, H, W), ctx_vec (B, L, D)
        B, H, W = 2, 8, 6   # 8×6 = 48 tokens
        batch = {
            "latents": torch.randn(B, 128, H, W, dtype=torch.bfloat16),
            "ctx_vec": torch.randn(B, 32, 15360, dtype=torch.bfloat16),
        }

        output = strategy.training_step(components, batch, step=0)

        assert isinstance(output, TrainStepOutput)
        assert output.loss.dim() == 0, "loss must be a scalar"
        assert torch.isfinite(output.loss), "loss must be finite"
        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics

    def test_training_step_loss_is_detached_in_metrics(self):
        """Metrics tensors must not require grad (use .detach())."""
        strategy = self._make_strategy()
        components = _mock_setup(strategy)

        B, H, W = 1, 4, 4
        batch = {
            "latents": torch.randn(B, 128, H, W, dtype=torch.bfloat16),
            "ctx_vec": torch.randn(B, 16, 15360, dtype=torch.bfloat16),
        }
        output = strategy.training_step(components, batch, step=0)

        assert not output.metrics["loss"].requires_grad
        assert not output.metrics["timestep_mean"].requires_grad

    @pytest.mark.parametrize("method", ["uniform", "sigmoid", "logit_normal", "shift"])
    def test_timestep_sampling_methods(self, method):
        """All timestep methods produce finite values in [0, 1]."""
        from trainer.arch.base import ModelStrategy

        t = ModelStrategy._sample_t(
            batch_size=1024,
            device=torch.device("cpu"),
            method=method,
        )
        assert t.min() >= 0.0 - 1e-6
        assert t.max() <= 1.0 + 1e-6
        assert torch.isfinite(t).all()

    def test_no_guidance_embed_for_klein(self):
        """Klein variants (use_guidance_embed=False) must not receive guidance tensor."""
        from trainer.arch.flux_2.strategy import Flux2Strategy
        from trainer.config.schema import TrainConfig
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS

        config = TrainConfig(
            model={
                "architecture": "flux_2",
                "base_model_path": "/fake",
                "gradient_checkpointing": False,
                "model_kwargs": {"model_version": "klein-4b"},
            },
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = Flux2Strategy(config)

        # Manually set up state for klein-4b (no guidance embed)
        flux2_config = FLUX2_CONFIGS["klein-4b"]
        device = torch.device("cpu")
        strategy._blocks_to_swap = 0
        strategy._device = device
        strategy._train_dtype = torch.bfloat16
        strategy._flux2_config = flux2_config
        strategy._model_version = "klein-4b"
        strategy._noise_offset_val = 0.0
        strategy._flow_shift = 1.0
        strategy._guidance_scale = 1.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        # Mock model that verifies guidance=None
        guidance_received = []

        class MockKleinModel(nn.Module):
            def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
                guidance_received.append(guidance)
                return x  # return same shape

        from trainer.arch.base import ModelComponents
        components = ModelComponents(model=MockKleinModel())

        B, H, W = 1, 4, 4
        batch = {
            "latents": torch.randn(B, 128, H, W, dtype=torch.bfloat16),
            "ctx_vec": torch.randn(B, 8, 7680, dtype=torch.bfloat16),
        }
        strategy.training_step(components, batch, step=0)

        assert guidance_received[0] is None, "Klein variants must pass guidance=None"
