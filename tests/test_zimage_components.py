"""Z-Image architecture component tests.

Tests follow the same pattern as test_wan_components.py:
  TestConfigs      - config constants and registry dict
  TestRegistry     - decorator-based registry discovery
  TinyMockZImage   - minimal mock matching ZImageTransformer2DModel forward sig
  TestTrainingStep - strategy.training_step() with mock model, synthetic batch
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_constants_present(self):
        """Core VAE constants must be present and have correct values."""
        from trainer.arch.zimage.components.configs import (
            ZIMAGE_VAE_SHIFT_FACTOR,
            ZIMAGE_VAE_SCALING_FACTOR,
            ZIMAGE_VAE_LATENT_CHANNELS,
            SEQ_MULTI_OF,
        )
        assert ZIMAGE_VAE_SHIFT_FACTOR == pytest.approx(0.1159)
        assert ZIMAGE_VAE_SCALING_FACTOR == pytest.approx(0.3611)
        assert ZIMAGE_VAE_LATENT_CHANNELS == 16
        assert SEQ_MULTI_OF == 32

    def test_default_config_fields(self):
        """ZIMAGE_DEFAULT_CONFIG has expected architecture fields."""
        from trainer.arch.zimage.components.configs import ZIMAGE_DEFAULT_CONFIG
        assert ZIMAGE_DEFAULT_CONFIG.in_channels == 16
        assert ZIMAGE_DEFAULT_CONFIG.dim == 3840
        assert ZIMAGE_DEFAULT_CONFIG.n_layers == 30
        assert ZIMAGE_DEFAULT_CONFIG.n_heads == 30
        assert ZIMAGE_DEFAULT_CONFIG.n_kv_heads == 30
        assert ZIMAGE_DEFAULT_CONFIG.cap_feat_dim == 2560
        # patch_size is a tuple of ints
        assert ZIMAGE_DEFAULT_CONFIG.patch_size == (2,)
        assert ZIMAGE_DEFAULT_CONFIG.f_patch_size == (1,)

    def test_configs_registry_has_base(self):
        """ZIMAGE_CONFIGS dict has at least one entry."""
        from trainer.arch.zimage.components.configs import ZIMAGE_CONFIGS
        assert "zimage-base" in ZIMAGE_CONFIGS

    def test_rope_dims(self):
        """ROPE_AXES_DIMS sum should equal head_dim (dim / n_heads)."""
        from trainer.arch.zimage.components.configs import (
            ZIMAGE_DEFAULT_CONFIG, ROPE_AXES_DIMS,
        )
        head_dim = ZIMAGE_DEFAULT_CONFIG.dim // ZIMAGE_DEFAULT_CONFIG.n_heads
        assert sum(ROPE_AXES_DIMS) == head_dim


# ---------------------------------------------------------------------------
# VAE normalization tests (no model weights needed)
# ---------------------------------------------------------------------------

class TestVAENormalization:
    def test_normalize_shape_preserved(self):
        from trainer.arch.zimage.components.vae import normalize_latents
        x = torch.randn(2, 16, 32, 32)
        out = normalize_latents(x)
        assert out.shape == x.shape

    def test_normalize_denormalize_roundtrip(self):
        """normalize then denormalize should recover the original (up to float precision)."""
        from trainer.arch.zimage.components.vae import (
            normalize_latents, denormalize_latents,
        )
        x = torch.randn(1, 16, 64, 64)
        roundtripped = denormalize_latents(normalize_latents(x))
        assert torch.allclose(roundtripped, x, atol=1e-5)

    def test_normalize_changes_values(self):
        """normalize_latents should change values (not identity)."""
        from trainer.arch.zimage.components.vae import (
            normalize_latents,
            ZIMAGE_VAE_SHIFT_FACTOR,
            ZIMAGE_VAE_SCALING_FACTOR,
        )
        x = torch.zeros(1, 16, 4, 4)  # all zeros -> shift only
        out = normalize_latents(x)
        expected = (0.0 - ZIMAGE_VAE_SHIFT_FACTOR) * ZIMAGE_VAE_SCALING_FACTOR
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-6)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_discovers_zimage(self):
        from trainer.registry import list_models
        assert "zimage" in list_models()

    def test_registry_resolves_zimage(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("zimage")
        assert cls.__name__ == "ZImageStrategy"

    def test_zimage_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("zimage")
        config = TrainConfig(
            model=ModelConfig(architecture="zimage", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "zimage"
        assert strategy.supports_video is False


# ---------------------------------------------------------------------------
# Tiny mock model - same forward signature as ZImageTransformer2DModel
# ---------------------------------------------------------------------------

class TinyMockZImage(nn.Module):
    """Minimal model matching ZImageTransformer2DModel's forward signature.

    ZImageTransformer2DModel.forward(x, t, cap_feats, cap_mask) -> Tensor [B, C, F, H, W]
    (Unlike WanModel which returns List[Tensor])
    """

    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.linear = nn.Linear(in_channels, in_channels, bias=False)
        # Initialize to near-identity so loss is non-trivial but finite
        nn.init.eye_(self.linear.weight)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor | None = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> torch.Tensor:
        # x: [B, C, F, H, W]
        # Returns: [B, C, F, H, W]
        B, C, F, H, W = x.shape
        flat = x.reshape(B, C, -1).permute(0, 2, 1)   # [B, F*H*W, C]
        out = self.linear(flat.float()).permute(0, 2, 1)  # [B, C, F*H*W]
        return out.reshape(B, C, F, H, W)

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        pass

    def disable_gradient_checkpointing(self):
        pass

    def enable_block_swap(self, *args, **kwargs):
        pass

    def move_to_device_except_swap_blocks(self, device: torch.device):
        pass

    def prepare_block_swap_before_forward(self):
        pass


# ---------------------------------------------------------------------------
# Mock setup helper - sets ALL self._* attributes that setup() would set
# ---------------------------------------------------------------------------

def _mock_setup_zimage(strategy) -> "ModelComponents":
    """Bypass strategy.setup() and inject a TinyMockZImage + cached attributes."""
    from trainer.arch.base import ModelComponents
    from trainer.arch.zimage.components.configs import ZIMAGE_DEFAULT_CONFIG

    cfg = strategy.config
    device = torch.device("cpu")
    train_dtype = torch.float32   # Use fp32 on CPU for test stability

    dfs = cfg.training.discrete_flow_shift
    flow_shift = math.exp(dfs) if dfs != 0 else 1.0

    # Exactly mirrors what ZImageStrategy.setup() caches
    strategy._blocks_to_swap = 0
    strategy._device = device
    strategy._train_dtype = train_dtype
    strategy._zimage_config = ZIMAGE_DEFAULT_CONFIG
    strategy._patch_size = ZIMAGE_DEFAULT_CONFIG.patch_size[0]  # 2
    strategy._split_attn = False
    strategy._noise_offset_val = cfg.training.noise_offset
    strategy._use_gradient_checkpointing = False
    strategy._flow_shift = flow_shift
    strategy._ts_method = cfg.training.timestep_sampling
    strategy._ts_min = cfg.training.min_timestep
    strategy._ts_max = cfg.training.max_timestep
    strategy._ts_sigmoid_scale = cfg.training.sigmoid_scale
    strategy._ts_logit_mean = cfg.training.logit_mean
    strategy._ts_logit_std = cfg.training.logit_std

    model = TinyMockZImage(in_channels=16).to(device)

    return ModelComponents(
        model=model,
        extra={
            "zimage_config": ZIMAGE_DEFAULT_CONFIG,
            "patch_size": 2,
        },
    )


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def _make_config(self, **training_kwargs):
        from trainer.config.schema import TrainConfig
        return TrainConfig(
            model={"architecture": "zimage", "base_model_path": "/fake",
                   "gradient_checkpointing": False},
            training={"method": "full_finetune", **training_kwargs},
            data={"dataset_config_path": "/fake.toml"},
        )

    def test_training_step_produces_loss(self):
        """Basic smoke test: training_step returns finite scalar loss."""
        from trainer.arch.zimage.strategy import ZImageStrategy
        from trainer.arch.base import TrainStepOutput

        config = self._make_config(timestep_sampling="uniform")
        strategy = ZImageStrategy(config)
        components = _mock_setup_zimage(strategy)

        # Synthetic batch - image only (4D latents [B, C, H, W])
        batch = {
            "latents": torch.randn(2, 16, 32, 32),   # [B, C, H, W]
            "llm_embed": [
                torch.randn(64, 2560),
                torch.randn(48, 2560),
            ],
        }

        output = strategy.training_step(components, batch, step=0)

        assert isinstance(output, TrainStepOutput)
        assert output.loss.dim() == 0            # scalar
        assert torch.isfinite(output.loss)
        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics

    def test_training_step_with_5d_latents(self):
        """If latents already have frame dim [B, C, 1, H, W], should still work."""
        from trainer.arch.zimage.strategy import ZImageStrategy

        config = self._make_config(timestep_sampling="uniform")
        strategy = ZImageStrategy(config)
        components = _mock_setup_zimage(strategy)

        batch = {
            "latents": torch.randn(1, 16, 1, 16, 16),  # [B, C, F, H, W]
            "llm_embed": [torch.randn(32, 2560)],
        }

        output = strategy.training_step(components, batch, step=0)
        assert torch.isfinite(output.loss)

    def test_training_step_with_stacked_embeds(self):
        """When llm_embed is already stacked [B, L, D], should handle it."""
        from trainer.arch.zimage.strategy import ZImageStrategy

        config = self._make_config()
        strategy = ZImageStrategy(config)
        components = _mock_setup_zimage(strategy)

        batch = {
            "latents": torch.randn(2, 16, 32, 32),
            "llm_embed": torch.randn(2, 64, 2560),   # pre-stacked
            "llm_mask": torch.ones(2, 64, dtype=torch.bool),
        }

        output = strategy.training_step(components, batch, step=0)
        assert torch.isfinite(output.loss)

    def test_metrics_are_detached(self):
        """Metrics tensors should be detached (no grad) for logging efficiency."""
        from trainer.arch.zimage.strategy import ZImageStrategy

        config = self._make_config()
        strategy = ZImageStrategy(config)
        components = _mock_setup_zimage(strategy)

        batch = {
            "latents": torch.randn(1, 16, 16, 16, requires_grad=False),
            "llm_embed": [torch.randn(20, 2560)],
        }

        output = strategy.training_step(components, batch, step=0)
        assert not output.metrics["loss"].requires_grad
        assert not output.metrics["timestep_mean"].requires_grad

    def test_reversed_timestep_range(self):
        """Sample raw t and verify the reversed t_model is in [0, 1]."""
        from trainer.arch.base import ModelStrategy

        t = ModelStrategy._sample_t(
            batch_size=1000,
            device=torch.device("cpu"),
            method="uniform",
        )
        # ZImage scaling: timesteps = t * 1000 + 1
        timesteps = t * 1000.0 + 1.0
        # t_model = (1000 - timesteps) / 1000 must be in [0, 1]
        t_model = (1000.0 - timesteps) / 1000.0
        assert t_model.min() >= -1e-3
        assert t_model.max() <= 1.0 + 1e-4

    @pytest.mark.parametrize("method", ["uniform", "sigmoid", "logit_normal", "shift"])
    def test_timestep_sampling_methods(self, method):
        """All sampling methods should return finite t in [0, 1]."""
        from trainer.arch.base import ModelStrategy

        t = ModelStrategy._sample_t(
            batch_size=500,
            device=torch.device("cpu"),
            method=method,
        )
        assert t.min() >= 0.0 - 1e-6
        assert t.max() <= 1.0 + 1e-6
        assert torch.isfinite(t).all()

    def test_noise_offset_applied(self):
        """With noise_offset > 0, training_step should still complete."""
        from trainer.arch.zimage.strategy import ZImageStrategy

        config = self._make_config()
        strategy = ZImageStrategy(config)
        components = _mock_setup_zimage(strategy)
        # Override noise offset
        strategy._noise_offset_val = 0.05

        batch = {
            "latents": torch.randn(2, 16, 32, 32),
            "llm_embed": [torch.randn(40, 2560), torch.randn(40, 2560)],
        }

        output = strategy.training_step(components, batch, step=0)
        assert torch.isfinite(output.loss)

    def test_seq_padding_multiple_of_32(self):
        """With split_attn=False and batch>1, text seq len should be padded to SEQ_MULTI_OF."""
        from trainer.arch.zimage.strategy import ZImageStrategy
        from trainer.arch.zimage.components.configs import SEQ_MULTI_OF

        config = self._make_config()
        strategy = ZImageStrategy(config)
        components = _mock_setup_zimage(strategy)
        # Ensure not split_attn
        strategy._split_attn = False

        # 4x4 latent / patch_size=2 -> image_seq_len = (4//2)*(4//2) = 4
        # With text len=10: total=14, must pad to next multiple of 32 -> 32
        # So text should be padded to 32 - 4 = 28 tokens
        captured_cap_feats = []

        original_forward = components.model.forward

        def capture_forward(x, t, cap_feats, cap_mask=None, **kwargs):
            captured_cap_feats.append(cap_feats.shape)
            return original_forward(x, t, cap_feats, cap_mask, **kwargs)

        components.model.forward = capture_forward

        batch = {
            "latents": torch.randn(2, 16, 4, 4),    # H=4, W=4, patch=2 -> seq=4
            "llm_embed": [
                torch.randn(10, 2560),
                torch.randn(10, 2560),
            ],
        }

        strategy.training_step(components, batch, step=0)

        if captured_cap_feats:
            _, L, _ = captured_cap_feats[0]
            total = 4 + L  # image_seq + text_seq
            assert total % SEQ_MULTI_OF == 0, (
                f"Total seq len {total} is not a multiple of {SEQ_MULTI_OF}"
            )
