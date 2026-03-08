"""Tests for HunyuanVideo architecture components.

Test levels:
  1. Import check - all modules import without error
  2. Config validation - correct block counts and dimensions
  3. Registry - hunyuan_video is discovered and resolves correctly
  4. Strategy properties - architecture name, supports_video
  5. Training step - mock forward produces scalar finite loss
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Import smoke tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_configs_import(self):
        from trainer.arch.hunyuan_video.components.configs import HunyuanVideoConfig, HUNYUAN_VIDEO_CONFIG
        assert isinstance(HUNYUAN_VIDEO_CONFIG, HunyuanVideoConfig)

    def test_layers_import(self):
        from trainer.arch.hunyuan_video.components.layers import (
            MLP, MLPEmbedder, FinalLayer, ModulateDiT,
            RMSNorm, get_activation_layer, get_norm_layer, modulate,
        )

    def test_attention_import(self):
        from trainer.arch.hunyuan_video.components.attention import attention, get_cu_seqlens

    def test_embeddings_import(self):
        from trainer.arch.hunyuan_video.components.embeddings import (
            TimestepEmbedder, PatchEmbed, TextProjection,
            SingleTokenRefiner, apply_rotary_emb,
            get_rotary_pos_embed_by_shape, get_nd_rotary_pos_embed,
        )

    def test_blocks_import(self):
        from trainer.arch.hunyuan_video.components.blocks import (
            MMDoubleStreamBlock, MMSingleStreamBlock,
        )

    def test_offloading_import(self):
        from trainer.arch.hunyuan_video.components.offloading import ModelOffloader

    def test_model_import(self):
        from trainer.arch.hunyuan_video.components.model import HunyuanVideoTransformer3DModel

    def test_strategy_import(self):
        from trainer.arch.hunyuan_video.strategy import HunyuanVideoStrategy

    def test_vae_import(self):
        from trainer.arch.hunyuan_video.components.vae import VAE_SCALING_FACTOR, VAE_VER


# ---------------------------------------------------------------------------
# 2. Config validation
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_default_block_counts(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        assert HUNYUAN_VIDEO_CONFIG.mm_double_blocks_depth == 20
        assert HUNYUAN_VIDEO_CONFIG.mm_single_blocks_depth == 40

    def test_rope_dim_list_sums_to_head_dim(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        cfg = HUNYUAN_VIDEO_CONFIG
        head_dim = cfg.hidden_size // cfg.heads_num
        assert sum(cfg.rope_dim_list) == head_dim, (
            f"rope_dim_list {cfg.rope_dim_list} sum {sum(cfg.rope_dim_list)} != head_dim {head_dim}"
        )

    def test_rope_dim_list_values(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        assert HUNYUAN_VIDEO_CONFIG.rope_dim_list == [16, 56, 56]

    def test_patch_size(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        assert HUNYUAN_VIDEO_CONFIG.patch_size == [1, 2, 2]

    def test_latent_channels(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        assert HUNYUAN_VIDEO_CONFIG.in_channels == 16
        assert HUNYUAN_VIDEO_CONFIG.out_channels == 16

    def test_guidance_embed_enabled(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        assert HUNYUAN_VIDEO_CONFIG.guidance_embed is True

    def test_text_dims(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        assert HUNYUAN_VIDEO_CONFIG.text_states_dim == 4096
        assert HUNYUAN_VIDEO_CONFIG.text_states_dim_2 == 768

    def test_hidden_size_heads(self):
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        assert HUNYUAN_VIDEO_CONFIG.hidden_size == 3072
        assert HUNYUAN_VIDEO_CONFIG.heads_num == 24


# ---------------------------------------------------------------------------
# 3. Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_discovers_hunyuan_video(self):
        from trainer.registry import list_models
        assert "hunyuan_video" in list_models()

    def test_registry_resolves_hunyuan_video(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("hunyuan_video")
        assert cls.__name__ == "HunyuanVideoStrategy"

    def test_strategy_architecture_property(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("hunyuan_video")
        config = TrainConfig(
            model=ModelConfig(architecture="hunyuan_video", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "hunyuan_video"

    def test_strategy_supports_video(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("hunyuan_video")
        config = TrainConfig(
            model=ModelConfig(architecture="hunyuan_video", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.supports_video is True


# ---------------------------------------------------------------------------
# 4. Layers unit tests
# ---------------------------------------------------------------------------

class TestLayers:
    def test_rms_norm(self):
        from trainer.arch.hunyuan_video.components.layers import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_mlp_embedder(self):
        from trainer.arch.hunyuan_video.components.layers import MLPEmbedder
        emb = MLPEmbedder(768, 128)
        x = torch.randn(2, 768)
        out = emb(x)
        assert out.shape == (2, 128)

    def test_modulate_dit(self):
        from trainer.arch.hunyuan_video.components.layers import ModulateDiT, get_activation_layer
        mod = ModulateDiT(64, factor=6, act_layer=get_activation_layer("silu"))
        x = torch.randn(2, 64)
        out = mod(x)
        assert out.shape == (2, 64 * 6)

    def test_modulate_fn(self):
        from trainer.arch.hunyuan_video.components.layers import modulate
        x = torch.randn(2, 8, 64)
        shift = torch.zeros(2, 64)
        scale = torch.zeros(2, 64)
        out = modulate(x, shift=shift, scale=scale)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 5. Embeddings unit tests
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_timestep_embedder(self):
        from trainer.arch.hunyuan_video.components.embeddings import TimestepEmbedder, get_activation_layer
        emb = TimestepEmbedder(128, get_activation_layer("silu"))
        t = torch.tensor([100.0, 500.0])
        out = emb(t)
        assert out.shape == (2, 128)

    def test_patch_embed_shape(self):
        from trainer.arch.hunyuan_video.components.embeddings import PatchEmbed
        # patch_size = [1, 2, 2], input = [B, 16, 4, 8, 8]
        embed = PatchEmbed(patch_size=[1, 2, 2], in_chans=16, embed_dim=64)
        x = torch.randn(1, 16, 4, 8, 8)
        out = embed(x)
        # T=4//1=4, H=8//2=4, W=8//2=4 => S = 4*4*4 = 64
        assert out.shape == (1, 64, 64)

    def test_rope_embedding_shape(self):
        from trainer.arch.hunyuan_video.components.embeddings import get_rotary_pos_embed_by_shape
        freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(
            patch_size=[1, 2, 2],
            hidden_size=128,
            heads_num=4,
            rope_dim_list=[8, 16, 8],  # sums to 32 = 128//4
            rope_theta=256.0,
            latents_size=[4, 8, 8],
        )
        # S = (4//1) * (8//2) * (8//2) = 4 * 4 * 4 = 64
        assert freqs_cos.shape == (64, 32)
        assert freqs_sin.shape == (64, 32)

    def test_apply_rotary_emb(self):
        from trainer.arch.hunyuan_video.components.embeddings import apply_rotary_emb, get_rotary_pos_embed_by_shape
        freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(
            patch_size=[1, 2, 2],
            hidden_size=128,
            heads_num=4,
            rope_dim_list=[8, 16, 8],
            rope_theta=256.0,
            latents_size=[4, 8, 8],
        )
        B, S, H, D = 1, 64, 4, 32
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        q_out, k_out = apply_rotary_emb(q, k, (freqs_cos, freqs_sin), head_first=False)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_single_token_refiner_shape(self):
        from trainer.arch.hunyuan_video.components.embeddings import SingleTokenRefiner
        refiner = SingleTokenRefiner(in_channels=64, hidden_size=64, heads_num=4, depth=1)
        x = torch.randn(2, 8, 64)    # [B, S_text, C_text]
        t = torch.tensor([100.0, 500.0])
        mask = torch.ones(2, 8, dtype=torch.bool)
        out = refiner(x, t, mask)
        assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# 6. Blocks unit tests
# ---------------------------------------------------------------------------

class TestBlocks:
    """Test MMDoubleStreamBlock and MMSingleStreamBlock with torch attention
    (no flash_attn dependency required for testing)."""

    # Small dimensions for fast CPU tests
    HIDDEN = 64
    HEADS = 4
    MLP_RATIO = 2.0
    B = 1
    IMG_LEN = 16
    TXT_LEN = 8

    def _make_cu_seqlens(self, batch_size: int, img_len: int, txt_len: int) -> torch.Tensor:
        """Build cu_seqlens for single-item batch (no varlen - same lengths)."""
        total = img_len + txt_len
        cu = torch.zeros(2 * batch_size + 1, dtype=torch.int32)
        for i in range(batch_size):
            cu[2 * i + 1] = (i * total) + total
            cu[2 * i + 2] = (i + 1) * total
        return cu

    def test_double_stream_block_shape(self):
        from trainer.arch.hunyuan_video.components.blocks import MMDoubleStreamBlock
        block = MMDoubleStreamBlock(
            hidden_size=self.HIDDEN,
            heads_num=self.HEADS,
            mlp_width_ratio=self.MLP_RATIO,
            attn_mode="torch",
            split_attn=True,  # split avoids cu_seqlens requirement
        )
        img = torch.randn(self.B, self.IMG_LEN, self.HIDDEN)
        txt = torch.randn(self.B, self.TXT_LEN, self.HIDDEN)
        vec = torch.randn(self.B, self.HIDDEN)
        img_out, txt_out = block(img, txt, vec)
        assert img_out.shape == img.shape
        assert txt_out.shape == txt.shape

    def test_single_stream_block_shape(self):
        from trainer.arch.hunyuan_video.components.blocks import MMSingleStreamBlock
        block = MMSingleStreamBlock(
            hidden_size=self.HIDDEN,
            heads_num=self.HEADS,
            mlp_width_ratio=self.MLP_RATIO,
            attn_mode="torch",
            split_attn=True,
        )
        x = torch.randn(self.B, self.IMG_LEN + self.TXT_LEN, self.HIDDEN)
        vec = torch.randn(self.B, self.HIDDEN)
        out = block(x, vec, self.TXT_LEN)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 7. Tiny mock model + training step
# ---------------------------------------------------------------------------

class TinyMockHunyuanVideo(nn.Module):
    """Minimal mock that mimics HunyuanVideoTransformer3DModel's forward signature."""

    def __init__(self, in_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: torch.Tensor = None,
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        guidance: torch.Tensor = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        return x + self._dummy * 0.0  # same shape as input, always finite


class TestTrainingStep:
    """Integration test: HunyuanVideoStrategy.training_step with a mock model."""

    B = 1
    F = 4   # video frames (latent)
    H = 8   # latent height
    W = 8   # latent width
    C = 16  # latent channels
    S_TXT = 8  # text sequence length

    def _make_batch(self) -> dict[str, torch.Tensor]:
        return {
            "latents": torch.randn(self.B, self.C, self.F, self.H, self.W),
            "text_states": torch.randn(self.B, self.S_TXT, 4096),
            "text_mask": torch.ones(self.B, self.S_TXT, dtype=torch.bool),
            "text_states_2": torch.randn(self.B, 768),
        }

    def _make_strategy(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("hunyuan_video")
        config = TrainConfig(
            model=ModelConfig(architecture="hunyuan_video", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        return strategy

    def _setup_strategy_with_mock(self, strategy):
        """Bypass real model loading by directly populating cached strategy state."""
        from trainer.arch.base import ModelComponents
        from trainer.arch.hunyuan_video.components.configs import HUNYUAN_VIDEO_CONFIG
        import math

        # Replicate what setup() would cache
        device = torch.device("cpu")
        strategy._blocks_to_swap = 0
        strategy._device = device
        strategy._train_dtype = torch.float32
        strategy._hv_config = HUNYUAN_VIDEO_CONFIG
        strategy._patch_size = HUNYUAN_VIDEO_CONFIG.patch_size
        strategy._guidance_scale = 1.0
        strategy._noise_offset_val = 0.0
        strategy._flow_shift = 1.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        mock_model = TinyMockHunyuanVideo(in_channels=self.C)
        components = ModelComponents(
            model=mock_model,
            extra={"hv_config": HUNYUAN_VIDEO_CONFIG},
        )
        return components

    def test_training_step_produces_loss(self):
        strategy = self._make_strategy()
        components = self._setup_strategy_with_mock(strategy)
        batch = self._make_batch()

        output = strategy.training_step(components, batch, step=0)

        assert output.loss is not None
        assert output.loss.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(output.loss), f"Loss is not finite: {output.loss}"

    def test_training_step_metrics(self):
        strategy = self._make_strategy()
        components = self._setup_strategy_with_mock(strategy)
        batch = self._make_batch()

        output = strategy.training_step(components, batch, step=0)

        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics
        assert torch.isfinite(output.metrics["loss"])
        assert torch.isfinite(output.metrics["timestep_mean"])

    def test_training_step_multiple_timestep_methods(self):
        """Verify all supported timestep sampling methods produce valid losses."""
        for method in ["uniform", "sigmoid", "logit_normal", "shift"]:
            strategy = self._make_strategy()
            components = self._setup_strategy_with_mock(strategy)
            strategy._ts_method = method
            batch = self._make_batch()
            output = strategy.training_step(components, batch, step=0)
            assert torch.isfinite(output.loss), f"Loss not finite for method={method}"
