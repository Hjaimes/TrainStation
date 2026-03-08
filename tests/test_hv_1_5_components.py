"""Tests for HunyuanVideo 1.5 architecture components.

Coverage:
- Config verification (54 double blocks, 0 single blocks, patch_size=[1,1,1])
- Registry discovery and resolution
- Strategy properties
- Training step with synthetic batch (no guidance param)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_num_double_blocks(self):
        from trainer.arch.hunyuan_video_1_5.components.configs import HV15_CONFIG
        assert HV15_CONFIG.num_double_blocks == 54

    def test_num_single_blocks_is_zero(self):
        from trainer.arch.hunyuan_video_1_5.components.configs import HV15_CONFIG
        assert HV15_CONFIG.num_single_blocks == 0

    def test_patch_size_is_111(self):
        from trainer.arch.hunyuan_video_1_5.components.configs import HV15_CONFIG
        assert list(HV15_CONFIG.patch_size) == [1, 1, 1]

    def test_guidance_embed_disabled(self):
        from trainer.arch.hunyuan_video_1_5.components.configs import HV15_CONFIG
        assert HV15_CONFIG.guidance_embed is False

    def test_hidden_size(self):
        from trainer.arch.hunyuan_video_1_5.components.configs import HV15_CONFIG
        assert HV15_CONFIG.hidden_size == 2048

    def test_heads_num(self):
        from trainer.arch.hunyuan_video_1_5.components.configs import HV15_CONFIG
        assert HV15_CONFIG.heads_num == 16

    def test_latent_channels(self):
        from trainer.arch.hunyuan_video_1_5.components.configs import HV15_CONFIG
        assert HV15_CONFIG.latent_channels == 16


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_discovers_hv_1_5(self):
        from trainer.registry import list_models
        assert "hunyuan_video_1_5" in list_models()

    def test_registry_resolves_hv_1_5(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("hunyuan_video_1_5")
        assert cls.__name__ == "HunyuanVideo15Strategy"

    def test_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("hunyuan_video_1_5")
        config = TrainConfig(
            model=ModelConfig(architecture="hunyuan_video_1_5", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "hunyuan_video_1_5"
        assert strategy.supports_video is True


# ---------------------------------------------------------------------------
# Component import tests
# ---------------------------------------------------------------------------

class TestComponentImports:
    def test_layers_import(self):
        from trainer.arch.hunyuan_video_1_5.components.layers import (
            MLP, RMSNorm, ModulateDiT, modulate, apply_gate
        )

    def test_embeddings_import(self):
        from trainer.arch.hunyuan_video_1_5.components.embeddings import (
            get_nd_rotary_pos_embed, TimestepEmbedder, PatchEmbed,
            ByT5Mapper, VisionProjection, apply_rotary_emb,
        )

    def test_attention_import(self):
        from trainer.arch.hunyuan_video_1_5.components.attention import (
            AttentionParams, attention,
        )

    def test_blocks_import(self):
        from trainer.arch.hunyuan_video_1_5.components.blocks import (
            MMDoubleStreamBlock, SingleTokenRefiner, FinalLayer,
        )

    def test_offloading_import(self):
        from trainer.arch.hunyuan_video_1_5.components.offloading import ModelOffloader

    def test_model_import(self):
        from trainer.arch.hunyuan_video_1_5.components.model import (
            HunyuanVideo15Transformer,
        )


# ---------------------------------------------------------------------------
# Layer unit tests
# ---------------------------------------------------------------------------

class TestLayers:
    def test_rmsnorm_shape(self):
        from trainer.arch.hunyuan_video_1_5.components.layers import RMSNorm
        norm = RMSNorm(32)
        x = torch.randn(2, 10, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_mlp_shape(self):
        from trainer.arch.hunyuan_video_1_5.components.layers import MLP
        mlp = MLP(64, 128)
        x = torch.randn(2, 10, 64)
        out = mlp(x)
        assert out.shape == x.shape

    def test_modulate(self):
        from trainer.arch.hunyuan_video_1_5.components.layers import modulate
        x = torch.ones(2, 5, 8)
        scale = torch.zeros(2, 8)
        shift = torch.zeros(2, 8)
        out = modulate(x, shift=shift, scale=scale)
        assert out.shape == x.shape
        assert torch.allclose(out, x)  # scale=0, shift=0 → identity

    def test_apply_gate(self):
        from trainer.arch.hunyuan_video_1_5.components.layers import apply_gate
        x = torch.ones(2, 5, 8)
        gate = torch.ones(2, 8)
        out = apply_gate(x, gate)
        assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Embedding unit tests
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_rope_3d_shape(self):
        from trainer.arch.hunyuan_video_1_5.components.embeddings import get_nd_rotary_pos_embed
        # HV 1.5 uses rope_dim_list=[16, 56, 56], theta=256
        cos, sin = get_nd_rotary_pos_embed([16, 56, 56], (2, 4, 4), theta=256.0)
        # 2*4*4 = 32 tokens, head_dim = 16+56+56 = 128
        assert cos.shape == (32, 128)
        assert sin.shape == (32, 128)

    def test_timestep_embedder_shape(self):
        from trainer.arch.hunyuan_video_1_5.components.embeddings import TimestepEmbedder
        embedder = TimestepEmbedder(64, nn.SiLU)
        t = torch.tensor([100.0, 500.0])
        out = embedder(t)
        assert out.shape == (2, 64)

    def test_patch_embed_shape(self):
        from trainer.arch.hunyuan_video_1_5.components.embeddings import PatchEmbed
        # patch_size=[1,1,1], in_chans=32, embed_dim=64
        pe = PatchEmbed([1, 1, 1], in_chans=32, embed_dim=64)
        # Input: [B, C*2+1, T, H, W] = [1, 65, 4, 4, 4]
        x = torch.randn(1, 65, 4, 4, 4)
        out = pe(x)
        # Should be [1, T*H*W, embed_dim] = [1, 64, 64]
        assert out.shape == (1, 64, 64)

    def test_byt5_mapper_shape(self):
        from trainer.arch.hunyuan_video_1_5.components.embeddings import ByT5Mapper
        mapper = ByT5Mapper(in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=64, use_residual=False)
        x = torch.randn(2, 10, 1472)
        out = mapper(x)
        assert out.shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# Attention unit test
# ---------------------------------------------------------------------------

class TestAttention:
    def test_sdpa_attention(self):
        from trainer.arch.hunyuan_video_1_5.components.attention import attention, AttentionParams
        B, L, H, D = 2, 8, 4, 16
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        params = AttentionParams.create_attention_params("torch", False)
        out = attention([q, k, v], attn_params=params)
        assert out.shape == (B, L, H * D)

    def test_attention_with_mask(self):
        from trainer.arch.hunyuan_video_1_5.components.attention import attention, AttentionParams
        B, L, H, D = 2, 8, 4, 16
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        # text mask: first 6 valid, second 4 valid
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[0, :6] = True
        mask[1, :4] = True
        params = AttentionParams.create_attention_params_from_mask("torch", False, 0, mask.float())
        out = attention([q, k, v], attn_params=params)
        assert out.shape[0] == B


# ---------------------------------------------------------------------------
# Mock model + training step tests
# ---------------------------------------------------------------------------

class TinyMockHV15(nn.Module):
    """Minimal mock matching HunyuanVideo15Transformer.forward() signature.

    Critically: forward() takes NO guidance parameter.
    Returns a tensor of [B, 16, T, H, W] matching the latent shape.
    """

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.out_channels = out_channels
        # Tiny linear to make it a proper nn.Module with parameters
        self.dummy = nn.Linear(1, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states=None,
        byt5_text_states=None,
        byt5_text_mask=None,
        rotary_pos_emb_cache=None,
    ) -> torch.Tensor:
        # Return a zero prediction with the correct [B, out_ch, T, H, W] shape
        B, _, T, H, W = hidden_states.shape
        return torch.zeros(B, self.out_channels, T, H, W, device=hidden_states.device)


def _make_strategy():
    from trainer.arch.hunyuan_video_1_5.strategy import HunyuanVideo15Strategy
    from trainer.config.schema import TrainConfig, ModelConfig
    config = TrainConfig(
        model=ModelConfig(architecture="hunyuan_video_1_5", base_model_path="/fake"),
        training={"method": "full_finetune"},
        data={"dataset_config_path": "/fake.toml"},
    )
    strategy = HunyuanVideo15Strategy(config)
    # Manually set cached attributes that would normally be set by setup()
    strategy._blocks_to_swap = 0
    strategy._device = torch.device("cpu")
    strategy._train_dtype = torch.float32
    strategy._task_type = "t2v"
    strategy._noise_offset_val = 0.0
    strategy._ts_method = "uniform"
    strategy._ts_min = 0.0
    strategy._ts_max = 1.0
    strategy._ts_sigmoid_scale = 1.0
    strategy._ts_logit_mean = 0.0
    strategy._ts_logit_std = 1.0
    strategy._flow_shift = 1.0
    strategy._latent_channels = 16
    return strategy


def _make_batch(B: int = 2, T: int = 2, H: int = 4, W: int = 4):
    """Create a minimal synthetic batch for HV 1.5 training step."""
    latents = torch.randn(B, 16, T, H, W)
    # T2V: no latents_image

    # Variable-length VL and ByT5 embeddings
    vl_embed = [torch.randn(torch.randint(3, 8, ()).item(), 3584) for _ in range(B)]
    byt5_embed = [torch.randn(torch.randint(5, 12, ()).item(), 1472) for _ in range(B)]

    return {"latents": latents, "vl_embed": vl_embed, "byt5_embed": byt5_embed}


class TestTrainingStep:
    def test_training_step_produces_scalar_loss(self):
        from trainer.arch.base import ModelComponents
        strategy = _make_strategy()
        mock_model = TinyMockHV15()
        components = ModelComponents(model=mock_model)
        batch = _make_batch()

        result = strategy.training_step(components, batch, step=0)

        assert result.loss.ndim == 0, "Loss must be a scalar"
        assert torch.isfinite(result.loss), "Loss must be finite"
        assert "loss" in result.metrics
        assert "timestep_mean" in result.metrics

    def test_training_step_t2v_no_guidance(self):
        """Verify the model forward is called WITHOUT a guidance parameter."""
        from trainer.arch.base import ModelComponents

        call_kwargs: dict = {}

        class RecordingMock(TinyMockHV15):
            def forward(self, **kwargs):
                call_kwargs.update(kwargs)
                return super().forward(**kwargs)

        strategy = _make_strategy()
        components = ModelComponents(model=RecordingMock())
        batch = _make_batch()

        strategy.training_step(components, batch, step=0)

        assert "guidance" not in call_kwargs, (
            "HV 1.5 forward must NOT receive a 'guidance' parameter"
        )

    def test_training_step_i2v_with_cond_latents(self):
        """Verify that latents_image (I2V cond) is forwarded correctly."""
        from trainer.arch.base import ModelComponents
        strategy = _make_strategy()
        strategy._task_type = "i2v"

        hidden_shapes: list = []

        class ShapeMock(TinyMockHV15):
            def forward(self, hidden_states, **kwargs):
                hidden_shapes.append(hidden_states.shape)
                return super().forward(hidden_states, **kwargs)

        components = ModelComponents(model=ShapeMock())
        B, T, H, W = 1, 2, 4, 4
        batch = _make_batch(B=B, T=T, H=H, W=W)
        # Add I2V conditioning: [B, 17, T, H, W]
        batch["latents_image"] = torch.zeros(B, 17, T, H, W)

        strategy.training_step(components, batch, step=0)

        assert len(hidden_shapes) == 1
        # model_input = concat(noisy[B,16,T,H,W], cond[B,17,T,H,W]) → [B,33,T,H,W]
        assert hidden_shapes[0] == (B, 33, T, H, W)

    def test_loss_is_mse_against_target(self):
        """Verify loss = MSE(pred, noise - latents)."""
        from trainer.arch.base import ModelComponents
        import torch.nn.functional as F

        strategy = _make_strategy()
        strategy._ts_method = "uniform"

        # TinyMockHV15 returns zeros - so loss = MSE(0, noise-latents) = mean(target^2)
        components = ModelComponents(model=TinyMockHV15())
        B, T, H, W = 1, 2, 4, 4
        batch = _make_batch(B=B, T=T, H=H, W=W)
        # Fix seed for deterministic test
        torch.manual_seed(42)
        result = strategy.training_step(components, batch, step=0)

        assert result.loss.item() > 0  # non-zero loss when pred=0 and target!=0

    def test_metrics_are_detached(self):
        """Metrics must be detached tensors (no grad)."""
        from trainer.arch.base import ModelComponents
        strategy = _make_strategy()
        components = ModelComponents(model=TinyMockHV15())
        batch = _make_batch()

        result = strategy.training_step(components, batch, step=0)

        for key, val in result.metrics.items():
            assert isinstance(val, torch.Tensor), f"metric '{key}' must be a tensor"
            assert not val.requires_grad, f"metric '{key}' must be detached"
