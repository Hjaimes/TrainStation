"""Kandinsky 5 component tests.

Tests follow four tiers:
    1. Config validation — TASK_CONFIGS structure and field values.
    2. Registry — auto-discovery and strategy resolution.
    3. Strategy properties — architecture name and supports_video.
    4. Training step — synthetic forward pass producing a finite scalar loss.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Config tests
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_task_configs_exist(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        # Must have at least 8 task variants
        assert len(TASK_CONFIGS) >= 8, f"Expected >= 8 tasks, got {len(TASK_CONFIGS)}"

    def test_all_expected_keys_present(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        expected = {
            "k5-lite-t2i-hd",
            "k5-lite-i2i-hd",
            "k5-lite-t2v-5s-sd",
            "k5-lite-t2v-10s-sd",
            "k5-lite-i2v-5s-sd",
            "k5-pro-t2v-5s-sd",
            "k5-pro-t2v-5s-hd",
            "k5-pro-t2v-10s-sd",
        }
        missing = expected - set(TASK_CONFIGS.keys())
        assert not missing, f"Missing task configs: {missing}"

    def test_lite_t2v_5s_fields(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        cfg = TASK_CONFIGS["k5-lite-t2v-5s-sd"]
        assert cfg.resolution == 512
        assert cfg.dit_params.visual_cond is True
        assert cfg.dit_params.model_dim == 1792
        assert cfg.dit_params.num_visual_blocks == 32
        assert cfg.dit_params.patch_size == (1, 2, 2)
        assert cfg.vae.name == "hunyuan"

    def test_lite_t2i_hd_fields(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        cfg = TASK_CONFIGS["k5-lite-t2i-hd"]
        assert cfg.resolution == 1024
        assert cfg.dit_params.visual_cond is False
        assert cfg.dit_params.model_dim == 2560
        assert cfg.dit_params.num_visual_blocks == 50
        assert cfg.vae.name == "flux"

    def test_i2i_has_instruct_type(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        cfg = TASK_CONFIGS["k5-lite-i2i-hd"]
        assert cfg.dit_params.instruct_type == "channel"

    def test_pro_t2v_5s_sd_has_larger_model(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        cfg = TASK_CONFIGS["k5-pro-t2v-5s-sd"]
        assert cfg.dit_params.model_dim == 4096
        assert cfg.dit_params.num_visual_blocks == 60

    def test_nabla_attention_config(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        cfg = TASK_CONFIGS["k5-lite-t2v-10s-sd"]
        assert cfg.attention.type == "nabla"
        assert cfg.attention.P == 0.9
        assert cfg.attention.wT == 11
        assert cfg.attention.add_sta is True

    def test_scale_factor_shapes(self):
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        for name, cfg in TASK_CONFIGS.items():
            assert len(cfg.scale_factor) == 3, f"{name} scale_factor must have 3 elements"


# ---------------------------------------------------------------------------
# 2. Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_discovers_kandinsky5(self):
        from trainer.registry import list_models
        assert "kandinsky5" in list_models()

    def test_registry_resolves_kandinsky5(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("kandinsky5")
        assert cls.__name__ == "Kandinsky5Strategy"

    def test_kandinsky5_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig

        cls = get_model_strategy("kandinsky5")
        config = TrainConfig(
            model=ModelConfig(architecture="kandinsky5", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "kandinsky5"
        assert strategy.supports_video is True


# ---------------------------------------------------------------------------
# 3. Component unit tests
# ---------------------------------------------------------------------------

class TestAttentionEngine:
    def test_sdpa_engine_resolves(self):
        from trainer.arch.kandinsky5.components.attention import SelfAttentionEngine
        engine = SelfAttentionEngine("sdpa")
        assert engine.supports_mask is True
        assert engine.get_attention() is not None

    def test_auto_engine_resolves(self):
        from trainer.arch.kandinsky5.components.attention import SelfAttentionEngine
        engine = SelfAttentionEngine("auto")
        assert engine.get_attention() is not None

    def test_invalid_engine_raises(self):
        from trainer.arch.kandinsky5.components.attention import SelfAttentionEngine
        with pytest.raises(ValueError, match="Unknown attention engine"):
            SelfAttentionEngine("nonexistent")


class TestUtils:
    def test_get_freqs_shape(self):
        from trainer.arch.kandinsky5.components.utils import get_freqs
        freqs = get_freqs(16)
        assert freqs.shape == (16,)
        assert freqs.dtype == torch.float32

    def test_fractal_flatten_unflatten_roundtrip(self):
        from trainer.arch.kandinsky5.components.utils import fractal_flatten, fractal_unflatten
        # Without fractal (block_mask=False)
        shape = (2, 4, 4)  # F, H, W
        x = torch.randn(2, 4, 4, 8)
        rope = torch.randn(2, 4, 4, 4, 1, 2, 2)
        x_flat, rope_flat = fractal_flatten(x, rope, shape, block_mask=False)
        assert x_flat.shape[0] == 2 * 4 * 4
        x_unflat = fractal_unflatten(x_flat, shape, block_mask=False)
        assert x_unflat.shape == x.shape

    def test_fast_sta_nabla_shape(self):
        from trainer.arch.kandinsky5.components.utils import fast_sta_nabla
        T, H, W = 2, 4, 4
        sta = fast_sta_nabla(T, H, W, wT=3, wH=3, wW=3, device="cpu")
        assert sta.shape == (T * H * W, T * H * W)

    def test_fast_sta_nabla_diagonal(self):
        """Every position must attend to itself."""
        from trainer.arch.kandinsky5.components.utils import fast_sta_nabla
        sta = fast_sta_nabla(2, 4, 4, wT=3, wH=3, wW=3, device="cpu")
        diag = sta.diagonal()
        assert diag.all(), "Diagonal of STA mask must be True (self-attention)"


class TestNNModules:
    def test_time_embeddings_shape(self):
        from trainer.arch.kandinsky5.components.nn import TimeEmbeddings
        te = TimeEmbeddings(model_dim=64, time_dim=32)
        # CPU only — skip autocast
        t = torch.rand(2)
        # autocast is CUDA-only; use float32 for CPU test
        with torch.no_grad():
            # Patch autocast for CPU
            out = te.forward.__wrapped__(te, t) if hasattr(te.forward, "__wrapped__") else None
            if out is None:
                try:
                    out = te(t)
                except RuntimeError:
                    pytest.skip("TimeEmbeddings autocast requires CUDA")
        assert out.shape == (2, 32)

    def test_feed_forward_shape(self):
        from trainer.arch.kandinsky5.components.nn import FeedForward
        ff = FeedForward(dim=64, ff_dim=256)
        x = torch.randn(8, 64)
        with torch.no_grad():
            out = ff(x)
        assert out.shape == (8, 64)

    def test_modulation_zero_init(self):
        from trainer.arch.kandinsky5.components.nn import Modulation
        mod = Modulation(time_dim=32, model_dim=16, num_params=6)
        # Zero-initialised weight means output should be zero for zero input
        x = torch.zeros(1, 32)
        try:
            out = mod(x)
            assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)
        except RuntimeError:
            pytest.skip("Modulation autocast requires CUDA")

    def test_visual_embeddings_shape(self):
        from trainer.arch.kandinsky5.components.nn import VisualEmbeddings
        ve = VisualEmbeddings(visual_dim=16, model_dim=64, patch_size=(1, 2, 2))
        # (F, H, W, C) channels-last
        x = torch.randn(4, 8, 8, 16)
        with torch.no_grad():
            out = ve(x)
        # Output: (F/pT, H/pH, W/pW, model_dim) = (4, 4, 4, 64)
        assert out.shape == (4, 4, 4, 64)


# ---------------------------------------------------------------------------
# 4. Training step tests — mock model for CPU-only execution
# ---------------------------------------------------------------------------

class TinyMockKandinsky5(nn.Module):
    """Minimal mock that accepts per-sample Kandinsky5 inputs and returns a plausible output.

    Matches the DiffusionTransformer3D call signature.
    Output shape: (F, H, W, C) channels-last.
    """

    def __init__(self, out_dim: int = 16) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.visual_cond = True
        self.blocks_to_swap = None
        self.linear = nn.Linear(1, 1)  # dummy param so optimiser has something

    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        pooled_text_embed: torch.Tensor,
        time: torch.Tensor,
        visual_rope_pos,
        text_rope_pos: torch.Tensor,
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        attention_mask=None,
    ) -> torch.Tensor:
        # x: (F, H, W, C_in); return (F, H, W, C_out)
        F_, H, W, _ = x.shape
        return torch.zeros(F_, H, W, self.out_dim, device=x.device, dtype=x.dtype)

    def enable_gradient_checkpointing(self, *_):
        pass

    def prepare_block_swap_before_forward(self):
        pass

    def move_to_device_except_swap_blocks(self, *_):
        pass


def _make_config(task: str = "k5-lite-t2v-5s-sd"):
    from trainer.config.schema import TrainConfig, ModelConfig
    return TrainConfig(
        model=ModelConfig(
            architecture="kandinsky5",
            base_model_path="/fake",
            model_kwargs={"task": task},
            gradient_checkpointing=False,
        ),
        training={"method": "full_finetune"},
        data={"dataset_config_path": "/fake.toml"},
    )


def _make_batch(bsz: int = 1, n_frames: int = 4, h: int = 8, w: int = 8, channels: int = 16):
    """Synthetic batch matching what the dataset pipeline would produce."""
    return {
        # Standard layout: (B, C, F, H, W)
        "latents": torch.randn(bsz, channels, n_frames, h, w),
        # text_embeds: (B, seq_len, dim) — Qwen embedding
        "text_embeds": torch.randn(bsz, 32, 3584),
        # pooled_embed: (B, dim) — CLIP pooled
        "pooled_embed": torch.randn(bsz, 768),
    }


class TestTrainingStep:
    def test_training_step_produces_finite_loss(self):
        """Smoke test: a single forward pass produces a finite scalar loss."""
        from trainer.arch.kandinsky5.strategy import Kandinsky5Strategy

        config = _make_config()
        strategy = Kandinsky5Strategy(config)

        # Inject cached state manually (bypasses setup() which needs real weights)
        strategy._device = torch.device("cpu")
        strategy._train_dtype = torch.float32
        strategy._task = "k5-lite-t2v-5s-sd"
        strategy._patch_size = (1, 2, 2)
        strategy._patch_volume = 4
        strategy._scale_factor = (1.0, 2.0, 2.0)
        strategy._visual_cond = True
        strategy._attn_conf = type("Attn", (), {"type": "flash"})()
        strategy._nabla_mask_cache = {}
        strategy._noise_offset_val = 0.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0
        strategy._flow_shift = 1.0

        mock_model = TinyMockKandinsky5(out_dim=16)
        from trainer.arch.base import ModelComponents
        components = ModelComponents(model=mock_model)

        batch = _make_batch(bsz=1)
        output = strategy.training_step(components, batch, step=0)

        assert output.loss is not None
        assert output.loss.dim() == 0, "Loss must be a scalar tensor."
        assert math.isfinite(output.loss.item()), "Loss must be finite."
        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics

    def test_training_step_per_sample_batch(self):
        """Verify the per-sample loop works correctly with batch_size > 1."""
        from trainer.arch.kandinsky5.strategy import Kandinsky5Strategy
        from trainer.arch.base import ModelComponents

        config = _make_config()
        strategy = Kandinsky5Strategy(config)

        strategy._device = torch.device("cpu")
        strategy._train_dtype = torch.float32
        strategy._task = "k5-lite-t2v-5s-sd"
        strategy._patch_size = (1, 2, 2)
        strategy._patch_volume = 4
        strategy._scale_factor = (1.0, 2.0, 2.0)
        strategy._visual_cond = True
        strategy._attn_conf = type("Attn", (), {"type": "flash"})()
        strategy._nabla_mask_cache = {}
        strategy._noise_offset_val = 0.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0
        strategy._flow_shift = 1.0

        mock_model = TinyMockKandinsky5(out_dim=16)
        components = ModelComponents(model=mock_model)

        batch = _make_batch(bsz=3)
        output = strategy.training_step(components, batch, step=0)

        assert math.isfinite(output.loss.item()), "Loss must be finite for batch_size=3."

    def test_training_step_no_visual_cond(self):
        """Verify the training step works for image models without visual conditioning."""
        from trainer.arch.kandinsky5.strategy import Kandinsky5Strategy
        from trainer.arch.base import ModelComponents

        config = _make_config(task="k5-lite-t2i-hd")
        strategy = Kandinsky5Strategy(config)

        strategy._device = torch.device("cpu")
        strategy._train_dtype = torch.float32
        strategy._task = "k5-lite-t2i-hd"
        strategy._patch_size = (1, 2, 2)
        strategy._patch_volume = 4
        strategy._scale_factor = (1.0, 1.0, 1.0)
        strategy._visual_cond = False  # image model — no visual conditioning
        strategy._attn_conf = type("Attn", (), {"type": "flash"})()
        strategy._nabla_mask_cache = {}
        strategy._noise_offset_val = 0.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0
        strategy._flow_shift = 1.0

        # Image model: mock outputs (1, H, W, C) — no temporal dim
        class ImageMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual_cond = False
                self.blocks_to_swap = None

            def forward(self, x, text_embed, pooled_text_embed, time, visual_rope_pos, text_rope_pos, **kw):
                # x: (1, H, W, C_in) -> out (1, H, W, C_out=16)
                F_, H, W, _ = x.shape
                return torch.zeros(F_, H, W, 16, device=x.device, dtype=x.dtype)

        components = ModelComponents(model=ImageMockModel())

        # Image latents are 3-D: (B, C, H, W)
        batch = {
            "latents": torch.randn(1, 16, 8, 8),
            "text_embeds": torch.randn(1, 32, 3584),
            "pooled_embed": torch.randn(1, 768),
        }
        output = strategy.training_step(components, batch, step=0)
        assert math.isfinite(output.loss.item())

    def test_metrics_are_detached(self):
        """Metrics tensors must be detached (no grad, no GPU sync on item())."""
        from trainer.arch.kandinsky5.strategy import Kandinsky5Strategy
        from trainer.arch.base import ModelComponents

        config = _make_config()
        strategy = Kandinsky5Strategy(config)

        strategy._device = torch.device("cpu")
        strategy._train_dtype = torch.float32
        strategy._task = "k5-lite-t2v-5s-sd"
        strategy._patch_size = (1, 2, 2)
        strategy._patch_volume = 4
        strategy._scale_factor = (1.0, 2.0, 2.0)
        strategy._visual_cond = True
        strategy._attn_conf = type("Attn", (), {"type": "flash"})()
        strategy._nabla_mask_cache = {}
        strategy._noise_offset_val = 0.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0
        strategy._flow_shift = 1.0

        components = ModelComponents(model=TinyMockKandinsky5(out_dim=16))
        batch = _make_batch(bsz=1)
        output = strategy.training_step(components, batch, step=0)

        for key, val in output.metrics.items():
            assert not val.requires_grad, f"Metric '{key}' must be detached."
