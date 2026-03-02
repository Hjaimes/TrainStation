"""QwenImage architecture tests.

Follows the same pattern as tests/test_wan_components.py.

Test levels:
  - Configs: verify preset keys and expected field values
  - Registry: auto-discovery and class resolution
  - TrainingStep: mock end-to-end forward pass with tiny model
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_all_mode_keys_present(self):
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        expected = {"t2i", "edit", "edit-2511", "layered"}
        assert set(QWEN_IMAGE_CONFIGS.keys()) == expected

    def test_t2i_fields(self):
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        cfg = QWEN_IMAGE_CONFIGS["t2i"]
        assert cfg.mode == "t2i"
        assert cfg.patch_size == 2
        assert cfg.in_channels == 64
        assert cfg.out_channels == 16
        assert cfg.num_layers == 60
        assert cfg.attention_head_dim == 128
        assert cfg.num_attention_heads == 24
        assert cfg.joint_attention_dim == 3584
        assert cfg.vae_scale_factor == 8
        assert cfg.latent_channels == 16
        assert cfg.zero_cond_t is False
        assert cfg.use_additional_t_cond is False
        assert cfg.use_layer3d_rope is False

    def test_edit_fields(self):
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        cfg = QWEN_IMAGE_CONFIGS["edit"]
        assert cfg.mode == "edit"
        assert cfg.zero_cond_t is False
        assert cfg.use_additional_t_cond is False
        assert cfg.use_layer3d_rope is False

    def test_edit_2511_fields(self):
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        cfg = QWEN_IMAGE_CONFIGS["edit-2511"]
        assert cfg.mode == "edit"
        assert cfg.zero_cond_t is True
        assert cfg.use_additional_t_cond is False
        assert cfg.use_layer3d_rope is False

    def test_layered_fields(self):
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        cfg = QWEN_IMAGE_CONFIGS["layered"]
        assert cfg.mode == "layered"
        assert cfg.zero_cond_t is False
        assert cfg.use_additional_t_cond is True
        assert cfg.use_layer3d_rope is True

    def test_axes_dims_rope(self):
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        # All modes share same RoPE axes
        for key, cfg in QWEN_IMAGE_CONFIGS.items():
            assert cfg.axes_dims_rope == (16, 56, 56), f"axes_dims_rope mismatch for mode '{key}'"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_discovers_qwen_image(self):
        from trainer.registry import list_models
        assert "qwen_image" in list_models()

    def test_registry_resolves_qwen_image(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("qwen_image")
        assert cls.__name__ == "QwenImageStrategy"

    def test_qwen_image_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig

        cls = get_model_strategy("qwen_image")
        config = TrainConfig(
            model=ModelConfig(architecture="qwen_image", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "qwen_image"
        assert strategy.supports_video is False


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------

class TestModules:
    def test_get_activation_silu(self):
        from trainer.arch.qwen_image.components.modules import get_activation
        act = get_activation("silu")
        assert isinstance(act, nn.SiLU)

    def test_get_activation_gelu(self):
        from trainer.arch.qwen_image.components.modules import get_activation
        act = get_activation("gelu")
        assert isinstance(act, nn.GELU)

    def test_get_activation_unknown_raises(self):
        from trainer.arch.qwen_image.components.modules import get_activation
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("badact")

    def test_rms_norm_forward(self):
        from trainer.arch.qwen_image.components.modules import RMSNorm
        norm = RMSNorm(64, eps=1e-6)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_feed_forward_forward(self):
        from trainer.arch.qwen_image.components.modules import FeedForward
        ff = FeedForward(dim=64, dim_out=64, activation_fn="gelu-approximate")
        x = torch.randn(2, 16, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_timestep_embedding_forward(self):
        from trainer.arch.qwen_image.components.modules import TimestepEmbedding
        emb = TimestepEmbedding(in_channels=256, time_embed_dim=512)
        x = torch.randn(4, 256)
        out = emb(x)
        assert out.shape == (4, 512)

    def test_qwen_embed_rope_forward(self):
        from trainer.arch.qwen_image.components.modules import QwenEmbedRope
        rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        device = torch.device("cpu")
        img_shapes = [[(1, 4, 4)]]  # 1 frame, 4×4 patch grid
        txt_seq_lens = [8]
        vid_freqs, txt_freqs = rope(img_shapes, txt_seq_lens, device)
        assert vid_freqs.shape[0] == 4 * 4  # 16 tokens
        assert txt_freqs.shape[0] == 8

    def test_qwen_embed_layer3d_rope_forward(self):
        from trainer.arch.qwen_image.components.modules import QwenEmbedLayer3DRope
        rope = QwenEmbedLayer3DRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        device = torch.device("cpu")
        # Two segments: layer 0 (target) + condition image
        img_shapes = [[(1, 4, 4), (1, 4, 4)]]
        txt_seq_lens = [8]
        vid_freqs, txt_freqs = rope(img_shapes, txt_seq_lens, device)
        # 2 entries × 16 tokens each = 32
        assert vid_freqs.shape[0] == 32
        assert txt_freqs.shape[0] == 8


# ---------------------------------------------------------------------------
# Latent pack/unpack tests
# ---------------------------------------------------------------------------

class TestLatentPacking:
    def test_pack_latents_single_frame(self):
        """Single frame: [B, C, 1, H, W] -> [B, H/2*W/2, C*4] via 2×2 pixel-unshuffle."""
        from trainer.arch.qwen_image.strategy import pack_latents
        latents = torch.randn(2, 16, 1, 8, 8)  # B, C, F=1, H, W
        packed = pack_latents(latents)
        # S = (H/2) * (W/2) = 4*4 = 16, C_packed = 16*4 = 64
        assert packed.shape == (2, 16, 64)  # B, S=(4*4), C_patch=64

    def test_pack_latents_layered(self):
        """Layered: [B, L, C, H, W] -> [B, L*H/2*W/2, C*4]."""
        from trainer.arch.qwen_image.strategy import pack_latents
        latents = torch.randn(1, 3, 16, 8, 8)  # B, L=3, C, H, W
        packed = pack_latents(latents)
        # S = 3 * (4*4) = 48, C_patch = 64
        assert packed.shape == (1, 48, 64)

    def test_pack_unpack_roundtrip(self):
        """Pack then unpack should recover the original spatial layout."""
        from trainer.arch.qwen_image.strategy import pack_latents, unpack_latents
        B, C, H, W = 1, 16, 8, 8
        latents = torch.randn(B, C, 1, H, W)
        packed = pack_latents(latents)
        # patch_h = H//2=4, patch_w = W//2=4
        unpacked = unpack_latents(packed, lat_h=4, lat_w=4, num_frames=1)
        assert unpacked.shape == (B, C, 1, H, W)
        assert torch.allclose(unpacked, latents, atol=1e-6)


# ---------------------------------------------------------------------------
# Tiny mock model for training_step test
# ---------------------------------------------------------------------------

class TinyMockQwenImage(nn.Module):
    """Tiny mock that matches QwenImageTransformer2DModel's forward signature.

    Input:  hidden_states [B, S, in_channels=64]  (packed, patchified, 2×2 pixel-unshuffle)
    Output: [B, S, patch_size²*out_channels=64]    (same packed shape for unpack by strategy)

    The real model: proj_out produces patch_size² × out_channels = 4 × 16 = 64 channels.
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 16  # latent channels
        # proj produces patch_size² * out_channels = 4 * 16 = 64 per token
        self.proj = nn.Linear(in_channels, 64)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.Tensor = None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance: torch.Tensor = None,
        attention_kwargs=None,
        additional_t_cond=None,
    ) -> torch.Tensor:
        # Produce [B, S, 64] (= patch_size² × out_channels)
        return self.proj(hidden_states)


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def _make_strategy(self):
        """Build a QwenImageStrategy with a minimal config (no model file needed)."""
        from trainer.arch.qwen_image.strategy import QwenImageStrategy
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        from trainer.config.schema import TrainConfig, ModelConfig

        config = TrainConfig(
            model=ModelConfig(architecture="qwen_image", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = QwenImageStrategy(config)

        # Manually set all _* attributes that setup() would set
        qwen_cfg = QWEN_IMAGE_CONFIGS["t2i"]
        device = torch.device("cpu")

        strategy._blocks_to_swap = 0
        strategy._device = device
        strategy._train_dtype = torch.float32
        strategy._qwen_config = qwen_cfg
        strategy._mode = "t2i"
        strategy._is_edit = False
        strategy._is_layered = False
        strategy._use_additional_t_cond = False
        strategy._noise_offset_val = 0.0
        strategy._flow_shift = 1.0
        strategy._gradient_checkpointing = False
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        return strategy

    def test_training_step_produces_loss(self):
        """Verify the training step runs end-to-end and returns valid loss."""
        strategy = self._make_strategy()

        # Tiny model: 4×4 latent grid, 1 frame
        B = 2
        LAT_H, LAT_W = 4, 4
        C = 16

        mock_model = TinyMockQwenImage(in_channels=64)
        components = type("MockComponents", (), {"model": mock_model})()

        # Build batch: latents [B, C, 1, H, W], vl_embed list of [S, D]
        latents = torch.randn(B, C, 1, LAT_H, LAT_W)
        vl_embed = [torch.randn(12, 3584) for _ in range(B)]  # variable length

        batch = {
            "latents": latents,
            "vl_embed": vl_embed,
        }

        output = strategy.training_step(components, batch, step=0)

        assert hasattr(output, "loss")
        assert output.loss.shape == ()  # scalar
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)
        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics

    def test_training_step_edit_mode(self):
        """Edit mode: control latents appended and output trimmed correctly."""
        from trainer.arch.qwen_image.strategy import QwenImageStrategy
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        from trainer.config.schema import TrainConfig, ModelConfig

        config = TrainConfig(
            model=ModelConfig(architecture="qwen_image", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = QwenImageStrategy(config)
        qwen_cfg = QWEN_IMAGE_CONFIGS["edit"]
        device = torch.device("cpu")

        strategy._blocks_to_swap = 0
        strategy._device = device
        strategy._train_dtype = torch.float32
        strategy._qwen_config = qwen_cfg
        strategy._mode = "edit"
        strategy._is_edit = True
        strategy._is_layered = False
        strategy._use_additional_t_cond = False
        strategy._noise_offset_val = 0.0
        strategy._flow_shift = 1.0
        strategy._gradient_checkpointing = False
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        B, C, LAT_H, LAT_W = 1, 16, 4, 4
        mock_model = TinyMockQwenImage(in_channels=64)
        components = type("MockComponents", (), {"model": mock_model})()

        batch = {
            "latents": torch.randn(B, C, 1, LAT_H, LAT_W),
            "vl_embed": [torch.randn(10, 3584)],
            "latents_control_0": torch.randn(B, C, 1, LAT_H, LAT_W),
        }

        output = strategy.training_step(components, batch, step=0)
        assert not torch.isnan(output.loss)

    def test_training_step_returns_detached_metrics(self):
        """Metrics should be detached tensors (no GPU sync)."""
        strategy = self._make_strategy()
        B, C, LAT_H, LAT_W = 1, 16, 4, 4
        mock_model = TinyMockQwenImage(in_channels=64)
        components = type("MockComponents", (), {"model": mock_model})()

        batch = {
            "latents": torch.randn(B, C, 1, LAT_H, LAT_W),
            "vl_embed": [torch.randn(8, 3584)],
        }
        output = strategy.training_step(components, batch, step=0)
        assert not output.metrics["loss"].requires_grad
        assert not output.metrics["timestep_mean"].requires_grad
