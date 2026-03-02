"""Tests for FramePack architecture components.

Test levels (fast → slow):
    1. Import checks — modules load without errors
    2. Config validation — expected fields and values present
    3. Registry — framepack discovered and resolves correctly
    4. Strategy properties — architecture name, supports_video
    5. Blocks — forward pass through individual layers with tiny tensors
    6. Utils — temporal packing helpers produce correct shapes
    7. Training step — mock model produces scalar finite loss
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Import checks
# ---------------------------------------------------------------------------

class TestImports:
    def test_configs_importable(self):
        from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIGS, FRAMEPACK_CONFIG
        assert FRAMEPACK_CONFIGS is not None
        assert FRAMEPACK_CONFIG is not None

    def test_blocks_importable(self):
        from trainer.arch.framepack.components.blocks import (
            HunyuanVideoTransformerBlock,
            HunyuanVideoSingleTransformerBlock,
            HunyuanVideoTokenRefiner,
            LayerNormFramePack,
            RMSNormFramePack,
        )

    def test_utils_importable(self):
        from trainer.arch.framepack.components.utils import (
            HunyuanVideoRotaryPosEmbed,
            ClipVisionProjection,
            HunyuanVideoPatchEmbed,
            HunyuanVideoPatchEmbedForCleanLatents,
            get_cu_seqlens,
            pad_for_3d_conv,
            center_down_sample_3d,
            crop_or_pad_yield_mask,
        )

    def test_model_importable(self):
        from trainer.arch.framepack.components.model import HunyuanVideoTransformer3DModelPacked

    def test_strategy_importable(self):
        from trainer.arch.framepack.strategy import FramePackStrategy


# ---------------------------------------------------------------------------
# 2. Config validation
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_config_keys_present(self):
        from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIGS
        assert "framepack" in FRAMEPACK_CONFIGS

    def test_framepack_config_fields(self):
        from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIG
        assert FRAMEPACK_CONFIG.num_attention_heads == 24
        assert FRAMEPACK_CONFIG.attention_head_dim == 128
        assert FRAMEPACK_CONFIG.inner_dim == 24 * 128  # 3072
        assert FRAMEPACK_CONFIG.num_layers == 20
        assert FRAMEPACK_CONFIG.num_single_layers == 40
        assert FRAMEPACK_CONFIG.num_refiner_layers == 2
        assert FRAMEPACK_CONFIG.patch_size == 2
        assert FRAMEPACK_CONFIG.patch_size_t == 1
        assert FRAMEPACK_CONFIG.in_channels == 16
        assert FRAMEPACK_CONFIG.text_embed_dim == 4096
        assert FRAMEPACK_CONFIG.pooled_projection_dim == 768
        assert FRAMEPACK_CONFIG.image_proj_dim == 1152
        assert FRAMEPACK_CONFIG.rope_axes_dim == (16, 56, 56)
        assert FRAMEPACK_CONFIG.rope_theta == 256.0

    def test_framepack_config_vae_dtype(self):
        from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIG
        assert FRAMEPACK_CONFIG.vae_dtype == "float16"
        assert FRAMEPACK_CONFIG.dit_dtype == "bfloat16"

    def test_framepack_config_temporal_packing(self):
        from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIG
        assert FRAMEPACK_CONFIG.clean_latents_1x_count == 1
        assert FRAMEPACK_CONFIG.clean_latents_2x_count == 2
        assert FRAMEPACK_CONFIG.clean_latents_4x_count == 16
        assert FRAMEPACK_CONFIG.latent_window_size == 9

    def test_framepack_config_architecture_name(self):
        from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIG
        assert FRAMEPACK_CONFIG.architecture == "framepack"


# ---------------------------------------------------------------------------
# 3. Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_discovers_framepack(self):
        from trainer.registry import list_models
        assert "framepack" in list_models()

    def test_registry_resolves_framepack(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("framepack")
        assert cls.__name__ == "FramePackStrategy"

    def test_framepack_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig, ModelConfig
        cls = get_model_strategy("framepack")
        config = TrainConfig(
            model=ModelConfig(architecture="framepack", base_model_path="/fake"),
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = cls(config)
        assert strategy.architecture == "framepack"
        assert strategy.supports_video is True


# ---------------------------------------------------------------------------
# 4. Individual block tests (tiny tensors, CPU only)
# ---------------------------------------------------------------------------

class TestBlocks:
    """Verify each block can do a forward pass with tiny synthetic inputs."""

    @pytest.fixture(autouse=True)
    def cpu_only(self):
        """Ensure tests run on CPU regardless of environment."""
        pass

    def test_layer_norm_framepack(self):
        from trainer.arch.framepack.components.blocks import LayerNormFramePack
        norm = LayerNormFramePack(8)
        x = torch.randn(2, 4, 8)
        out = norm(x)
        assert out.shape == x.shape

    def test_rms_norm_framepack(self):
        from trainer.arch.framepack.components.blocks import RMSNormFramePack
        norm = RMSNormFramePack(8, eps=1e-6)
        x = torch.randn(2, 4, 8)
        out = norm(x)
        assert out.shape == x.shape

    def test_feedforward_gelu(self):
        from trainer.arch.framepack.components.blocks import FeedForward
        ff = FeedForward(dim=16, mult=2, activation_fn="gelu-approximate")
        x = torch.randn(2, 4, 16)
        out = ff(x)
        assert out.shape == x.shape

    def test_feedforward_linear_silu(self):
        from trainer.arch.framepack.components.blocks import FeedForward
        ff = FeedForward(dim=16, mult=2, activation_fn="linear-silu")
        x = torch.randn(2, 4, 16)
        out = ff(x)
        assert out.shape == x.shape

    def test_combined_timestep_guidance_proj(self):
        from trainer.arch.framepack.components.blocks import CombinedTimestepGuidanceTextProjEmbeddings
        embed = CombinedTimestepGuidanceTextProjEmbeddings(embedding_dim=32, pooled_projection_dim=16)
        timestep = torch.tensor([500.0])
        guidance = torch.tensor([10000.0])
        pooled = torch.randn(1, 16)
        out = embed(timestep, guidance, pooled)
        assert out.shape == (1, 32)

    def test_token_refiner(self):
        from trainer.arch.framepack.components.blocks import HunyuanVideoTokenRefiner
        # tiny dimensions to keep test fast
        refiner = HunyuanVideoTokenRefiner(
            in_channels=8,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
        )
        hidden_states = torch.randn(1, 6, 8)  # [B, L_text, text_dim]
        timestep = torch.tensor([500.0])
        attention_mask = torch.ones(1, 6, dtype=torch.long)
        out = refiner(hidden_states, timestep, attention_mask)
        assert out.shape == (1, 6, 2 * 4)  # [B, L_text, heads*head_dim]


# ---------------------------------------------------------------------------
# 5. Utils tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_pad_for_3d_conv(self):
        from trainer.arch.framepack.components.utils import pad_for_3d_conv
        x = torch.randn(1, 16, 3, 5, 5)
        padded = pad_for_3d_conv(x, kernel_size=(2, 4, 4))
        # T must be divisible by 2: ceil(3/2)*2 = 4
        assert padded.shape[2] == 4
        # H/W must be divisible by 4: ceil(5/4)*4 = 8
        assert padded.shape[3] == 8
        assert padded.shape[4] == 8

    def test_center_down_sample_3d(self):
        from trainer.arch.framepack.components.utils import center_down_sample_3d
        x = torch.randn(1, 4, 4, 8, 8)
        ds = center_down_sample_3d(x, kernel_size=(2, 2, 2))
        assert ds.shape == (1, 4, 2, 4, 4)

    def test_rope_embed_shape(self):
        from trainer.arch.framepack.components.utils import HunyuanVideoRotaryPosEmbed
        rope = HunyuanVideoRotaryPosEmbed(rope_dim=(4, 8, 8), theta=256.0)
        frame_indices = torch.arange(3).unsqueeze(0)  # [1, 3]
        result = rope(frame_indices=frame_indices, height=4, width=4, device=torch.device("cpu"))
        # Shape: [B, 2*(DT+DY+DX), T, H, W]
        expected_dim = 2 * (4 + 8 + 8)  # 40
        assert result.shape == (1, expected_dim, 3, 4, 4)

    def test_get_cu_seqlens(self):
        from trainer.arch.framepack.components.utils import get_cu_seqlens
        # batch=2, text_len max=4, img_len=9
        text_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int32)
        cu_seqlens, seq_len = get_cu_seqlens(text_mask, img_len=9)
        assert cu_seqlens.shape == (5,)  # 2*B+1
        assert seq_len[0].item() == 3 + 9  # 12
        assert seq_len[1].item() == 2 + 9  # 11

    def test_crop_or_pad_yield_mask_pad(self):
        from trainer.arch.framepack.components.utils import crop_or_pad_yield_mask
        x = torch.randn(1, 4, 8)
        padded, mask = crop_or_pad_yield_mask(x, length=8)
        assert padded.shape == (1, 8, 8)
        assert mask.shape == (1, 8)
        assert mask[0, :4].all()
        assert not mask[0, 4:].any()

    def test_crop_or_pad_yield_mask_crop(self):
        from trainer.arch.framepack.components.utils import crop_or_pad_yield_mask
        x = torch.randn(1, 10, 8)
        cropped, mask = crop_or_pad_yield_mask(x, length=6)
        assert cropped.shape == (1, 6, 8)
        assert mask.all()

    def test_clip_vision_projection(self):
        from trainer.arch.framepack.components.utils import ClipVisionProjection
        proj = ClipVisionProjection(in_channels=16, out_channels=32)
        x = torch.randn(1, 4, 16)
        out = proj(x)
        assert out.shape == (1, 4, 32)

    def test_patch_embed_shape(self):
        from trainer.arch.framepack.components.utils import HunyuanVideoPatchEmbed
        embed = HunyuanVideoPatchEmbed(patch_size=(1, 2, 2), in_chans=16, embed_dim=32)
        x = torch.randn(1, 16, 4, 8, 8)
        out = embed(x)
        # T stays same (patch_t=1), H/W halved (patch=2)
        assert out.shape == (1, 32, 4, 4, 4)


# ---------------------------------------------------------------------------
# 6. Mock model + training step
# ---------------------------------------------------------------------------

class TinyMockFramePack(nn.Module):
    """Minimal mock that matches HunyuanVideoTransformer3DModelPacked's forward signature.

    Returns a tensor matching the input latent shape without any real computation.
    Used exclusively to test the training step logic without loading weights.
    """

    def __init__(self, in_channels: int = 16):
        super().__init__()
        self.in_channels = in_channels
        # Dummy parameter so the module has something to train
        self.dummy = nn.Parameter(torch.zeros(1))

    def enable_gradient_checkpointing(self) -> None:
        pass

    def enable_block_swap(self, *args, **kwargs) -> None:
        pass

    def move_to_device_except_swap_blocks(self, device) -> None:
        pass

    def prepare_block_swap_before_forward(self) -> None:
        pass

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
        image_embeddings=None,
        return_dict=True,
        **kwargs,
    ):
        # Return noise-shaped tensor: same shape as hidden_states
        out = hidden_states * self.dummy + hidden_states.detach()
        if return_dict:
            from types import SimpleNamespace
            return SimpleNamespace(sample=out)
        return (out,)


class TestTrainingStep:
    """Test FramePackStrategy.training_step with a mock model."""

    def _make_strategy(self):
        from trainer.arch.framepack.strategy import FramePackStrategy
        from trainer.config.schema import TrainConfig, ModelConfig

        config = TrainConfig(
            model=ModelConfig(
                architecture="framepack",
                base_model_path="/fake/model.safetensors",
                gradient_checkpointing=False,
                block_swap_count=0,
            ),
            training={
                "method": "full_finetune",
                "timestep_sampling": "uniform",
                "noise_offset": 0.0,
            },
            data={"dataset_config_path": "/fake.toml"},
        )
        return FramePackStrategy(config)

    def _make_batch(self, device: torch.device, dtype: torch.dtype):
        """Build a synthetic FramePack training batch."""
        B, C, T, H, W = 1, 16, 3, 8, 8
        T1 = 1   # 1x clean context frames
        T2 = 2   # 2x clean context frames
        T4 = 4   # 4x clean context frames (reduced for speed)
        L_text = 8
        L_img = 4  # SigLIP tokens

        return {
            "latents": torch.randn(B, C, T, H, W, device=device, dtype=dtype),
            "latent_indices": torch.arange(T).unsqueeze(0).to(device),
            "latents_clean": torch.randn(B, C, T1, H, W, device=device, dtype=dtype),
            "clean_latent_indices": torch.zeros(B, T1, dtype=torch.long, device=device),
            "latents_clean_2x": torch.randn(B, C, T2, H, W, device=device, dtype=dtype),
            "clean_latent_2x_indices": torch.zeros(B, T2, dtype=torch.long, device=device),
            "latents_clean_4x": torch.randn(B, C, T4, H, W, device=device, dtype=dtype),
            "clean_latent_4x_indices": torch.zeros(B, T4, dtype=torch.long, device=device),
            "llama_vec": torch.randn(B, L_text, 4096, device=device, dtype=dtype),
            "llama_attention_mask": torch.ones(B, L_text, dtype=torch.long, device=device),
            "clip_l_pooler": torch.randn(B, 768, device=device, dtype=dtype),
            "image_embeddings": torch.randn(B, L_img, 1152, device=device, dtype=dtype),
        }

    def test_training_step_produces_loss(self):
        """End-to-end: strategy.training_step returns scalar finite loss."""
        from trainer.arch.framepack.strategy import FramePackStrategy
        from trainer.arch.base import ModelComponents

        strategy = self._make_strategy()

        # Manually inject cached state (normally set by setup())
        device = torch.device("cpu")
        strategy._device = device
        strategy._train_dtype = torch.float32
        strategy._blocks_to_swap = 0
        strategy._guidance_scale = 10.0
        strategy._noise_offset_val = 0.0
        strategy._flow_shift = 1.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        mock_model = TinyMockFramePack(in_channels=16)
        components = ModelComponents(model=mock_model)

        batch = self._make_batch(device=device, dtype=torch.float32)

        output = strategy.training_step(components, batch, step=0)

        assert output.loss is not None
        assert output.loss.shape == ()  # scalar
        assert torch.isfinite(output.loss), f"Loss is not finite: {output.loss}"
        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics
        # Metrics should be detached (no grad)
        assert not output.metrics["loss"].requires_grad
        assert not output.metrics["timestep_mean"].requires_grad

    def test_training_step_without_multiscale_latents(self):
        """Training step works when 2x/4x latents are absent from batch."""
        from trainer.arch.framepack.strategy import FramePackStrategy
        from trainer.arch.base import ModelComponents

        strategy = self._make_strategy()
        device = torch.device("cpu")
        strategy._device = device
        strategy._train_dtype = torch.float32
        strategy._blocks_to_swap = 0
        strategy._guidance_scale = 10.0
        strategy._noise_offset_val = 0.0
        strategy._flow_shift = 1.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        mock_model = TinyMockFramePack(in_channels=16)
        components = ModelComponents(model=mock_model)

        B, C, T, H, W = 1, 16, 3, 8, 8
        # Only include 1x clean latents — no 2x or 4x
        batch = {
            "latents": torch.randn(B, C, T, H, W),
            "latent_indices": torch.arange(T).unsqueeze(0),
            "latents_clean": torch.randn(B, C, 1, H, W),
            "clean_latent_indices": torch.zeros(B, 1, dtype=torch.long),
            "llama_vec": torch.randn(B, 8, 4096),
            "llama_attention_mask": torch.ones(B, 8, dtype=torch.long),
            "clip_l_pooler": torch.randn(B, 768),
            "image_embeddings": torch.randn(B, 4, 1152),
        }

        output = strategy.training_step(components, batch, step=1)
        assert torch.isfinite(output.loss)

    def test_training_step_with_noise_offset(self):
        """Noise offset runs without error."""
        from trainer.arch.framepack.strategy import FramePackStrategy
        from trainer.arch.base import ModelComponents

        strategy = self._make_strategy()
        device = torch.device("cpu")
        strategy._device = device
        strategy._train_dtype = torch.float32
        strategy._blocks_to_swap = 0
        strategy._guidance_scale = 10.0
        strategy._noise_offset_val = 0.05  # non-zero
        strategy._flow_shift = 1.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        mock_model = TinyMockFramePack(in_channels=16)
        components = ModelComponents(model=mock_model)
        batch = self._make_batch(device=device, dtype=torch.float32)
        output = strategy.training_step(components, batch, step=2)
        assert torch.isfinite(output.loss)

    def test_timestep_sampling_methods(self):
        """All timestep sampling methods execute without error."""
        from trainer.arch.base import ModelStrategy

        device = torch.device("cpu")
        for method in ["uniform", "sigmoid", "logit_normal", "shift"]:
            t = ModelStrategy._sample_t(
                batch_size=4, device=device, method=method
            )
            assert t.shape == (4,)
            assert (t >= 0.0).all() and (t <= 1.0).all()

    def test_unknown_timestep_method_raises(self):
        from trainer.arch.base import ModelStrategy

        with pytest.raises(ValueError, match="Unknown timestep sampling method"):
            ModelStrategy._sample_t(4, torch.device("cpu"), method="bogus")
