"""Tests for Flux 1 architecture components.

All tests run on CPU with tiny synthetic tensors - no real weights required.
Follows the pattern of tests/test_flux2_components.py.
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
    def test_configs_exist(self):
        """FLUX1_CONFIGS must have 'dev' and 'schnell' variants."""
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS
        assert "dev" in FLUX1_CONFIGS
        assert "schnell" in FLUX1_CONFIGS

    def test_config_frozen(self):
        """Flux1Config is a frozen dataclass - must raise on mutation."""
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS
        cfg = FLUX1_CONFIGS["dev"]
        with pytest.raises((AttributeError, TypeError)):
            cfg.hidden_size = 999  # type: ignore[misc]

    def test_dev_has_guidance(self):
        """Dev variant must have use_guidance_embed=True."""
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS
        assert FLUX1_CONFIGS["dev"].use_guidance_embed is True

    def test_schnell_no_guidance(self):
        """Schnell variant must have use_guidance_embed=False."""
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS
        assert FLUX1_CONFIGS["schnell"].use_guidance_embed is False

    def test_dimensions(self):
        """Core dimensions must match the Flux 1 spec."""
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS
        for variant_name in ("dev", "schnell"):
            cfg = FLUX1_CONFIGS[variant_name]
            assert cfg.hidden_size == 3072, f"{variant_name}: hidden_size"
            assert cfg.num_attention_heads == 24, f"{variant_name}: num_attention_heads"
            assert cfg.head_dim == 128, f"{variant_name}: head_dim"
            assert cfg.num_double_blocks == 19, f"{variant_name}: num_double_blocks"
            assert cfg.num_single_blocks == 38, f"{variant_name}: num_single_blocks"
            assert cfg.in_channels == 64, f"{variant_name}: in_channels (16*2*2)"
            assert cfg.latent_channels == 16, f"{variant_name}: latent_channels"
            assert cfg.context_dim == 4096, f"{variant_name}: context_dim (T5-XXL)"
            assert cfg.pooled_dim == 768, f"{variant_name}: pooled_dim (CLIP-L)"
            assert cfg.rope_axes == (16, 56, 56), f"{variant_name}: rope_axes"
            # Verify rope_axes sum to head_dim
            assert sum(cfg.rope_axes) == cfg.head_dim, (
                f"{variant_name}: rope_axes sum {sum(cfg.rope_axes)} != head_dim {cfg.head_dim}"
            )

    def test_activation_is_geglu(self):
        """Both variants must use GEGLU activation (not SiLU)."""
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS
        for cfg in FLUX1_CONFIGS.values():
            assert cfg.activation == "geglu"


# ---------------------------------------------------------------------------
# 2. Utils tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_pack_unpack_roundtrip(self):
        """pack_latents then unpack_latents must recover original tensor exactly."""
        from trainer.arch.flux_1.components.utils import pack_latents, unpack_latents
        B, C, H, W = 2, 16, 8, 8
        x = torch.randn(B, C, H, W)
        packed = pack_latents(x)
        h_packed, w_packed = H // 2, W // 2
        recovered = unpack_latents(packed, h_packed, w_packed)
        assert recovered.shape == x.shape
        assert torch.allclose(x, recovered), "round-trip must be lossless"

    def test_pack_shape(self):
        """(1, 16, 8, 8) -> (1, 16, 64): seq_len=16=(8/2)*(8/2), channels=64=16*2*2."""
        from trainer.arch.flux_1.components.utils import pack_latents
        x = torch.randn(1, 16, 8, 8)
        packed = pack_latents(x)
        assert packed.shape == (1, 16, 64), (
            f"Expected (1, 16, 64), got {packed.shape}. "
            "seq_len = (8/2)*(8/2) = 16, channels = 16*2*2 = 64"
        )

    def test_img_ids_shape(self):
        """prepare_img_ids(4, 4) must return (1, 16, 3)."""
        from trainer.arch.flux_1.components.utils import prepare_img_ids
        ids = prepare_img_ids(4, 4)
        assert ids.shape == (1, 16, 3), f"Expected (1, 16, 3), got {ids.shape}"

    def test_txt_ids_shape(self):
        """prepare_txt_ids(77) must return (1, 77, 3)."""
        from trainer.arch.flux_1.components.utils import prepare_txt_ids
        ids = prepare_txt_ids(77)
        assert ids.shape == (1, 77, 3), f"Expected (1, 77, 3), got {ids.shape}"

    def test_txt_ids_all_zeros(self):
        """Text position IDs must be all zeros (no spatial positions for text)."""
        from trainer.arch.flux_1.components.utils import prepare_txt_ids
        ids = prepare_txt_ids(20)
        assert ids.sum().item() == 0.0, "Text IDs must be all zeros"

    def test_img_ids_3d_format(self):
        """Image IDs have 3 dimensions: [channel=0, y, x]."""
        from trainer.arch.flux_1.components.utils import prepare_img_ids
        h, w = 3, 4
        ids = prepare_img_ids(h, w)  # (1, 12, 3)
        # Channel dim should always be 0
        assert (ids[..., 0] == 0).all(), "channel dimension must be 0"
        # Y positions should be in [0, h-1]
        assert ids[..., 1].max().item() == h - 1
        # X positions should be in [0, w-1]
        assert ids[..., 2].max().item() == w - 1

    def test_pack_non_square(self):
        """Pack/unpack must work for non-square spatial dimensions."""
        from trainer.arch.flux_1.components.utils import pack_latents, unpack_latents
        B, C, H, W = 1, 16, 8, 12
        x = torch.randn(B, C, H, W)
        packed = pack_latents(x)
        h_packed, w_packed = H // 2, W // 2
        expected_seq = h_packed * w_packed
        assert packed.shape == (B, expected_seq, 64)
        recovered = unpack_latents(packed, h_packed, w_packed)
        assert torch.allclose(x, recovered)


# ---------------------------------------------------------------------------
# 3. Embedding tests
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_rope_output_shape(self):
        """Flux1RoPE with axes (4, 14, 14) on positions (1, 10, 3) -> (1, 10, 32)."""
        from trainer.arch.flux_1.components.embeddings import Flux1RoPE
        rope = Flux1RoPE(axes_dim=(4, 14, 14))
        positions = torch.zeros(1, 10, 3)  # (B, N, 3)
        out = rope(positions)
        assert out.shape == (1, 10, 32), f"Expected (1, 10, 32), got {out.shape}"

    def test_rope_axes_sum_matches_dim(self):
        """Output last dim must equal sum of axes_dim."""
        from trainer.arch.flux_1.components.embeddings import Flux1RoPE
        axes = (8, 16, 8)
        rope = Flux1RoPE(axes_dim=axes)
        pos = torch.randn(2, 5, 3)
        out = rope(pos)
        assert out.shape[-1] == sum(axes)

    def test_timestep_embedding_shape(self):
        """TimestepEmbedding with hidden=64: input (2,) -> (2, 64)."""
        from trainer.arch.flux_1.components.embeddings import TimestepEmbedding
        hidden = 64
        embed = TimestepEmbedding(hidden_size=hidden)
        t = torch.rand(2)
        out = embed(t)
        assert out.shape == (2, hidden), f"Expected (2, {hidden}), got {out.shape}"

    def test_guidance_embedding_shape(self):
        """GuidanceEmbedding with hidden=64: input (2,) -> (2, 64)."""
        from trainer.arch.flux_1.components.embeddings import GuidanceEmbedding
        hidden = 64
        embed = GuidanceEmbedding(hidden_size=hidden)
        g = torch.rand(2)
        out = embed(g)
        assert out.shape == (2, hidden), f"Expected (2, {hidden}), got {out.shape}"

    def test_mlp_embedder_shape(self):
        """MLPEmbedder with in_dim=768, hidden=64: input (2, 768) -> (2, 64)."""
        from trainer.arch.flux_1.components.embeddings import MLPEmbedder
        embedder = MLPEmbedder(in_dim=768, hidden_size=64)
        x = torch.randn(2, 768)
        out = embedder(x)
        assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"

    def test_rope_output_finite(self):
        """RoPE output must be finite for all-zero positions."""
        from trainer.arch.flux_1.components.embeddings import Flux1RoPE
        rope = Flux1RoPE(axes_dim=(16, 56, 56))
        positions = torch.zeros(1, 32, 3)
        out = rope(positions)
        assert torch.isfinite(out).all(), "RoPE must produce finite values"


# ---------------------------------------------------------------------------
# 4. Block tests (tiny dimensions)
# ---------------------------------------------------------------------------

TINY_HIDDEN = 64
TINY_HEADS = 2
TINY_HEAD_DIM = TINY_HIDDEN // TINY_HEADS  # 32
TINY_MLP_RATIO = 2.0


class TestBlocks:
    def _make_rope(self, seq_len: int, hidden: int = TINY_HIDDEN, heads: int = TINY_HEADS) -> torch.Tensor:
        """Create synthetic RoPE tensor for tests: (1, seq_len, head_dim)."""
        head_dim = hidden // heads
        return torch.randn(1, seq_len, head_dim)

    def test_double_stream_block_names(self):
        """Block class must be named Flux1DoubleStreamBlock (not DoubleStreamBlock)."""
        from trainer.arch.flux_1.components.blocks import Flux1DoubleStreamBlock
        assert Flux1DoubleStreamBlock.__name__ == "Flux1DoubleStreamBlock"

    def test_single_stream_block_names(self):
        """Block class must be named Flux1SingleStreamBlock (not SingleStreamBlock)."""
        from trainer.arch.flux_1.components.blocks import Flux1SingleStreamBlock
        assert Flux1SingleStreamBlock.__name__ == "Flux1SingleStreamBlock"

    def test_double_stream_block_forward(self):
        """Flux1DoubleStreamBlock: img+txt shapes preserved through forward pass."""
        from trainer.arch.flux_1.components.blocks import Flux1DoubleStreamBlock
        B, img_len, txt_len = 2, 16, 10
        block = Flux1DoubleStreamBlock(
            hidden_size=TINY_HIDDEN,
            num_heads=TINY_HEADS,
            mlp_ratio=TINY_MLP_RATIO,
        )
        img = torch.randn(B, img_len, TINY_HIDDEN)
        txt = torch.randn(B, txt_len, TINY_HIDDEN)
        vec = torch.randn(B, TINY_HIDDEN)
        img_rope = self._make_rope(img_len).expand(B, -1, -1)
        txt_rope = self._make_rope(txt_len).expand(B, -1, -1)

        with torch.no_grad():
            img_out, txt_out = block(img, txt, vec, img_rope, txt_rope)

        assert img_out.shape == (B, img_len, TINY_HIDDEN), f"img shape mismatch: {img_out.shape}"
        assert txt_out.shape == (B, txt_len, TINY_HIDDEN), f"txt shape mismatch: {txt_out.shape}"
        assert torch.isfinite(img_out).all(), "img output must be finite"
        assert torch.isfinite(txt_out).all(), "txt output must be finite"

    def test_single_stream_block_forward(self):
        """Flux1SingleStreamBlock: combined shape preserved through forward pass."""
        from trainer.arch.flux_1.components.blocks import Flux1SingleStreamBlock
        B, combined_len = 2, 26
        block = Flux1SingleStreamBlock(
            hidden_size=TINY_HIDDEN,
            num_heads=TINY_HEADS,
            mlp_ratio=TINY_MLP_RATIO,
        )
        x = torch.randn(B, combined_len, TINY_HIDDEN)
        vec = torch.randn(B, TINY_HIDDEN)
        rope = self._make_rope(combined_len).expand(B, -1, -1)

        with torch.no_grad():
            out = block(x, vec, rope)

        assert out.shape == (B, combined_len, TINY_HIDDEN), f"output shape mismatch: {out.shape}"
        assert torch.isfinite(out).all(), "output must be finite"

    def test_geglu_activation(self):
        """GEGLU must halve the hidden dim: (B, L, 2*D) -> (B, L, D)."""
        from trainer.arch.flux_1.components.blocks import _GEGLUActivation
        act = _GEGLUActivation()
        D = 32
        x = torch.randn(2, 10, 2 * D)
        out = act(x)
        assert out.shape == (2, 10, D), f"Expected (2, 10, {D}), got {out.shape}"

    def test_geglu_is_not_silu(self):
        """GEGLU should produce different results from a plain SiLU gate."""
        from trainer.arch.flux_1.components.blocks import _GEGLUActivation
        act = _GEGLUActivation()
        torch.manual_seed(42)
        x = torch.randn(1, 4, 8)
        out = act(x)
        # If it were SiLU: silu(x1) * x2 - verify it uses gelu gate
        x1, x2 = x.chunk(2, dim=-1)
        expected_geglu = x1 * torch.nn.functional.gelu(x2)
        assert torch.allclose(out, expected_geglu, atol=1e-5), "Must use gelu gate (GEGLU)"

    def test_double_block_per_block_modulation(self):
        """Each Flux1DoubleStreamBlock must have its own img_mod and txt_mod."""
        from trainer.arch.flux_1.components.blocks import Flux1DoubleStreamBlock
        block = Flux1DoubleStreamBlock(TINY_HIDDEN, TINY_HEADS, TINY_MLP_RATIO)
        assert hasattr(block, "img_mod"), "Must have img_mod per-block modulation"
        assert hasattr(block, "txt_mod"), "Must have txt_mod per-block modulation"

    def test_single_block_per_block_modulation(self):
        """Each Flux1SingleStreamBlock must have its own modulation."""
        from trainer.arch.flux_1.components.blocks import Flux1SingleStreamBlock
        block = Flux1SingleStreamBlock(TINY_HIDDEN, TINY_HEADS, TINY_MLP_RATIO)
        assert hasattr(block, "modulation"), "Must have modulation per-block"


# ---------------------------------------------------------------------------
# 5. Model tests (tiny dimensions)
# ---------------------------------------------------------------------------

def _make_tiny_config():
    """Create a tiny Flux1Config for testing (not from FLUX1_CONFIGS)."""
    from trainer.arch.flux_1.components.configs import Flux1Config
    return Flux1Config(
        name="flux-1-test",
        num_double_blocks=1,
        num_single_blocks=1,
        hidden_size=64,
        num_attention_heads=2,
        head_dim=32,
        mlp_ratio=2.0,
        latent_channels=16,
        patch_size=2,
        in_channels=16,    # smaller for test
        context_dim=32,
        pooled_dim=16,
        rope_axes=(4, 4, 4),   # sums to 12 but head_dim=32 - need to match
        use_guidance_embed=True,
        activation="geglu",
    )


def _make_tiny_config_valid():
    """Create a tiny Flux1Config where rope_axes sum == head_dim."""
    from trainer.arch.flux_1.components.configs import Flux1Config
    # hidden=64, heads=2 -> head_dim=32; rope_axes must sum to 32
    return Flux1Config(
        name="flux-1-test",
        num_double_blocks=1,
        num_single_blocks=1,
        hidden_size=64,
        num_attention_heads=2,
        head_dim=32,
        mlp_ratio=2.0,
        latent_channels=16,
        patch_size=2,
        in_channels=16,
        context_dim=32,
        pooled_dim=16,
        rope_axes=(8, 12, 12),  # sums to 32 = head_dim
        use_guidance_embed=True,
        activation="geglu",
    )


class TestModel:
    def test_model_tiny_forward(self):
        """Tiny Flux1Transformer forward pass produces correct output shape."""
        from trainer.arch.flux_1.components.model import Flux1Transformer
        cfg = _make_tiny_config_valid()

        model = Flux1Transformer(cfg)
        model.eval()

        B = 1
        img_seq_len = 16  # arbitrary
        txt_seq_len = 10

        x = torch.randn(B, img_seq_len, cfg.in_channels)
        x_ids = torch.zeros(B, img_seq_len, 3)
        ctx = torch.randn(B, txt_seq_len, cfg.context_dim)
        ctx_ids = torch.zeros(B, txt_seq_len, 3)
        timesteps = torch.rand(B)
        pooled_text = torch.randn(B, cfg.pooled_dim)
        guidance = torch.ones(B)

        with torch.no_grad():
            out = model(
                x=x,
                x_ids=x_ids,
                timesteps=timesteps,
                ctx=ctx,
                ctx_ids=ctx_ids,
                guidance=guidance,
                pooled_text=pooled_text,
            )

        assert out.shape == (B, img_seq_len, cfg.in_channels), (
            f"Expected (B={B}, L={img_seq_len}, C={cfg.in_channels}), got {out.shape}"
        )
        assert torch.isfinite(out).all(), "Model output must be finite"

    def test_model_schnell_no_guidance(self):
        """Model with use_guidance_embed=False ignores guidance parameter."""
        from trainer.arch.flux_1.components.configs import Flux1Config
        from trainer.arch.flux_1.components.model import Flux1Transformer

        cfg = Flux1Config(
            name="flux-1-schnell-test",
            num_double_blocks=1,
            num_single_blocks=1,
            hidden_size=64,
            num_attention_heads=2,
            head_dim=32,
            mlp_ratio=2.0,
            latent_channels=16,
            patch_size=2,
            in_channels=16,
            context_dim=32,
            pooled_dim=16,
            rope_axes=(8, 12, 12),
            use_guidance_embed=False,
            activation="geglu",
        )

        model = Flux1Transformer(cfg)
        model.eval()

        B, img_len, txt_len = 1, 8, 5
        with torch.no_grad():
            out = model(
                x=torch.randn(B, img_len, cfg.in_channels),
                x_ids=torch.zeros(B, img_len, 3),
                timesteps=torch.rand(B),
                ctx=torch.randn(B, txt_len, cfg.context_dim),
                ctx_ids=torch.zeros(B, txt_len, 3),
                guidance=None,  # schnell: no guidance
                pooled_text=torch.randn(B, cfg.pooled_dim),
            )
        assert out.shape == (B, img_len, cfg.in_channels)

    def test_model_has_double_single_blocks(self):
        """Model must have Flux1DoubleStreamBlock and Flux1SingleStreamBlock."""
        from trainer.arch.flux_1.components.model import Flux1Transformer
        from trainer.arch.flux_1.components.blocks import Flux1DoubleStreamBlock, Flux1SingleStreamBlock
        cfg = _make_tiny_config_valid()
        model = Flux1Transformer(cfg)

        assert len(model.double_blocks) == cfg.num_double_blocks
        assert len(model.single_blocks) == cfg.num_single_blocks
        assert isinstance(model.double_blocks[0], Flux1DoubleStreamBlock)
        assert isinstance(model.single_blocks[0], Flux1SingleStreamBlock)


# ---------------------------------------------------------------------------
# 6. Strategy tests
# ---------------------------------------------------------------------------

class TestStrategy:
    def test_architecture_name(self):
        """Flux1Strategy.architecture must return 'flux_1'."""
        from trainer.arch.flux_1.strategy import Flux1Strategy
        from trainer.config.schema import TrainConfig
        config = TrainConfig(
            model={"architecture": "flux_1", "base_model_path": "/fake"},
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = Flux1Strategy(config)
        assert strategy.architecture == "flux_1"

    def test_supports_video_false(self):
        """Flux1Strategy must not support video."""
        from trainer.arch.flux_1.strategy import Flux1Strategy
        from trainer.config.schema import TrainConfig
        config = TrainConfig(
            model={"architecture": "flux_1", "base_model_path": "/fake"},
            training={"method": "full_finetune"},
            data={"dataset_config_path": "/fake.toml"},
        )
        strategy = Flux1Strategy(config)
        assert strategy.supports_video is False

    def test_registry_discovery(self):
        """'flux_1' must appear in list_models() after discovery."""
        from trainer.registry import list_models
        models = list_models()
        assert "flux_1" in models, f"'flux_1' not in registered models: {models}"

    def test_registry_resolves_flux1(self):
        """get_model_strategy('flux_1') must return Flux1Strategy."""
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("flux_1")
        assert cls.__name__ == "Flux1Strategy"

    def test_timestep_sampling_uniform(self):
        """Uniform timestep sampling must produce values in [0, 1]."""
        from trainer.arch.base import ModelStrategy
        t = ModelStrategy._sample_t(512, torch.device("cpu"), method="uniform")
        assert t.min() >= 0.0 - 1e-6
        assert t.max() <= 1.0 + 1e-6
        assert torch.isfinite(t).all()

    @pytest.mark.parametrize("method", ["uniform", "sigmoid", "logit_normal", "shift"])
    def test_timestep_sampling_methods(self, method):
        """All timestep methods produce finite values in [0, 1]."""
        from trainer.arch.base import ModelStrategy
        t = ModelStrategy._sample_t(
            batch_size=256, device=torch.device("cpu"), method=method
        )
        assert t.min() >= 0.0 - 1e-6
        assert t.max() <= 1.0 + 1e-6
        assert torch.isfinite(t).all()

    def test_flux1_training_step(self):
        """Feed a synthetic batch through Flux1Strategy.training_step, verify output."""
        from trainer.arch.flux_1.strategy import Flux1Strategy
        from trainer.arch.base import ModelComponents, TrainStepOutput
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS

        config = _make_flux1_train_config()
        strategy = Flux1Strategy(config)
        components = _mock_setup_flux1(strategy)

        # Flux 1 batch: latents (B, 16, H, W), ctx_vec (B, L, 4096)
        B, H, W = 1, 8, 8
        batch = {
            "latents": torch.randn(B, 16, H, W),
            "ctx_vec": torch.randn(B, 10, 4096),
        }

        output = strategy.training_step(components, batch, step=0)

        assert isinstance(output, TrainStepOutput)
        assert output.loss.dim() == 0, "Loss must be scalar"
        assert torch.isfinite(output.loss), "Loss must be finite"
        assert output.loss.item() > 0, "Loss must be positive"
        assert "loss" in output.metrics
        assert "timestep_mean" in output.metrics

    def test_flux1_metrics_detached(self):
        """Metrics tensors must not require grad (use .detach())."""
        from trainer.arch.flux_1.strategy import Flux1Strategy

        config = _make_flux1_train_config()
        strategy = Flux1Strategy(config)
        components = _mock_setup_flux1(strategy)

        B, H, W = 1, 4, 4
        batch = {
            "latents": torch.randn(B, 16, H, W),
            "ctx_vec": torch.randn(B, 8, 4096),
        }
        output = strategy.training_step(components, batch, step=0)

        assert not output.metrics["loss"].requires_grad
        assert not output.metrics["timestep_mean"].requires_grad

    def test_flux1_pack_in_training_step(self):
        """Verify latents get packed from 16ch to 64ch before model receives them."""
        from trainer.arch.flux_1.strategy import Flux1Strategy
        from trainer.arch.base import ModelComponents

        config = _make_flux1_train_config()
        strategy = Flux1Strategy(config)

        received_x_shapes = []

        class ShapeCaptureMock(nn.Module):
            """Mock model that records the shape of x."""
            def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance=None, pooled_text=None):
                received_x_shapes.append(x.shape)
                # Return same shape as x
                return x

        device = torch.device("cpu")
        strategy._blocks_to_swap = 0
        strategy._device = device
        strategy._train_dtype = torch.float32
        strategy._flux1_config = __import__(
            "trainer.arch.flux_1.components.configs", fromlist=["FLUX1_CONFIGS"]
        ).FLUX1_CONFIGS["dev"]
        strategy._model_version = "dev"
        strategy._noise_offset_val = 0.0
        strategy._flow_shift = 1.0
        strategy._guidance_scale = 1.0
        strategy._ts_method = "uniform"
        strategy._ts_min = 0.0
        strategy._ts_max = 1.0
        strategy._ts_sigmoid_scale = 1.0
        strategy._ts_logit_mean = 0.0
        strategy._ts_logit_std = 1.0

        components = ModelComponents(model=ShapeCaptureMock())

        B, H, W = 1, 8, 8  # 16ch latents, H=W=8
        batch = {
            "latents": torch.randn(B, 16, H, W),
            "ctx_vec": torch.randn(B, 10, 4096),
        }
        strategy.training_step(components, batch, step=0)

        assert len(received_x_shapes) == 1
        shape = received_x_shapes[0]
        # Expected: (1, (8/2)*(8/2), 64) = (1, 16, 64)
        assert shape == (1, 16, 64), (
            f"Expected packed shape (1, 16, 64), got {shape}. "
            "Latents must be packed from (B, 16, H, W) to (B, HW/4, 64)"
        )

    def test_schnell_no_guidance_in_training_step(self):
        """Schnell variant must pass guidance=None to the model."""
        from trainer.arch.flux_1.strategy import Flux1Strategy
        from trainer.arch.base import ModelComponents
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS

        config = _make_flux1_train_config(model_version="schnell")
        strategy = Flux1Strategy(config)

        guidance_received = []

        class GuidanceCaptureMock(nn.Module):
            def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance=None, pooled_text=None):
                guidance_received.append(guidance)
                return x

        _mock_setup_flux1_raw(strategy, model_version="schnell")
        components = ModelComponents(model=GuidanceCaptureMock())

        B, H, W = 1, 4, 4
        batch = {
            "latents": torch.randn(B, 16, H, W),
            "ctx_vec": torch.randn(B, 8, 4096),
        }
        strategy.training_step(components, batch, step=0)

        assert guidance_received[0] is None, "Schnell must pass guidance=None"


# ---------------------------------------------------------------------------
# Helper: mock strategy setup (avoids loading real weights)
# ---------------------------------------------------------------------------

def _make_flux1_train_config(model_version: str = "dev"):
    """Build a TrainConfig for Flux1Strategy testing."""
    from trainer.config.schema import TrainConfig
    return TrainConfig(
        model={
            "architecture": "flux_1",
            "base_model_path": "/fake",
            "gradient_checkpointing": False,
            "model_kwargs": {"model_version": model_version},
        },
        training={"method": "full_finetune", "timestep_sampling": "uniform"},
        data={"dataset_config_path": "/fake.toml"},
    )


class TinyMockFlux1(nn.Module):
    """Minimal model with Flux 1 forward signature.

    Returns a single tensor (B, HW, in_channels) with the same shape as input x.
    """

    def __init__(self, in_channels: int = 64):
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
        guidance: torch.Tensor | None = None,
        pooled_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear(x.float()).to(x.dtype)

    def enable_gradient_checkpointing(self) -> None:
        pass

    def prepare_block_swap_before_forward(self) -> None:
        pass


def _mock_setup_flux1(strategy) -> "ModelComponents":
    """Set all self._* attributes that setup() would set, return TinyMockFlux1."""
    from trainer.arch.base import ModelComponents
    from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS

    cfg = strategy.config
    flux1_config = FLUX1_CONFIGS["dev"]
    device = torch.device("cpu")

    strategy._blocks_to_swap = 0
    strategy._device = device
    strategy._train_dtype = torch.float32
    strategy._flux1_config = flux1_config
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

    # 64 in_channels (packed)
    model = TinyMockFlux1(in_channels=64).to(device)

    return ModelComponents(
        model=model,
        extra={"flux1_config": flux1_config, "model_version": "dev"},
    )


def _mock_setup_flux1_raw(strategy, model_version: str = "dev") -> None:
    """Set all strategy._* fields for testing without returning ModelComponents."""
    from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS

    flux1_config = FLUX1_CONFIGS[model_version]
    device = torch.device("cpu")

    strategy._blocks_to_swap = 0
    strategy._device = device
    strategy._train_dtype = torch.float32
    strategy._flux1_config = flux1_config
    strategy._model_version = model_version
    strategy._noise_offset_val = 0.0
    strategy._flow_shift = 1.0
    strategy._guidance_scale = 1.0
    strategy._ts_method = "uniform"
    strategy._ts_min = 0.0
    strategy._ts_max = 1.0
    strategy._ts_sigmoid_scale = 1.0
    strategy._ts_logit_mean = 0.0
    strategy._ts_logit_std = 1.0
