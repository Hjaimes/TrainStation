# Phase 5b: SDXL, SD3, Flux 1 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SDXL, SD3, and Flux 1 architectures to the training app using our established ModelStrategy pattern, bringing total architectures to 12.

**Architecture:** Three independent architecture directories under `trainer/arch/`, each with `strategy.py`, `__init__.py`, and `components/`. SDXL wraps diffusers' UNet directly (epsilon prediction). SD3 and Flux 1 have custom model code following our existing conventions (SD3: custom MMDiT; Flux 1: based on our Flux 2 structure). All three use the same ModelStrategy base class.

**Tech Stack:** Python 3.10+, PyTorch, diffusers (SDXL only), einops

---

## What Already Exists (DO NOT REBUILD)

| File | Purpose |
|------|---------|
| `trainer/arch/base.py` | `ModelStrategy`, `ModelComponents`, `TrainStepOutput` |
| `trainer/arch/flux_2/` | Complete Flux 2 architecture (template for Flux 1) |
| `trainer/arch/wan/` | Wan architecture (template for strategy structure) |
| `trainer/registry.py` | `register_model()`, `get_model_strategy()`, `list_models()` |
| `trainer/networks/arch_configs.py` | `ARCH_NETWORK_CONFIGS` dict + `get_arch_config()` |
| `tests/test_imports.py` | Canary imports for all modules |

---

## Architecture A: SDXL

### Task A1: Create `trainer/arch/sdxl/components/configs.py`

**Files:** Create `trainer/arch/sdxl/components/__init__.py` and `trainer/arch/sdxl/components/configs.py`

SDXL variant configs as frozen dataclasses:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SDXLConfig:
    """SDXL variant configuration."""
    name: str
    prediction_type: str           # "epsilon" or "v_prediction"
    latent_channels: int           # 4
    vae_scaling_factor: float      # 0.13025
    num_train_timesteps: int       # 1000
    # UNet params (for validation, diffusers loads internally)
    in_channels: int               # 4
    cross_attention_dim: int       # 2048  (CLIP-L 768 + CLIP-G 1280)
    # Time ID dimensions
    time_ids_size: int             # 6  (orig_h, orig_w, crop_top, crop_left, target_h, target_w)


SDXL_CONFIGS = {
    "base": SDXLConfig(
        name="sdxl-base-1.0",
        prediction_type="epsilon",
        latent_channels=4,
        vae_scaling_factor=0.13025,
        num_train_timesteps=1000,
        in_channels=4,
        cross_attention_dim=2048,
        time_ids_size=6,
    ),
    "v_pred": SDXLConfig(
        name="sdxl-base-1.0-vpred",
        prediction_type="v_prediction",
        latent_channels=4,
        vae_scaling_factor=0.13025,
        num_train_timesteps=1000,
        in_channels=4,
        cross_attention_dim=2048,
        time_ids_size=6,
    ),
}
```

### Task A2: Create `trainer/arch/sdxl/components/utils.py`

**Files:** Create `trainer/arch/sdxl/components/utils.py`

DDPM schedule helpers and time ID builder:

```python
import torch
import math

def compute_alphas_cumprod(num_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012) -> torch.Tensor:
    """Compute alpha_bar schedule (scaled linear beta schedule, matching diffusers)."""
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)

def build_time_ids(
    original_size: tuple[int, int],
    crop_coords: tuple[int, int],
    target_size: tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build add_time_ids tensor [6] for SDXL conditioning."""
    return torch.tensor([
        original_size[0], original_size[1],
        crop_coords[0], crop_coords[1],
        target_size[0], target_size[1],
    ], dtype=dtype)

def get_velocity(latents: torch.Tensor, noise: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    """Compute v-prediction target: v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * latents."""
    return alpha_bar_t.sqrt() * noise - (1.0 - alpha_bar_t).sqrt() * latents
```

### Task A3: Create `trainer/arch/sdxl/components/model.py`

**Files:** Create `trainer/arch/sdxl/components/model.py`

Thin wrapper around diffusers UNet:

```python
import logging
import torch
from diffusers import UNet2DConditionModel

logger = logging.getLogger(__name__)

def load_sdxl_unet(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> UNet2DConditionModel:
    """Load SDXL UNet from a single-file checkpoint or diffusers directory.

    Handles both formats:
    - Directory with model_index.json (diffusers format)
    - Single .safetensors file
    """
    logger.info("Loading SDXL UNet from %s", model_path)

    if model_path.endswith(".safetensors"):
        model = UNet2DConditionModel.from_single_file(
            model_path, torch_dtype=dtype,
        )
    else:
        model = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", torch_dtype=dtype,
        )

    model = model.to(device=device)
    logger.info("SDXL UNet loaded: %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    return model
```

### Task A4: Create `trainer/arch/sdxl/strategy.py`

**Files:** Create `trainer/arch/sdxl/strategy.py`

The SDXL strategy — epsilon prediction (DDPM noise schedule), NOT flow matching:

```python
"""SDXL model strategy. Epsilon prediction with DDPM noise schedule."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from trainer.arch.base import ModelStrategy, ModelComponents, TrainStepOutput

logger = logging.getLogger(__name__)

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


class SDXLStrategy(ModelStrategy):
    """Strategy for SDXL image generation (epsilon or v-prediction).

    Uses diffusers UNet2DConditionModel. Image-only. No block swap needed.
    Key difference from flow-matching architectures: uses DDPM discrete timesteps
    (0..999) and alpha_bar noise schedule instead of continuous t ∈ [0,1].
    """

    @property
    def architecture(self) -> str:
        return "sdxl"

    @property
    def supports_video(self) -> bool:
        return False

    def setup(self) -> ModelComponents:
        from trainer.arch.sdxl.components.configs import SDXL_CONFIGS
        from trainer.arch.sdxl.components.model import load_sdxl_unet
        from trainer.arch.sdxl.components.utils import compute_alphas_cumprod

        cfg = self.config
        train_dtype = _resolve_dtype(cfg.model.dtype)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_version = cfg.model.model_kwargs.get("model_version", "base")
        if model_version not in SDXL_CONFIGS:
            raise ValueError(
                f"Unknown SDXL model_version '{model_version}'. "
                f"Available: {list(SDXL_CONFIGS)}"
            )
        sdxl_config = SDXL_CONFIGS[model_version]

        model = load_sdxl_unet(
            model_path=cfg.model.base_model_path,
            dtype=train_dtype,
            device=device,
        )

        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Pre-compute alpha_bar schedule on device (cached for training loop)
        alphas_cumprod = compute_alphas_cumprod(sdxl_config.num_train_timesteps).to(device)

        # Cache everything
        self._device = device
        self._train_dtype = train_dtype
        self._sdxl_config = sdxl_config
        self._alphas_cumprod = alphas_cumprod
        self._noise_offset_val = cfg.training.noise_offset
        self._ts_min = int(cfg.training.min_timestep * sdxl_config.num_train_timesteps)
        self._ts_max = int(cfg.training.max_timestep * sdxl_config.num_train_timesteps)

        return ModelComponents(
            model=model,
            extra={"sdxl_config": sdxl_config, "model_version": model_version},
        )

    def training_step(
        self,
        components: ModelComponents,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> TrainStepOutput:
        """Epsilon-prediction training step for SDXL.

        Batch format:
            latents:    (B, 4, H/8, W/8) — 4-channel latents
            ctx_vec:    (B, L, 2048) — concatenated CLIP-L + CLIP-G hidden states
            pooled_vec: (B, 1280) — CLIP-G pooled output (optional, from batch or default)

        Pipeline:
        1. Sample discrete timesteps 0..999.
        2. Add noise: noisy = sqrt(alpha_bar_t) * latents + sqrt(1 - alpha_bar_t) * noise.
        3. Build add_time_ids and added_cond_kwargs.
        4. UNet forward: pred = unet(noisy, t, encoder_hidden_states, added_cond_kwargs).
        5. Loss = MSE(pred, target) where target = noise (epsilon) or velocity (v-pred).
        """
        from trainer.arch.sdxl.components.utils import build_time_ids, get_velocity

        device = self._device
        train_dtype = self._train_dtype
        config = self._sdxl_config

        latents = batch["latents"].to(device=device, dtype=train_dtype)
        ctx_vec = batch["ctx_vec"].to(device=device, dtype=train_dtype)
        bsz = latents.shape[0]

        # Pooled embeddings (CLIP-G pool or zeros)
        if "pooled_vec" in batch:
            pooled = batch["pooled_vec"].to(device=device, dtype=train_dtype)
        else:
            pooled = torch.zeros(bsz, 1280, device=device, dtype=train_dtype)

        # --- Noise ---
        noise = torch.empty_like(latents).normal_()
        if self._noise_offset_val > 0:
            noise.add_(
                torch.randn(bsz, latents.shape[1], 1, 1, device=device, dtype=train_dtype),
                alpha=self._noise_offset_val,
            )

        # --- Discrete timesteps ---
        timesteps = torch.randint(
            self._ts_min, self._ts_max,
            (bsz,), device=device, dtype=torch.long,
        )

        # --- DDPM noisy latents ---
        alpha_bar_t = self._alphas_cumprod[timesteps].to(train_dtype)
        sqrt_alpha_bar = alpha_bar_t.sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1, 1)
        noisy_latents = sqrt_alpha_bar * latents + sqrt_one_minus_alpha_bar * noise

        # --- Conditioning ---
        # Build time IDs per sample (assume training at native resolution, no crop)
        h_pixels = latents.shape[2] * 8
        w_pixels = latents.shape[3] * 8
        time_ids = build_time_ids(
            original_size=(h_pixels, w_pixels),
            crop_coords=(0, 0),
            target_size=(h_pixels, w_pixels),
            dtype=train_dtype,
        ).to(device).unsqueeze(0).expand(bsz, -1)

        added_cond_kwargs = {
            "text_embeds": pooled,
            "time_ids": time_ids,
        }

        # --- Forward pass ---
        model_pred = components.model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=ctx_vec,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # --- Loss ---
        if config.prediction_type == "epsilon":
            target = noise
        else:  # v_prediction
            target = get_velocity(latents, noise, alpha_bar_t.view(-1, 1, 1, 1))

        loss = F.mse_loss(model_pred.to(train_dtype), target, reduction="mean")

        return TrainStepOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "timestep_mean": timesteps.float().mean().detach(),
            },
        )
```

### Task A5: Create `trainer/arch/sdxl/__init__.py`

**Files:** Create `trainer/arch/sdxl/__init__.py`

```python
from trainer.registry import register_model
from trainer.arch.sdxl.strategy import SDXLStrategy

register_model("sdxl")(SDXLStrategy)
```

### Task A6: Create `tests/test_sdxl_components.py`

**Files:** Create `tests/test_sdxl_components.py`

Tests — all must pass without GPU or real model weights:

1. **Config tests:**
   - `test_configs_exist` — `SDXL_CONFIGS` has "base" and "v_pred"
   - `test_config_frozen` — assigning to a field raises `FrozenInstanceError`
   - `test_base_config_values` — base has prediction_type="epsilon", latent_channels=4, vae_scaling_factor=0.13025

2. **Utils tests:**
   - `test_alphas_cumprod_shape` — returns `[1000]`, decreasing, in (0,1)
   - `test_alphas_cumprod_decreasing` — each value < previous
   - `test_build_time_ids` — returns shape `[6]` with correct values
   - `test_build_time_ids_values` — `build_time_ids((512, 768), (0, 0), (512, 768))` == `[512, 768, 0, 0, 512, 768]`
   - `test_get_velocity` — verify `v = sqrt(a)*noise - sqrt(1-a)*latents` numerically
   - `test_epsilon_vs_vpred_target_differ` — epsilon target == noise, v-pred target != noise

3. **Strategy tests:**
   - `test_architecture_name` — returns `"sdxl"`
   - `test_supports_video_false` — returns `False`
   - `test_registry_discovery` — `"sdxl" in list_models()`

4. **Training step mock test:**
   - `test_sdxl_epsilon_training_step` — Create minimal mock: subclass SDXLStrategy, override `setup()` to use a tiny `nn.Linear`-based mock that mimics UNet output shape `(B, 4, H, W)`. Call `training_step()` with synthetic batch `{latents: (1,4,8,8), ctx_vec: (1,77,2048)}`. Verify:
     - loss is scalar, finite, > 0
     - metrics has "loss" and "timestep_mean"
   - `test_sdxl_vpred_training_step` — Same but with v_pred config. Verify loss != epsilon loss (different targets).

---

## Architecture B: SD3

### Task B1: Create `trainer/arch/sd3/components/configs.py`

**Files:** Create `trainer/arch/sd3/components/__init__.py` and `trainer/arch/sd3/components/configs.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SD3Config:
    """SD3 variant configuration."""
    name: str
    num_layers: int            # joint transformer blocks
    num_single_layers: int     # single (image-only) blocks (0 for SD3.0)
    hidden_size: int           # 1536 for medium, 2048 for large
    num_attention_heads: int
    patch_size: int            # 2
    latent_channels: int       # 16
    vae_scaling_factor: float  # 1.5305
    vae_shift_factor: float    # 0.0609
    pooled_projection_dim: int # 2048 (CLIP-L 768 + CLIP-G 1280)
    caption_projection_dim: int  # 4096 (T5-XXL)
    joint_attention_dim: int   # 4096 (T5-XXL hidden dim for cross-attn)
    dual_attention_layers: tuple[int, ...] | None  # which layers have dual attention (SD3.5)


SD3_CONFIGS = {
    "sd3-medium": SD3Config(
        name="sd3-medium",
        num_layers=24,
        num_single_layers=0,
        hidden_size=1536,
        num_attention_heads=24,
        patch_size=2,
        latent_channels=16,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        pooled_projection_dim=2048,
        caption_projection_dim=4096,
        joint_attention_dim=4096,
        dual_attention_layers=None,
    ),
    "sd3.5-medium": SD3Config(
        name="sd3.5-medium",
        num_layers=24,
        num_single_layers=12,
        hidden_size=1536,
        num_attention_heads=24,
        patch_size=2,
        latent_channels=16,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        pooled_projection_dim=2048,
        caption_projection_dim=4096,
        joint_attention_dim=4096,
        dual_attention_layers=tuple(range(24)),
    ),
    "sd3.5-large": SD3Config(
        name="sd3.5-large",
        num_layers=38,
        num_single_layers=12,
        hidden_size=2048,
        num_attention_heads=32,
        patch_size=2,
        latent_channels=16,
        vae_scaling_factor=1.5305,
        vae_shift_factor=0.0609,
        pooled_projection_dim=2048,
        caption_projection_dim=4096,
        joint_attention_dim=4096,
        dual_attention_layers=tuple(range(38)),
    ),
}
```

### Task B2: Create `trainer/arch/sd3/components/layers.py`

**Files:** Create `trainer/arch/sd3/components/layers.py`

Core building blocks for SD3:

- `AdaLayerNormZero` — adaptive layer norm with timestep + pooled conditioning. Produces 6 modulation params (shift, scale, gate for self-attn and cross-attn).
- `AdaLayerNormContinuous` — for final norm. Uses continuous embedding, produces shift + scale.
- `FeedForward` — GELU MLP: `Linear(d, 4d) → GELU → Linear(4d, d)`

```python
class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, num_embeds: int = 6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, num_embeds * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, ...]:
        emb = self.linear(self.silu(emb))
        chunks = emb.unsqueeze(1).chunk(self.linear.out_features // x.shape[-1], dim=-1)
        # Returns (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        x = self.norm(x)
        return (x,) + tuple(chunks)
```

### Task B3: Create `trainer/arch/sd3/components/embeddings.py`

**Files:** Create `trainer/arch/sd3/components/embeddings.py`

- `PatchEmbed` — 2D patch embedding: `Conv2d(in_channels, hidden, kernel=patch, stride=patch)` + position embedding
- `CombinedTimestepTextProjEmbeddings` — timestep embedding (sinusoidal → MLP) + pooled text projection, summed

```python
class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 2, in_channels: int = 16, embed_dim: int = 1536, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, latent: Tensor) -> Tensor:
        # (B, C, H, W) → (B, H/p * W/p, D)
        return self.proj(latent).flatten(2).transpose(1, 2)


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(256)  # sinusoidal
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.text_embedder = nn.Linear(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: Tensor, pooled_projection: Tensor) -> Tensor:
        t_emb = self.timestep_embedder(self.time_proj(timestep))
        p_emb = self.text_embedder(pooled_projection)
        return t_emb + p_emb
```

### Task B4: Create `trainer/arch/sd3/components/blocks.py`

**Files:** Create `trainer/arch/sd3/components/blocks.py`

The two core block types:

**`JointTransformerBlock`** — bidirectional text+image attention:
1. AdaLayerNormZero → modulation params for image stream
2. Separate QKV projections for image and text
3. Concatenate image + text along sequence dim → single self-attention
4. Split back → image and text outputs
5. Gated residual + MLP for both streams

**`SD3SingleTransformerBlock`** — image-only (no text stream):
1. AdaLayerNorm → modulation for image
2. Self-attention on image tokens only
3. Gated residual + MLP

Key details:
- QK norm (RMSNorm) applied per-head before attention
- `nn.functional.scaled_dot_product_attention` for attention computation
- Both blocks take `(hidden_states, encoder_hidden_states, temb)` as input

### Task B5: Create `trainer/arch/sd3/components/model.py`

**Files:** Create `trainer/arch/sd3/components/model.py`

`SD3Transformer2DModel`:
- `PatchEmbed` for latent → tokens
- `nn.Linear` context projector (4096 → hidden_size) for T5 text embeddings
- `CombinedTimestepTextProjEmbeddings` for timestep + pooled conditioning
- Stack of `JointTransformerBlock` (+ optional `SD3SingleTransformerBlock`)
- `AdaLayerNormContinuous` final norm
- `nn.Linear` output projector (hidden → patch_size² × latent_channels)
- Unpatchify to spatial

Forward signature: `(hidden_states, encoder_hidden_states, timestep, pooled_projections)`

Loader function: `load_sd3_model(config, device, path, dtype)` — load from safetensors with key remapping.

### Task B6: Create `trainer/arch/sd3/strategy.py`

**Files:** Create `trainer/arch/sd3/strategy.py`

Flow-matching strategy (same flow formulation as Flux 2):

```python
class SD3Strategy(ModelStrategy):
    @property
    def architecture(self) -> str:
        return "sd3"

    @property
    def supports_video(self) -> bool:
        return False

    def setup(self) -> ModelComponents:
        # Load SD3Transformer2DModel, cache config values
        # Timestep sampling uses same _sample_timesteps pattern as Flux 2
        ...

    def training_step(self, components, batch, step) -> TrainStepOutput:
        # Batch: latents (B,16,H/8,W/8), ctx_vec (B,L,4096), pooled_vec (B,2048)
        # 1. Sample noise, timesteps (continuous [0,1])
        # 2. noisy = (1-t)*latents + t*noise
        # 3. Forward: model(noisy, ctx_vec, timestep=t, pooled_projections=pooled_vec)
        # 4. target = noise - latents
        # 5. loss = MSE(pred, target)
        ...
```

Timestep sampling: reuse same `_sample_timesteps` static method from Flux 2 (copy, don't import — architectures are self-contained). SD3 uses timesteps scaled to `[0, 1000]` for the model: `model_timestep = t * 1000`.

### Task B7: Create `trainer/arch/sd3/__init__.py`

**Files:** Create `trainer/arch/sd3/__init__.py`

```python
from trainer.registry import register_model
from trainer.arch.sd3.strategy import SD3Strategy

register_model("sd3")(SD3Strategy)
```

### Task B8: Create `tests/test_sd3_components.py`

**Files:** Create `tests/test_sd3_components.py`

Tests:

1. **Config tests:**
   - `test_configs_exist` — SD3_CONFIGS has "sd3-medium", "sd3.5-medium", "sd3.5-large"
   - `test_config_frozen` — frozen dataclass
   - `test_medium_no_single_blocks` — sd3-medium has `num_single_layers=0`
   - `test_sd35_has_single_blocks` — sd3.5-medium has `num_single_layers=12`

2. **Layer tests:**
   - `test_ada_layer_norm_zero_output` — input (B,L,D), output tuple of 7 tensors
   - `test_feed_forward_shape` — input (B,L,D), output (B,L,D)

3. **Embedding tests:**
   - `test_patch_embed_shape` — input (B,16,32,32), output (B, 256, D) where 256 = (32/2)*(32/2)
   - `test_combined_timestep_text_proj` — outputs shape (B, D)

4. **Block tests:**
   - `test_joint_block_forward` — input image (B,HW,D) + text (B,L,D), output same shapes
   - `test_single_block_forward` — input image (B,HW,D), output same shape

5. **Model tests:**
   - `test_model_tiny_forward` — Create tiny SD3Transformer2DModel (2 layers, hidden=64, 4 heads), verify forward produces (B, 16, H, W)

6. **Strategy tests:**
   - `test_architecture_name` — returns "sd3"
   - `test_supports_video_false`
   - `test_registry_discovery` — `"sd3" in list_models()`
   - `test_sd3_training_step` — Mock model (tiny Linear-based), verify loss is finite scalar with correct metrics
   - `test_sd3_flow_matching_target` — Verify target = noise - latents (not just noise like SDXL)

---

## Architecture C: Flux 1

### Task C1: Create `trainer/arch/flux_1/components/configs.py`

**Files:** Create `trainer/arch/flux_1/components/__init__.py` and `trainer/arch/flux_1/components/configs.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Flux1Config:
    """Flux 1 variant configuration."""
    name: str
    num_double_blocks: int    # 19
    num_single_blocks: int    # 38
    hidden_size: int          # 3072
    num_attention_heads: int  # 24
    head_dim: int             # 128  (24 * 128 = 3072)
    mlp_ratio: float          # 4.0
    latent_channels: int      # 16  (packed to 64 = 16 * 2 * 2)
    patch_size: int           # 2
    in_channels: int          # 64  (after packing)
    context_dim: int          # 4096  (T5-XXL)
    pooled_dim: int           # 768   (CLIP-L)
    rope_axes: tuple[int, int, int]  # (16, 56, 56)
    use_guidance_embed: bool  # True for dev, False for schnell
    activation: str           # "geglu"


FLUX1_CONFIGS = {
    "dev": Flux1Config(
        name="flux-1-dev",
        num_double_blocks=19,
        num_single_blocks=38,
        hidden_size=3072,
        num_attention_heads=24,
        head_dim=128,
        mlp_ratio=4.0,
        latent_channels=16,
        patch_size=2,
        in_channels=64,
        context_dim=4096,
        pooled_dim=768,
        rope_axes=(16, 56, 56),
        use_guidance_embed=True,
        activation="geglu",
    ),
    "schnell": Flux1Config(
        name="flux-1-schnell",
        num_double_blocks=19,
        num_single_blocks=38,
        hidden_size=3072,
        num_attention_heads=24,
        head_dim=128,
        mlp_ratio=4.0,
        latent_channels=16,
        patch_size=2,
        in_channels=64,
        context_dim=4096,
        pooled_dim=768,
        rope_axes=(16, 56, 56),
        use_guidance_embed=False,
        activation="geglu",
    ),
}
```

### Task C2: Create `trainer/arch/flux_1/components/utils.py`

**Files:** Create `trainer/arch/flux_1/components/utils.py`

Pack/unpack latents and 3D RoPE position IDs:

- `pack_latents(x)` — `(B, 16, H, W)` → `(B, (H/2)*(W/2), 64)` via 2×2 packing
- `unpack_latents(x, h, w)` — inverse
- `prepare_img_ids(h, w)` — 3D position IDs `(1, HW, 3)` for [channel_index, y, x]
- `prepare_txt_ids(seq_len)` — text position IDs `(1, L, 3)` (all zeros)
- `rope_3d(positions, axes_dim)` — 3D rotary positional embeddings matching `(16, 56, 56)` axes

Key difference from Flux 2:
- Flux 2 uses 4D position IDs `(B, HW, 4)` via `prc_img`/`prc_txt` — extra dimension for time
- Flux 1 uses 3D position IDs `(B, HW, 3)` — no temporal dimension (image-only)
- Flux 2 packs 128ch → 128, Flux 1 packs 16ch → 64

### Task C3: Create `trainer/arch/flux_1/components/embeddings.py`

**Files:** Create `trainer/arch/flux_1/components/embeddings.py`

- `Flux1RoPE` — 3D rotary embeddings with axes `(16, 56, 56)`. Computes cos/sin from 3D position IDs.
- `TimestepEmbedding` — sinusoidal → MLP (same pattern as SD3, but standalone)
- `GuidanceEmbedding` — same as timestep embedding, for guidance scale (dev only)
- `MLPEmbedder` — `Linear(in, out) → SiLU → Linear(out, out)` for pooled text projection

### Task C4: Create `trainer/arch/flux_1/components/blocks.py`

**Files:** Create `trainer/arch/flux_1/components/blocks.py`

Based on our Flux 2 block structure with key differences:

**`Flux1DoubleStreamBlock`:**
- Per-block `AdaLayerNorm` modulation (NOT global shared like Flux 2)
- Image stream + text stream, separate QKV projections
- Concatenate → joint attention with RoPE → split
- GEGLU FFN (not SiLU-gated like Flux 2)
- Gated residual for both streams

**`Flux1SingleStreamBlock`:**
- Per-block modulation
- Combined image+text as single sequence
- Self-attention with RoPE
- GEGLU FFN
- Gated residual

**GEGLU activation:**
```python
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)
```

### Task C5: Create `trainer/arch/flux_1/components/model.py`

**Files:** Create `trainer/arch/flux_1/components/model.py`

`Flux1Transformer`:
- `nn.Linear(in_channels, hidden_size)` image input projection
- `nn.Linear(context_dim, hidden_size)` text input projection
- `TimestepEmbedding` + optional `GuidanceEmbedding` + `MLPEmbedder` for pooled text
- Stack of `Flux1DoubleStreamBlock` (19) + `Flux1SingleStreamBlock` (38)
- Final norm + `nn.Linear(hidden_size, in_channels)` output projection

Forward: `(x, x_ids, timesteps, ctx, ctx_ids, guidance=None)` → `(B, HW, 64)`

Block swap support: same pattern as Flux 2 — `enable_block_swap()`, `move_to_device_except_swap_blocks()`, `prepare_block_swap_before_forward()`.

Loader: `load_flux1_model(config, device, path, attn_mode, ...)` — similar to `load_flux2_model`.

### Task C6: Create `trainer/arch/flux_1/strategy.py`

**Files:** Create `trainer/arch/flux_1/strategy.py`

Flow-matching strategy following Flux 2's pattern closely:

```python
class Flux1Strategy(ModelStrategy):
    @property
    def architecture(self) -> str:
        return "flux_1"

    @property
    def supports_video(self) -> bool:
        return False

    def setup(self) -> ModelComponents:
        # Same structure as Flux2Strategy.setup()
        # Load Flux1Transformer, handle block swap, cache config values
        ...

    # Block swap lifecycle hooks — same as Flux 2
    def on_before_accelerate_prepare(self, components, accelerator): ...
    def on_after_accelerate_prepare(self, components, accelerator): ...
    def on_before_training_step(self, components): ...

    @staticmethod
    def _sample_timesteps(...): ...  # Same as Flux 2 (copy, self-contained)

    def training_step(self, components, batch, step) -> TrainStepOutput:
        # Batch: latents (B,16,H,W), ctx_vec (B,L,4096)
        # 1. Pack latents: (B,16,H,W) → (B,HW,64) + 3D position IDs
        # 2. Sample noise, timesteps (continuous [0,1])
        # 3. noisy = (1-t)*latents + t*noise (in packed space)
        # 4. Guidance vector (dev=1.0, schnell=None)
        # 5. Forward: model(noisy, img_ids, t, ctx, ctx_ids, guidance)
        # 6. Unpack: (B,HW,64) → (B,16,H,W) [note: 16ch not 128ch]
        # 7. target = noise - latents, loss = MSE
        ...
```

Key differences from Flux 2 training_step:
- Latents are `(B, 16, H, W)` not `(B, 128, H, W)`
- Pack to 64 channels not 128
- 3D position IDs via `prepare_img_ids`/`prepare_txt_ids` not `prc_img`/`prc_txt`
- Model timesteps stay in `[0, 1]` (same as Flux 2)

### Task C7: Create `trainer/arch/flux_1/__init__.py`

**Files:** Create `trainer/arch/flux_1/__init__.py`

```python
from trainer.registry import register_model
from trainer.arch.flux_1.strategy import Flux1Strategy

register_model("flux_1")(Flux1Strategy)
```

### Task C8: Create `tests/test_flux1_components.py`

**Files:** Create `tests/test_flux1_components.py`

Tests:

1. **Config tests:**
   - `test_configs_exist` — FLUX1_CONFIGS has "dev" and "schnell"
   - `test_config_frozen` — frozen dataclass
   - `test_dev_has_guidance` — dev has `use_guidance_embed=True`
   - `test_schnell_no_guidance` — schnell has `use_guidance_embed=False`
   - `test_dimensions` — hidden_size=3072, 24 heads × 128 dim, 19+38 blocks

2. **Utils tests:**
   - `test_pack_unpack_roundtrip` — pack then unpack recovers original tensor
   - `test_pack_shape` — `(1, 16, 8, 8)` → `(1, 16, 64)` where 16 = (8/2)*(8/2)
   - `test_img_ids_shape` — `prepare_img_ids(4, 4)` → `(1, 16, 3)`
   - `test_txt_ids_shape` — `prepare_txt_ids(77)` → `(1, 77, 3)`
   - `test_txt_ids_all_zeros` — text IDs are all zeros

3. **Block tests:**
   - `test_double_stream_block_forward` — input img (B,HW,D) + txt (B,L,D) + temb (B,D) + rope, output same shapes
   - `test_single_stream_block_forward` — input combined (B,HW+L,D) + temb + rope, output same shape
   - `test_geglu_activation` — verify GEGLU halves hidden dim

4. **Model tests (tiny):**
   - `test_model_tiny_forward` — Create tiny Flux1Transformer (1 double, 1 single, hidden=64, 2 heads), verify forward produces `(B, HW, in_channels)`

5. **Strategy tests:**
   - `test_architecture_name` — returns "flux_1"
   - `test_supports_video_false`
   - `test_registry_discovery` — `"flux_1" in list_models()`
   - `test_flux1_training_step` — Mock model, verify flow-matching loss (target = noise - latents)
   - `test_flux1_pack_in_training_step` — Verify latents get packed from 16ch to 64ch in the step

---

## Integration

### Task D1: Add LoRA configs to `arch_configs.py`

**Files:** Modify `trainer/networks/arch_configs.py`

Add three entries to `ARCH_NETWORK_CONFIGS`:

```python
# SDXL (UNet-based — targets diffusers Transformer2DModel blocks)
"sdxl": {
    "target_modules": ["Transformer2DModel"],
    "default_exclude_patterns": [r".*norm.*"],
},
# SD3 (MMDiT — targets joint + single blocks)
"sd3": {
    "target_modules": ["JointTransformerBlock", "SD3SingleTransformerBlock"],
    "default_exclude_patterns": [r".*norm.*", r".*ada_norm.*"],
},
# Flux 1 (same block names as Flux 2 but different architecture)
"flux_1": {
    "target_modules": ["Flux1DoubleStreamBlock", "Flux1SingleStreamBlock"],
    "default_exclude_patterns": [
        r".*(modulation).*",
        r".*(norm).*",
    ],
},
```

### Task D2: Update `tests/test_imports.py`

**Files:** Modify `tests/test_imports.py`

Add canary imports at the end:

```python
# Phase 5b: SDXL, SD3, Flux 1
def test_import_sdxl_strategy():
    from trainer.arch.sdxl.strategy import SDXLStrategy

def test_import_sdxl_configs():
    from trainer.arch.sdxl.components.configs import SDXL_CONFIGS

def test_import_sd3_strategy():
    from trainer.arch.sd3.strategy import SD3Strategy

def test_import_sd3_configs():
    from trainer.arch.sd3.components.configs import SD3_CONFIGS

def test_import_sd3_model():
    from trainer.arch.sd3.components.model import SD3Transformer2DModel

def test_import_flux1_strategy():
    from trainer.arch.flux_1.strategy import Flux1Strategy

def test_import_flux1_configs():
    from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS

def test_import_flux1_model():
    from trainer.arch.flux_1.components.model import Flux1Transformer
```

### Task D3: Run full test suite

```
python -m pytest tests/ -v
```

**Expected:** All tests pass. `list_models()` returns 12 architectures: wan, zimage, flux_2, qwen_image, flux_kontext, framepack, kandinsky5, hunyuan_video, hunyuan_video_1_5, sdxl, sd3, flux_1.

---

## Verification (Phase 5b Gate)

1. All tests pass: `python -m pytest tests/ -v`
2. `list_models()` returns 12 architectures
3. Each architecture's `training_step` works with mock batch (verified by per-arch tests)
4. SDXL epsilon prediction verified: target == noise (not noise - latents)
5. SD3 + Flux 1 flow matching verified: target == noise - latents
6. Flux 1 packs 16ch → 64ch (not 128ch like Flux 2)
7. `get_arch_config()` works for "sdxl", "sd3", "flux_1"

## Failure Modes

1. **SDXL diffusers import** — `from diffusers import UNet2DConditionModel` must be deferred (inside `setup()`) to avoid import-time GPU dependency
2. **SDXL discrete vs continuous timesteps** — SDXL uses `torch.long` timesteps 0..999, NOT float [0,1]. Mixing up breaks the noise schedule completely
3. **SD3 timestep scaling** — SD3 model expects timesteps scaled to [0,1000] while flow matching samples in [0,1]. Must multiply: `model_t = t * 1000`
4. **Flux 1 vs Flux 2 channel confusion** — Flux 1 has 16ch latents (packed to 64), Flux 2 has 128ch. Using wrong constants silently produces wrong shapes
5. **Block name collisions** — Flux 1 blocks must be named `Flux1DoubleStreamBlock` (not `DoubleStreamBlock`) to avoid confusing LoRA targeting with Flux 2/Kontext
6. **GEGLU dimension mismatch** — GEGLU halves the MLP hidden dimension. The input projection must produce 2× the hidden dim, or shapes won't match
