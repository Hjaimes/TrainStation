# Phase 5a: Port 8 Musubi_Tuner Architectures — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port 8 model architectures from Musubi_Tuner into the training app's strategy pattern, with isolated components, smoke tests, and LoRA configs.

**Architecture:** Each architecture gets its own directory under `trainer/arch/{name}/` with `__init__.py`, `strategy.py`, and `components/`. All follow the WanStrategy template. Tiered execution: Tier 1 (simple, parallel), Tier 2 (medium, parallel), Tier 3 (complex, sequential).

**Tech Stack:** Python, PyTorch, safetensors

---

## Reference Files

| File | Purpose |
|------|---------|
| `trainer/arch/wan/strategy.py` | **Template** — all strategies follow this pattern |
| `trainer/arch/wan/__init__.py` | Registration pattern |
| `trainer/arch/base.py` | `ModelStrategy`, `ModelComponents`, `TrainStepOutput` |
| `trainer/networks/arch_configs.py` | Already has entries for all 8 architectures |
| `tests/test_wan_components.py` | Component test template |
| `tests/test_wan_e2e.py` | E2E test template (TinyMockDiT, _mock_setup) |

**Porting Policy (from CLAUDE.md):**
- `print()` → `logger.info()`/`logger.warning()`
- Remove `logging.basicConfig()` — app configures logging centrally
- `torch.concat` → `torch.cat`
- Remove dead/commented-out code
- Pre-allocate buffers, cache constants, use `set` for O(1) lookups
- In-place tensor ops where safe (`.normal_()`, `.add_(alpha=)`)
- Return `.detach()` for metrics instead of `.item()`

---

## Milestone 1: Tier 1 — Simple Architectures (3 parallel agents)

### Task 1: Z-Image Architecture

**Registry name:** `zimage`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/zimage_train_network.py` — training script
- `Musubi_Tuner/src/musubi_tuner/zimage/zimage_model.py` — transformer model
- `Musubi_Tuner/src/musubi_tuner/zimage/zimage_config.py` — config presets
- `Musubi_Tuner/src/musubi_tuner/zimage/zimage_autoencoder.py` — VAE
- `Musubi_Tuner/src/musubi_tuner/zimage/zimage_utils.py` — utilities

**Create files:**
```
trainer/arch/zimage/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # ZImageTransformerBlock + loader
    ├── configs.py        # Variant presets
    └── vae.py            # ZImageVAE (shift=0.1159, scale=0.3611)
tests/test_zimage_components.py
```

**Key implementation details:**
- **Reversed flow matching:** target = `latents - noise` (NOT `noise - latents` like Wan)
- **Timestep reversal:** `(1000 - t) / 1000` — sends reversed timesteps to model
- **VAE normalization:** shift=0.1159, scale=0.3611 applied to latents
- **Text encoder:** Qwen3-4B (not part of component porting — uses cached embeddings)
- **SEQ_MULTI_OF:** 32 — sequence length must be multiple of 32
- **Image only:** `supports_video = False`, adds dummy frame dimension to latents
- **Latent channels:** 16

**`trainer/arch/zimage/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import ZImageStrategy

register_model("zimage")(ZImageStrategy)
```

**`trainer/arch/zimage/strategy.py` key differences from Wan:**
- `architecture` property returns `"zimage"`
- `supports_video` returns `False`
- `setup()` loads ZImageTransformer from configs, caches VAE shift/scale
- `training_step()`:
  - Latents have shape [B, C, H, W] — add dummy frame dim for model: [B, C, 1, H, W]
  - Apply VAE normalization: `latents = (latents - shift) / scale`
  - Reversed timestep: `model_timesteps = (1000 - timesteps) / 1000`
  - Target: `noise - latents` (same direction as Wan but opposite VAE transform)
  - MSE loss
  - No block swap needed (small model)

**Mock test pattern (TinyMockDiT):**
- Forward signature: `forward(self, x, t, context, seq_len)` — same as Wan
- Returns list of tensors per batch item
- Test verifies: scalar loss, finite, has loss/timestep_mean metrics
- Registry test: `"zimage" in list_models()`

---

### Task 2: Flux 2 Architecture

**Registry name:** `flux_2`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/flux_2_train_network.py` — training script
- `Musubi_Tuner/src/musubi_tuner/flux_2/flux2_models.py` — DoubleStreamBlock, SingleStreamBlock
- `Musubi_Tuner/src/musubi_tuner/flux_2/flux2_utils.py` — model loading utils
- `Musubi_Tuner/src/musubi_tuner/flux/flux_models.py` — shared Flux base models (if needed)

**Create files:**
```
trainer/arch/flux_2/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # Flux2Transformer (DoubleStreamBlock + SingleStreamBlock)
    ├── configs.py        # Model variants (dev, klein-4b, klein-9b, etc.)
    └── utils.py          # Position ID generation, packing helpers
tests/test_flux2_components.py
```

**Key implementation details:**
- **Dual-stream architecture:** DoubleStreamBlock processes img+txt, SingleStreamBlock merges
- **128-channel latents** (8x more than typical)
- **prc_img packing:** Pack spatial dims with channel dim
- **4D position IDs:** `(batch, channel, row, col)` generated from latent shape
- **Latent channels:** 128 (16 channels × 2×2×2 packing)
- **Text encoders:** Mistral3 + Qwen3 (cached)
- **Model variants:** dev, klein-4b, klein-9b, klein-base-4b, klein-base-9b
- **Block swap supported** — similar pattern to Wan

**`trainer/arch/flux_2/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import Flux2Strategy

register_model("flux_2")(Flux2Strategy)
```

**`trainer/arch/flux_2/strategy.py` key differences from Wan:**
- `architecture` returns `"flux_2"`
- `supports_video` returns `False` (image-only)
- `setup()`:
  - Load Flux2 transformer from variant configs
  - Pre-compute position ID templates from config
  - Block swap support (same lifecycle hooks as Wan)
- `training_step()`:
  - Pack latents into `prc_img` format
  - Generate position IDs from latent spatial dims
  - Standard flow matching: target = `noise - latents`
  - Forward: `model(img=prc_img, txt=txt_emb, timesteps=t, img_ids=img_ids, txt_ids=txt_ids)`
  - MSE loss

**Mock test pattern:**
- TinyMockDiT with forward signature matching Flux2: `forward(self, img, txt, timesteps, img_ids, txt_ids)`
- Returns tensor (not list — Flux models return single tensor)
- Config test: verify variant presets exist

---

### Task 3: Qwen Image Architecture

**Registry name:** `qwen_image`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/qwen_image_train_network.py` — training script
- `Musubi_Tuner/src/musubi_tuner/qwen_image/qwen_image_model.py` — QwenImageTransformerBlock
- `Musubi_Tuner/src/musubi_tuner/qwen_image/qwen_image_modules.py` — attention, MLP
- `Musubi_Tuner/src/musubi_tuner/qwen_image/qwen_image_autoencoder_kl.py` — VAE
- `Musubi_Tuner/src/musubi_tuner/qwen_image/qwen_image_utils.py` — utilities

**Create files:**
```
trainer/arch/qwen_image/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # QwenImageTransformerBlock + loader
    ├── configs.py        # Task mode presets (t2i, edit, layered)
    ├── vae.py            # QwenImageVAE (per-channel normalization)
    └── modules.py        # Attention, MLP blocks
tests/test_qwen_image_components.py
```

**Key implementation details:**
- **Three modes:** t2i (text-to-image), edit, layered — selected via config
- **2x2 patchification:** 16 channels → 64 channels via pixel unshuffle
- **Text encoder:** Qwen2.5-VL (cached embeddings with `img_shapes` and `txt_seq_lens`)
- **Per-channel VAE normalization:** Each channel has its own mean/std
- **Latent channels:** 16 (patchified to 64)
- **Image only:** `supports_video = False`
- **Forward requires:** `img_shapes` (list of [H,W]) and `txt_seq_lens` (list of int)

**`trainer/arch/qwen_image/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import QwenImageStrategy

register_model("qwen_image")(QwenImageStrategy)
```

**`trainer/arch/qwen_image/strategy.py` key differences from Wan:**
- `architecture` returns `"qwen_image"`
- `supports_video` returns `False`
- `setup()`:
  - Load QwenImage transformer
  - Cache patchification constants
  - Determine mode (t2i/edit/layered) from config kwargs
- `training_step()`:
  - Apply per-channel normalization to latents
  - Patchify latents: pixel_unshuffle 2x2 → 64 channels
  - Extract `img_shapes` and `txt_seq_lens` from batch
  - Standard flow matching: target = `noise - latents`
  - Forward: `model(x, t, context, img_shapes=..., txt_seq_lens=...)`
  - MSE loss

**Mock test pattern:**
- TinyMockDiT with forward: `forward(self, x, t, context, img_shapes=None, txt_seq_lens=None)`
- Config test: verify mode presets
- Registry: `"qwen_image" in list_models()`

---

### Task 4: Tier 1 Integration

After all 3 Tier 1 architectures are implemented:

1. **Update `tests/test_imports.py`** — add canary imports:
```python
# Phase 5: Tier 1 architectures
def test_import_zimage_strategy():
    from trainer.arch.zimage.strategy import ZImageStrategy

def test_import_flux2_strategy():
    from trainer.arch.flux_2.strategy import Flux2Strategy

def test_import_qwen_image_strategy():
    from trainer.arch.qwen_image.strategy import QwenImageStrategy
```

2. **Run tests:**
```bash
python -m pytest tests/test_zimage_components.py tests/test_flux2_components.py tests/test_qwen_image_components.py tests/test_imports.py -v
```

---

## Milestone 2: Tier 2 — Medium Architectures (3 parallel agents)

### Task 5: Flux Kontext Architecture

**Registry name:** `flux_kontext`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/flux_kontext_train_network.py` — training script
- `Musubi_Tuner/src/musubi_tuner/flux/flux_models.py` — shared Flux model code
- `Musubi_Tuner/src/musubi_tuner/flux/flux_utils.py` — shared Flux utilities

**Create files:**
```
trainer/arch/flux_kontext/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # FluxKontextTransformer (DoubleStreamBlock + SingleStreamBlock)
    ├── configs.py        # Kontext variant configs
    └── utils.py          # Position IDs, control image processing
tests/test_flux_kontext_components.py
```

**Key implementation details:**
- **Control image via sequence concatenation:** Control latents concatenated with noise latents along sequence dim
- **Text encoders:** T5-XXL + CLIP-L (cached)
- **control_lengths parameter:** Tells model where control sequence ends
- **Prediction slicing:** After forward pass, slice output to remove control portion
- **Same block types as Flux 2:** DoubleStreamBlock + SingleStreamBlock
- **Image only:** `supports_video = False`

**`trainer/arch/flux_kontext/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import FluxKontextStrategy

register_model("flux_kontext")(FluxKontextStrategy)
```

**`trainer/arch/flux_kontext/strategy.py` key differences from Wan:**
- `architecture` returns `"flux_kontext"`
- `supports_video` returns `False`
- `setup()`:
  - Load FluxKontext transformer
  - Cache control-related constants
- `training_step()`:
  - Extract control latents and target latents from batch
  - Concatenate control + noisy latents along sequence dim
  - Compute `control_lengths` tensor
  - Generate position IDs for full concatenated sequence
  - Forward: `model(img=concat_img, txt=txt, timesteps=t, img_ids=ids, txt_ids=txt_ids, control_lengths=control_lengths)`
  - Slice prediction to remove control portion
  - Standard flow matching loss on target portion only
  - MSE loss

**Self-contained:** Even though Flux Kontext shares block types with Flux 2, each architecture keeps its own copy in its components directory. No cross-architecture imports.

---

### Task 6: FramePack Architecture

**Registry name:** `framepack`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/fpack_train_network.py` — training script
- `Musubi_Tuner/src/musubi_tuner/frame_pack/hunyuan_video_packed.py` — HunyuanVideoTransformer3DModelPacked
- `Musubi_Tuner/src/musubi_tuner/frame_pack/hunyuan.py` — base HunyuanVideo blocks
- `Musubi_Tuner/src/musubi_tuner/frame_pack/clip_vision.py` — CLIP vision encoder
- `Musubi_Tuner/src/musubi_tuner/frame_pack/framepack_utils.py` — utilities
- `Musubi_Tuner/src/musubi_tuner/frame_pack/utils.py` — additional utilities

**Create files:**
```
trainer/arch/framepack/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # HunyuanVideoTransformer3DModelPacked
    ├── configs.py        # FramePack configs
    ├── blocks.py         # HunyuanVideoTransformerBlock, SingleTransformerBlock
    └── utils.py          # Temporal packing, RoPE, etc.
tests/test_framepack_components.py
```

**Key implementation details:**
- **Packed temporal format:** Multi-scale context (1x/2x/4x clean latents packed together)
- **Always I2V:** Image-to-video only, first frame is conditioning
- **HunyuanVideo-based blocks:** HunyuanVideoTransformerBlock + HunyuanVideoSingleTransformerBlock
- **CLIP vision encoder:** For image_embeddings conditioning
- **Mixed precision:** VAE runs in float16, DiT in bfloat16
- **Video architecture:** `supports_video = True`
- **Block swap supported**

**`trainer/arch/framepack/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import FramePackStrategy

register_model("framepack")(FramePackStrategy)
```

**`trainer/arch/framepack/strategy.py` key differences from Wan:**
- `architecture` returns `"framepack"`
- `supports_video` returns `True`
- `setup()`:
  - Load HunyuanVideoTransformer3DModelPacked
  - Setup CLIP vision encoder for image conditioning
  - Block swap lifecycle hooks (same as Wan pattern)
- `training_step()`:
  - Unpack temporal context from batch (multi-scale 1x/2x/4x)
  - Extract first frame as conditioning image
  - Generate image_embeddings via CLIP vision
  - Standard flow matching: target = `noise - latents`
  - Forward: `model(x, t, context, image_embeddings=...)`
  - MSE loss

---

### Task 7: Kandinsky 5 Architecture

**Registry name:** `kandinsky5`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/kandinsky5_train_network.py` — training script
- `Musubi_Tuner/src/musubi_tuner/kandinsky5/models/dit.py` — DiT model
- `Musubi_Tuner/src/musubi_tuner/kandinsky5/models/attention.py` — nabla/STA attention
- `Musubi_Tuner/src/musubi_tuner/kandinsky5/models/nn.py` — neural network blocks
- `Musubi_Tuner/src/musubi_tuner/kandinsky5/models/text_embedders.py` — text encoder
- `Musubi_Tuner/src/musubi_tuner/kandinsky5/models/vae.py` — VAE (stub pattern)
- `Musubi_Tuner/src/musubi_tuner/kandinsky5/models/utils.py` — utilities
- `Musubi_Tuner/src/musubi_tuner/kandinsky5/configs.py` — task-based configs

**Create files:**
```
trainer/arch/kandinsky5/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # DiT with TransformerEncoderBlock + TransformerDecoderBlock
    ├── configs.py        # Task-based configs (7 variants)
    ├── attention.py      # nabla/STA sparse attention
    ├── nn.py             # Neural network building blocks
    └── utils.py          # Utilities
tests/test_kandinsky5_components.py
```

**Key implementation details:**
- **Per-sample iteration:** Training iterates over samples individually (no batching in forward pass)
- **Task-based configs:** 7 variants (e.g., t2v, i2v, various resolutions)
- **Include-pattern LoRA:** Uses `default_include_patterns` in arch_configs (unique among all archs)
- **Channels-last format:** Tensors in (F, H, W, C) format, not (C, F, H, W)
- **VAE stub pattern:** VAE object exists but may not do actual encoding (latents pre-cached)
- **nabla/STA sparse attention:** Custom attention implementation
- **Visual conditioning:** Control signal via channel concatenation
- **Video architecture:** `supports_video = True`

**`trainer/arch/kandinsky5/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import Kandinsky5Strategy

register_model("kandinsky5")(Kandinsky5Strategy)
```

**`trainer/arch/kandinsky5/strategy.py` key differences from Wan:**
- `architecture` returns `"kandinsky5"`
- `supports_video` returns `True`
- `setup()`:
  - Load DiT from task-based config
  - Cache per-sample iteration flag
  - No block swap (architecture handles memory differently)
- `training_step()`:
  - **Per-sample loop:** Iterate over batch items individually
  - Convert from standard (B,C,F,H,W) to channels-last (F,H,W,C) if needed
  - Standard flow matching: target = `noise - latents`
  - Forward: `model(x, t, context, ...)` per sample
  - Accumulate loss across samples, divide by batch size
  - MSE loss

---

### Task 8: Tier 2 Integration

After all 3 Tier 2 architectures are implemented:

1. **Update `tests/test_imports.py`** — add canary imports:
```python
# Phase 5: Tier 2 architectures
def test_import_flux_kontext_strategy():
    from trainer.arch.flux_kontext.strategy import FluxKontextStrategy

def test_import_framepack_strategy():
    from trainer.arch.framepack.strategy import FramePackStrategy

def test_import_kandinsky5_strategy():
    from trainer.arch.kandinsky5.strategy import Kandinsky5Strategy
```

2. **Run tests:**
```bash
python -m pytest tests/test_flux_kontext_components.py tests/test_framepack_components.py tests/test_kandinsky5_components.py tests/test_imports.py -v
```

---

## Milestone 3: Tier 3 — Complex Architectures (sequential)

### Task 9: HunyuanVideo Architecture

**Registry name:** `hunyuan_video`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/hv_train_network.py` — training script (3096 lines)
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/models.py` — MMDoubleStreamBlock + MMSingleStreamBlock
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/attention.py` — attention implementation
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/posemb_layers.py` — RoPE position embeddings
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/embed_layers.py` — embedding layers
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/mlp_layers.py` — MLP layers
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/norm_layers.py` — normalization
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/activation_layers.py` — activations
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/modulate_layers.py` — modulation
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/vae.py` — VAE
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/text_encoder.py` — LLM text encoder
- `Musubi_Tuner/src/musubi_tuner/hunyuan_model/helpers.py` — helper functions
- `Musubi_Tuner/src/musubi_tuner/modules/custom_offloading_utils.py` — ModelOffloader for block swap

**Create files:**
```
trainer/arch/hunyuan_video/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # HunyuanVideoTransformer (20 double + 40 single blocks)
    ├── configs.py        # HV model configs
    ├── blocks.py         # MMDoubleStreamBlock, MMSingleStreamBlock
    ├── attention.py      # Attention with RoPE
    ├── embeddings.py     # Position embeddings (RoPE), timestep, text
    ├── layers.py         # MLP, norm, activation, modulation layers
    ├── vae.py            # HunyuanVideo VAE
    └── offloading.py     # ModelOffloader for block swap
tests/test_hunyuan_video_components.py
```

**Key implementation details:**
- **Largest architecture:** 20 double-stream + 40 single-stream blocks
- **Complex block swap:** Dual ModelOffloader (one for double blocks, one for single blocks)
- **RoPE position embeddings:** Cached cos/sin tensors for 3D positions
- **Text encoders:** LLM (Llama-based) + CLIP-L (cached)
- **guidance_embed:** Boolean flag, model has guidance embedding layer
- **Video architecture:** `supports_video = True`
- **Patch size:** (1, 2, 2) — spatial-only patching

**`trainer/arch/hunyuan_video/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import HunyuanVideoStrategy

register_model("hunyuan_video")(HunyuanVideoStrategy)
```

**`trainer/arch/hunyuan_video/strategy.py` key differences from Wan:**
- `architecture` returns `"hunyuan_video"`
- `supports_video` returns `True`
- `setup()`:
  - Load HunyuanVideoTransformer with all sub-modules
  - Pre-compute and cache RoPE cos/sin tensors (stored in `components.extra`)
  - Complex block swap setup with dual ModelOffloader
  - `guidance_embed=True` on model
- Lifecycle hooks:
  - `on_before_accelerate_prepare()` — disable device placement for block swap
  - `on_after_accelerate_prepare()` — re-place blocks, prepare swap
  - `on_before_training_step()` — prepare block swap before forward
- `training_step()`:
  - Extract latents [B, C, F, H, W] and text conditioning
  - Generate RoPE embeddings from latent spatial dimensions
  - Standard flow matching: target = `noise - latents`
  - Forward: `model(x, t, text_states, text_mask, freqs_cos, freqs_sin, guidance=...)`
  - MSE loss

---

### Task 10: HunyuanVideo 1.5 Architecture

**Registry name:** `hunyuan_video_1_5`
**Source files:**
- `Musubi_Tuner/src/musubi_tuner/hv_1_5_train_network.py` — training script (504 lines)
- `Musubi_Tuner/src/musubi_tuner/hunyuan_video_1_5/` — HV 1.5 specific components (if exists)
- Shares many patterns with HunyuanVideo but has key differences

**Create files:**
```
trainer/arch/hunyuan_video_1_5/
├── __init__.py
├── strategy.py
└── components/
    ├── __init__.py
    ├── model.py          # HV 1.5 Transformer (54 double blocks only)
    ├── configs.py        # HV 1.5 configs
    ├── blocks.py         # MMDoubleStreamBlock (self-contained copy)
    ├── attention.py      # Attention with RoPE
    ├── embeddings.py     # Position embeddings
    ├── layers.py         # MLP, norm, activation, modulation
    └── offloading.py     # Simpler block swap (single offloader)
tests/test_hv_1_5_components.py
```

**Key implementation details:**
- **54 double blocks only** — no single-stream blocks (unlike HunyuanVideo's 20+40)
- **patch_size = [1, 1, 1]** — no spatial patching
- **Text encoders:** Qwen2.5-VL + ByT5 (different from HV's LLM + CLIP-L)
- **No guidance embedding** — `guidance_embed=False`
- **Simpler block swap:** Single ModelOffloader (only double blocks)
- **Video architecture:** `supports_video = True`
- **Self-contained:** Copies its own block definitions even though HV shares similar ones

**`trainer/arch/hunyuan_video_1_5/__init__.py`:**
```python
from trainer.registry import register_model
from .strategy import HunyuanVideo15Strategy

register_model("hunyuan_video_1_5")(HunyuanVideo15Strategy)
```

**`trainer/arch/hunyuan_video_1_5/strategy.py` key differences from HunyuanVideo:**
- `architecture` returns `"hunyuan_video_1_5"`
- `setup()`:
  - Load HV 1.5 transformer (54 double blocks)
  - Simpler block swap with single offloader
  - No guidance embedding
  - Different text encoder format (Qwen2.5-VL + ByT5)
- `training_step()`:
  - Same flow matching pattern as HunyuanVideo
  - Different forward signature (no guidance parameter)
  - patch_size [1,1,1] affects latent dimension calculations

---

### Task 11: Tier 3 Integration

After both Tier 3 architectures are implemented:

1. **Update `tests/test_imports.py`** — add canary imports:
```python
# Phase 5: Tier 3 architectures
def test_import_hunyuan_video_strategy():
    from trainer.arch.hunyuan_video.strategy import HunyuanVideoStrategy

def test_import_hv_1_5_strategy():
    from trainer.arch.hunyuan_video_1_5.strategy import HunyuanVideo15Strategy
```

2. **Run tests:**
```bash
python -m pytest tests/test_hunyuan_video_components.py tests/test_hv_1_5_components.py tests/test_imports.py -v
```

---

## Milestone 4: Full Suite Verification

### Task 12: Run complete test suite

```bash
python -m pytest tests/ -v
```

**Expected:** 183 existing + ~60-80 new = ~250+ tests passing.

### Task 13: Verify registry

```python
from trainer.registry import list_models
models = list_models()
expected = {"wan", "zimage", "flux_2", "qwen_image", "flux_kontext", "framepack", "kandinsky5", "hunyuan_video", "hunyuan_video_1_5"}
assert expected.issubset(set(models))
```

---

## Per-Architecture Test Template

Every architecture test file follows this pattern (adapted from `tests/test_wan_components.py` and `tests/test_wan_e2e.py`):

```python
"""Tests for {name} architecture components."""
import torch
import pytest
from trainer.arch.base import ModelComponents, TrainStepOutput


class TestConfigs:
    def test_config_keys(self):
        from trainer.arch.{name}.components.configs import {CONFIGS_CONST}
        assert len({CONFIGS_CONST}) > 0

    def test_default_config_fields(self):
        from trainer.arch.{name}.components.configs import {CONFIGS_CONST}
        cfg = list({CONFIGS_CONST}.values())[0]
        # Verify key fields exist (arch-specific)


class TestRegistry:
    def test_registry_discovers(self):
        from trainer.registry import list_models
        assert "{registry_name}" in list_models()

    def test_registry_resolves(self):
        from trainer.registry import get_model_strategy
        cls = get_model_strategy("{registry_name}")
        assert cls.__name__ == "{StrategyClass}"

    def test_strategy_properties(self):
        from trainer.registry import get_model_strategy
        from trainer.config.schema import TrainConfig
        cls = get_model_strategy("{registry_name}")
        config = TrainConfig(
            model={"architecture": "{registry_name}", "base_model_path": "/fake"},
            training={"method": "full_finetune"},
            data={"datasets": [{"path": "/fake"}]},
        )
        strategy = cls(config)
        assert strategy.architecture == "{registry_name}"
        assert strategy.supports_video is {True/False}


class TinyMock{Name}(torch.nn.Module):
    """Tiny model mimicking {Name} forward signature."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)

    def forward(self, {arch_specific_args}):
        # Minimal forward that returns correct shape
        ...

    def enable_gradient_checkpointing(self):
        pass


def _mock_setup(strategy):
    """Replace strategy.setup() with mock model."""
    import math
    device = torch.device("cpu")
    # Set all cached attributes that setup() would set
    strategy._device = device
    strategy._train_dtype = torch.bfloat16
    strategy._blocks_to_swap = 0
    # ... arch-specific cached values ...

    model = TinyMock{Name}().to(device)
    return ModelComponents(model=model, extra={...})


class TestTrainingStep:
    def test_training_step_produces_loss(self, tmp_path):
        from trainer.config.schema import TrainConfig
        from trainer.arch.{name}.strategy import {StrategyClass}

        config = TrainConfig(
            model={"architecture": "{registry_name}", "base_model_path": "/fake",
                   "gradient_checkpointing": False},
            training={"method": "full_finetune", "timestep_sampling": "uniform"},
            data={"datasets": [{"path": str(tmp_path)}]},
        )
        strategy = {StrategyClass}(config)
        components = _mock_setup(strategy)

        batch = {
            "latents": torch.randn({arch_specific_shape}, dtype=torch.bfloat16),
            # ... arch-specific batch keys ...
        }

        output = strategy.training_step(components, batch, step=0)
        assert isinstance(output, TrainStepOutput)
        assert output.loss.dim() == 0  # scalar
        assert torch.isfinite(output.loss)
        assert "loss" in output.metrics
```

---

## Agent Instructions Summary

Each parallel agent implementing an architecture must:

1. **Read the Musubi_Tuner source files** listed in the task
2. **Port components** following the porting policy (logging, torch.cat, cleanup, optimization)
3. **Create strategy.py** following the WanStrategy template pattern exactly
4. **Create __init__.py** with `register_model("{name}")(Strategy)`
5. **Create smoke test** following the template above
6. **Verify** the architecture appears in `list_models()` and the mock training step produces a finite scalar loss

**Critical rules:**
- NO cross-architecture imports. Every arch is self-contained.
- NO modifications to files outside `trainer/arch/{name}/` and `tests/test_{name}_*.py`
- Each component directory has its own `__init__.py`
- Strategy classes cache all config values in `setup()` — `training_step()` only reads `self._*`
- All flow matching: target = `noise - latents` (except Z-Image which is `latents - noise`)

---

## Failure Modes

1. **Import cycles** — Components must not import from strategy; strategy imports from components
2. **Missing cached attributes** — If `training_step()` reads `self._foo`, `setup()` must set it. Mock setup must also set it.
3. **Wrong flow matching direction** — Z-Image uses reversed target; all others use `noise - latents`
4. **Cross-arch imports** — Even if HV and HV1.5 share code, each gets its own copy
5. **Registry collision** — Each architecture's `__init__.py` must use unique registry name
6. **Block swap lifecycle** — Missing `on_before_training_step()` causes GPU OOM with block swap enabled
