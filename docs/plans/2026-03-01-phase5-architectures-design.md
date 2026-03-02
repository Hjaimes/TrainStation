# Phase 5: Remaining Architectures — Design Document

**Date:** 2026-03-01
**Phase:** 5 of spec (Weeks 11-15)
**Scope:** 8 Musubi_Tuner architectures now, 3 AI-Toolkit architectures (SDXL/SD3/Flux1) deferred to Phase 5b after review.

---

## Goal

Port 8 model architectures from Musubi_Tuner into the training app's strategy pattern. Each architecture gets its own isolated directory under `trainer/arch/`, a LoRA config entry, and smoke tests. No shared/framework files are modified.

## Architectures (Phase 5a — Musubi_Tuner)

| Architecture | Registry Name | Train Script LOC | Tier | Key Challenge |
|---|---|---|---|---|
| Z-Image | `zimage` | 363 | 1 | Simplest, pattern validation |
| Flux 2 | `flux_2` | 363 | 1 | Clean flow matching |
| Qwen Image | `qwen_image` | 621 | 1 | Multi-encoder stacking |
| Flux Kontext | `flux_kontext` | 405 | 2 | Flux variant |
| FramePack | `frame_pack` | 637 | 2 | Packed temporal format |
| Kandinsky 5 | `kandinsky5` | 982 | 2 | Include-pattern LoRA targeting |
| HunyuanVideo | `hunyuan_video` | 3096 | 3 | Complex block swap, largest |
| HunyuanVideo 1.5 | `hv_1_5` | 504 | 3 | Massive components, shares HV patterns |

## Deferred (Phase 5b — AI-Toolkit)

| Architecture | Source | Notes |
|---|---|---|
| SDXL | AI-Toolkit | UNet-based, epsilon prediction (not flow matching) |
| SD3 | AI-Toolkit | 3 text encoders, no LoRA in reference |
| Flux 1 | AI-Toolkit | GPU splitting, dual-stream |

These are deferred until the Musubi_Tuner architectures are complete and reviewed.

---

## Per-Architecture Pattern

Every architecture follows this structure:

```
trainer/arch/{name}/
├── __init__.py          # register_model("{name}")(Strategy)
├── strategy.py          # ModelStrategy subclass
└── components/
    ├── __init__.py
    ├── model.py         # Transformer/DiT + loader
    ├── configs.py       # Variant presets
    ├── vae.py           # VAE (if arch-specific)
    └── *.py             # Text encoders, attention, utils
```

Plus:
- Entry in `trainer/networks/arch_configs.py`
- Tests in `tests/test_{name}_components.py`
- Canary import in `tests/test_imports.py`

### Strategy Requirements

All implement `ModelStrategy` from `trainer/arch/base.py`:
- `architecture` property — registry name
- `setup()` → `ModelComponents` — load model, VAE, text encoders
- `training_step(components, batch, step)` → `TrainStepOutput` — noise, timesteps, forward, loss
- Lifecycle hooks for block swap (HunyuanVideo, Flux 2)

All Musubi_Tuner architectures use **flow matching**: `target = noise - latents`, `loss = MSE(pred, target)`.

---

## Component Porting Strategy

**What gets ported:**
- Model definitions (transformer/DiT)
- VAE encoders
- Text encoders (T5, CLIP, Mistral, XLM-RoBERTa)
- Config presets, attention implementations, utility functions

**What does NOT get ported:**
- Inference pipelines / generation scripts
- Flow matching solvers (40K+ lines, sampling concern for later)
- GUI code, cache scripts (we have our own)

**Porting policy (from CLAUDE.md):**
- `print()` → `logger.info()`/`logger.warning()`
- Remove `logging.basicConfig()`
- `torch.concat` → `torch.cat`
- Remove dead/commented-out code
- Pre-allocate buffers, cache constants, use `set` for O(1) lookups

**Self-contained:** Each architecture keeps its own components directory. No cross-architecture imports, even when architectures share code (HV/HV1.5, Flux/Kontext).

---

## Execution Plan

### Tier 1 — Simple (3 parallel agents)
Z-Image, Flux 2, Qwen Image

### Tier 2 — Medium (3 parallel agents)
Flux Kontext, FramePack, Kandinsky 5

### Tier 3 — Complex (2 sequential)
HunyuanVideo, then HunyuanVideo 1.5

### After each tier:
- Update `arch_configs.py` with LoRA entries
- Update `test_imports.py` with canary imports
- Run full test suite

---

## Testing

**Per-architecture:**
- `test_{name}_components.py` — canary imports, config validation
- Smoke test with `TinyMock` model (same pattern as Wan E2E tests)
- Verify `training_step()` produces scalar loss with gradients

**Gate:** `python -m pytest tests/ -v` passes, `list_models()` returns all registered architectures.

---

## LoRA Target Configs

| Architecture | Target Modules | Exclude Patterns |
|---|---|---|
| zimage | `ZImageTransformerBlock` | `.*(_modulation\|_refiner).*` |
| flux_2 | `DoubleStreamBlock`, `SingleStreamBlock` | `modulation.lin`, `norm` |
| qwen_image | `QwenImageTransformerBlock` | `.*(_mod_).*` |
| flux_kontext | `DoubleStreamBlock`, `SingleStreamBlock` | `modulation.lin`, `norm` |
| frame_pack | `HunyuanVideoTransformerBlock`, `HunyuanVideoSingleTransformerBlock` | `.*norm.*` |
| kandinsky5 | `TransformerEncoderBlock`, `TransformerDecoderBlock` | **Uses include patterns** |
| hunyuan_video | `MMDoubleStreamBlock`, `MMSingleStreamBlock` | `.*(_in).*` |
| hv_1_5 | `MMDoubleStreamBlock` | `.*(_in).*` |
