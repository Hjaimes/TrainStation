# Phase 5b: SDXL, SD3, Flux 1 — Design Document

**Date:** 2026-03-01
**Phase:** 5b (follows Phase 5a's 8 Musubi_Tuner architectures)
**Scope:** 3 architectures using our own ModelStrategy pattern, informed by AI-Toolkit reference.

---

## Goal

Add SDXL, SD3, and Flux 1 architectures to the training app. Unlike Phase 5a (which ported from Musubi_Tuner), these are built using our established pattern with AI-Toolkit as conceptual reference only — no code porting.

## Approach

- **SDXL:** Uses `diffusers.UNet2DConditionModel` directly (thin loader wrapper). First epsilon-prediction architecture.
- **SD3:** Custom MMDiT in our components (JointTransformerBlock). Flow matching.
- **Flux 1:** Custom model based on our Flux 2 structure (same block concepts, different dimensions). Flow matching.

All 3 follow the same `trainer/arch/{name}/` directory structure with `__init__.py`, `strategy.py`, and `components/`.

---

## Architecture: SDXL

**Registry name:** `sdxl`

| Property | Value |
|---|---|
| Backbone | `diffusers.UNet2DConditionModel` (not DiT) |
| Prediction | Epsilon (default) or v-prediction |
| Latent channels | 4, shape `[B, 4, H/8, W/8]` |
| VAE | AutoencoderKL, scaling_factor=0.13025 |
| Text encoders | CLIP-L (768d hidden) + CLIP-G (1280d hidden + 1280d pool) |
| Cross-attention | Concatenated TE1+TE2 hidden = 2048d |
| Special conditioning | `add_time_ids` (6 values: orig_size, crop, target_size) |
| Video | No (image-only) |
| Block swap | Not needed |

**Training step (epsilon mode):**
1. Latents `[B, 4, H/8, W/8]` from cache
2. Sample noise, sample discrete timesteps 0..999
3. `noisy = sqrt(alpha_bar_t) * latents + sqrt(1 - alpha_bar_t) * noise`
4. Build `add_time_ids` and `added_cond_kwargs`
5. `pred = unet(noisy, t, encoder_hidden_states, added_cond_kwargs)`
6. `target = noise` (epsilon) or `get_velocity(latents, noise, t)` (v-pred)
7. `loss = MSE(pred, target)`

**Components:**
```
trainer/arch/sdxl/components/
├── __init__.py
├── model.py      # load_sdxl_unet() — thin diffusers wrapper
├── configs.py    # Prediction type, VAE params, model variants
└── utils.py      # build_time_ids(), DDPM schedule helpers
```

---

## Architecture: SD3

**Registry name:** `sd3`

| Property | Value |
|---|---|
| Backbone | Custom SD3Transformer2DModel (MMDiT) |
| Prediction | Flow matching (rectified flow) |
| Latent channels | 16, shape `[B, 16, H/8, W/8]` |
| Patch size | 2 (effective spatial factor 16) |
| VAE | AutoencoderKL, 16ch, has shift_factor |
| Text encoders | CLIP-L (768d pool) + CLIP-G (1280d pool) + T5-XXL (4096d seq) |
| Pooled projections | cat(clip_l_pool, clip_g_pool) = 2048d |
| Video | No (image-only) |

**MMDiT blocks:**
- `JointTransformerBlock` — bidirectional text+image attention. Both streams attend to each other via concatenated Q/K/V then split.
- `SD3SingleTransformerBlock` — image-only blocks (SD3.5 variants only)
- `AdaLayerNormZero` — timestep+pooled conditioning modulation

**Variants:**
- SD3.0: All joint blocks, no single blocks
- SD3.5 Medium/Large: Joint + single blocks (controlled by `dual_attention_layers`)

**Training step:**
- Standard flow matching: `noisy = (1-t)*latents + t*noise`, `target = noise - latents`
- Forward: `model(hidden_states, timestep, encoder_hidden_states, pooled_projections)`
- MSE loss

**Components:**
```
trainer/arch/sd3/components/
├── __init__.py
├── model.py         # SD3Transformer2DModel + loader
├── configs.py       # SD3 variant configs
├── blocks.py        # JointTransformerBlock, SD3SingleTransformerBlock
├── embeddings.py    # PatchEmbed, CombinedTimestepTextProjEmbeddings
└── layers.py        # AdaLayerNormZero, AdaLayerNormContinuous, MLP
```

---

## Architecture: Flux 1

**Registry name:** `flux_1`

| Property | Value |
|---|---|
| Backbone | Custom Flux1Transformer (based on our Flux 2 structure) |
| Prediction | Flow matching (rectified flow) |
| Latent channels | 16 raw, packed 2x2 → 64 for transformer |
| Latent shape | `[B, 16, H/8, W/8]` → packed `[B, (H/16)*(W/16), 64]` |
| Text encoders | T5-XXL (4096d seq) + CLIP-L (768d pool) |
| RoPE | 3D `(16, 56, 56)` |
| Double blocks | 19 |
| Single blocks | 38 |
| Hidden size | 3072 (24 heads × 128 dim) |
| MLP ratio | 4.0, GEGLU activation |
| Modulation | Per-block AdaLayerNorm (not global like Flux 2) |
| Guidance | Dev=True, Schnell=False |
| Video | No (image-only) |
| Block swap | Supported |

**Key differences from Flux 2:**
- 16ch latents (not 32ch), packed to 64 (not 128)
- T5+CLIP text encoders (not Mistral+Qwen)
- 3D RoPE `(16,56,56)` (not 4D `(32,32,32,32)`)
- Per-block modulation (not global shared)
- GEGLU FFN (not SiLU-gated)
- 19+38 blocks (not 8+48)

**Components:**
```
trainer/arch/flux_1/components/
├── __init__.py
├── model.py       # Flux1Transformer
├── configs.py     # dev, schnell variant configs
├── blocks.py      # DoubleStreamBlock, SingleStreamBlock
├── embeddings.py  # 3D RoPE, PatchEmbed, timestep embed
└── utils.py       # pack/unpack latents, position IDs
```

---

## LoRA Target Configs

| Architecture | Target Modules | Exclude Patterns |
|---|---|---|
| `sdxl` | `Transformer2DModel` | `.*norm.*` |
| `sd3` | `JointTransformerBlock`, `SD3SingleTransformerBlock` | `.*norm.*`, `.*ada_norm.*` |
| `flux_1` | `DoubleStreamBlock`, `SingleStreamBlock` | `modulation.lin`, `norm` |

---

## Testing

Per-architecture: config validation, registry discovery, strategy properties, TinyMock training step.
SDXL gets extra test for epsilon prediction (target == noise).

**Gate:** `python -m pytest tests/ -v` passes, `list_models()` returns all 12 architectures.

## Execution

All 3 in parallel (independent directories). Then integration pass for `arch_configs.py`, `test_imports.py`, full suite.
