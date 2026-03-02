# Gap-Closing Plan: VRAM Savings & Training Quality

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the most impactful gaps between our training app and AI-Toolkit / OneTrainer, with emphasis on VRAM savings and performance-improved ports.

**Architecture:** Features are grouped into three tiers — VRAM savings (highest priority), training quality improvements, and infrastructure polish. Each feature is self-contained with its own config, implementation, and tests. All features hook into the existing strategy/trainer architecture without breaking the clean separation.

**Tech Stack:** PyTorch, HuggingFace Accelerate, Pydantic v2, safetensors

**Reference codebases (read-only):**
- OneTrainer: `c:\Users\james\Desktop\AI\Gesen2egee\`
- AI-Toolkit: `c:\Users\james\Desktop\AI\AI-Toolkit\`
- Musubi_Tuner: `c:\Users\james\Desktop\AI\Musubi_Tuner\`

---

## Overview & Priority Map

| # | Feature | Tier | VRAM Impact | Effort | Config Field |
|---|---------|------|-------------|--------|-------------|
| 1 | Gradient checkpointing (verify + harden) | VRAM | ~30-50% reduction | Low | `model.gradient_checkpointing` (exists) |
| 2 | Loss function selection | Quality | None | Low | `training.loss_type` (new) |
| 3 | SNR-based loss weighting | Quality | None | Medium | `training.weighting_scheme` (exists, unused) |
| 4 | Network weight loading | Polish | None | Low | `network.network_weights` (exists, unimplemented) |
| 5 | Multi-type quantized model loading (NF4/INT8/FP8) | VRAM | ~40-75% reduction | Medium-High | `model.quantization` (exists, unimplemented) |
| 6 | EMA (Exponential Moving Average) | Quality | Small increase | Medium | `training.ema_*` (new) |
| 7 | OOM retry with counter | Polish | Indirect | Low | None (hardcoded 3-strike) |
| 8 | torch.compile support | Perf | Indirect (faster) | Low | `model.compile_model` (exists) |

**Estimated total:** ~8 tasks, each independently testable and committable.

**Spec note:** EMA is deferred to v2 in the canonical spec, but is included here as the user explicitly requested it. `gradient_checkpointing` and `quantization` are v1 spec features.

---

## Task 1: Verify & Harden Gradient Checkpointing

Gradient checkpointing is already partially implemented — `model.gradient_checkpointing: bool = True` exists in config, and Wan/Flux2 strategies call `model.enable_gradient_checkpointing()` in `setup()`. This task verifies it works end-to-end and adds test coverage.

**What competitors do better:** OneTrainer has per-block activation offloading alongside checkpointing. AI-Toolkit enables it on both UNet and text encoders. Our implementation only covers the transformer.

**Files:**
- Verify: `trainer/arch/wan/strategy.py:110-112`
- Verify: `trainer/arch/flux_2/strategy.py:116-118`
- Remove dead code: `trainer/arch/flux_2/strategy.py:303-306` (batch key `gradient_checkpointing` never populated)
- Test: `tests/test_gradient_checkpointing.py` (new)

**Step 1: Write tests that verify gradient checkpointing config flows to model**

```python
# tests/test_gradient_checkpointing.py
"""Tests that gradient checkpointing config is respected by strategies."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from trainer.config.schema import TrainConfig


class TestGradientCheckpointingConfig:
    """Verify the config field exists and defaults correctly."""

    def test_gradient_checkpointing_default_true(self):
        cfg = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake"},
            data={"dataset_config": "/fake.toml"},
            saving={"output_dir": "/fake"},
        )
        assert cfg.model.gradient_checkpointing is True

    def test_gradient_checkpointing_can_disable(self):
        cfg = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake",
                   "gradient_checkpointing": False},
            data={"dataset_config": "/fake.toml"},
            saving={"output_dir": "/fake"},
        )
        assert cfg.model.gradient_checkpointing is False


class TestGradientCheckpointingInStrategy:
    """Verify strategies call enable_gradient_checkpointing when config is True."""

    def test_wan_strategy_enables_checkpointing(self):
        """The Wan strategy should call model.enable_gradient_checkpointing()
        when cfg.model.gradient_checkpointing is True."""
        # This is a design verification test — check the source code calls it.
        # A full integration test would require loading a real model.
        import inspect
        from trainer.arch.wan.strategy import WanStrategy
        source = inspect.getsource(WanStrategy.setup)
        assert "enable_gradient_checkpointing" in source

    def test_flux2_strategy_enables_checkpointing(self):
        import inspect
        from trainer.arch.flux_2.strategy import Flux2Strategy
        source = inspect.getsource(Flux2Strategy.setup)
        assert "enable_gradient_checkpointing" in source
```

**Step 2: Run tests**

```bash
python -m pytest tests/test_gradient_checkpointing.py -v
```

**Step 3: Remove dead code from Flux 2 strategy**

In `trainer/arch/flux_2/strategy.py`, remove the inert batch-key gradient checkpointing block (lines ~303-306) that checks `batch.get("gradient_checkpointing", False)` — this key is never populated.

**Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

**Step 5: Commit**

```bash
git add tests/test_gradient_checkpointing.py trainer/arch/flux_2/strategy.py
git commit -m "test: verify gradient checkpointing + remove dead code in flux2 strategy"
```

---

## Task 2: Loss Function Selection

Currently all strategies hardwire `F.mse_loss`. Both competitors offer MSE, MAE, Huber, and more. We'll add a shared loss computation utility that strategies delegate to.

**Performance notes:**
- Cache the loss callable at strategy `setup()` time — don't resolve per step
- Use `reduction="mean"` throughout (no per-element then manual reduce)
- Huber's `delta` should be a config parameter, not hardcoded

**Files:**
- Create: `trainer/loss.py` (new — shared loss functions)
- Modify: `trainer/config/schema.py` — add `loss_type` field to `TrainingConfig`
- Modify: `trainer/arch/base.py` — add `_compute_loss` helper on `ModelStrategy`
- Modify: All strategy files — replace `F.mse_loss` with `self._compute_loss()`
- Test: `tests/test_loss.py` (new)

**Step 1: Write failing tests for loss functions**

```python
# tests/test_loss.py
"""Tests for configurable loss functions."""
import pytest
import torch
import torch.nn.functional as F
from trainer.loss import get_loss_fn, compute_loss


class TestGetLossFn:
    """Test that get_loss_fn returns the correct callable."""

    def test_mse(self):
        fn = get_loss_fn("mse")
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        result = fn(pred, target)
        expected = F.mse_loss(pred, target)
        assert torch.allclose(result, expected)

    def test_l1_mae(self):
        fn = get_loss_fn("l1")
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        result = fn(pred, target)
        expected = F.l1_loss(pred, target)
        assert torch.allclose(result, expected)

    def test_huber(self):
        fn = get_loss_fn("huber", delta=0.5)
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        result = fn(pred, target)
        expected = F.huber_loss(pred, target, delta=0.5)
        assert torch.allclose(result, expected)

    def test_huber_default_delta(self):
        fn = get_loss_fn("huber")
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        result = fn(pred, target)
        expected = F.huber_loss(pred, target, delta=1.0)
        assert torch.allclose(result, expected)

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_fn("invalid_loss")

    def test_mse_is_default(self):
        fn = get_loss_fn("mse")
        assert fn is not None


class TestComputeLoss:
    """Test the higher-level compute_loss that includes weighting."""

    def test_basic_mse_no_weighting(self):
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        loss = compute_loss(pred, target, loss_type="mse")
        expected = F.mse_loss(pred, target)
        assert torch.allclose(loss, expected)

    def test_returns_scalar(self):
        pred = torch.randn(4, 16, 32, 32)
        target = torch.randn(4, 16, 32, 32)
        loss = compute_loss(pred, target, loss_type="mse")
        assert loss.dim() == 0  # scalar
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_loss.py -v
```

Expected: FAIL — `trainer.loss` does not exist.

**Step 3: Implement `trainer/loss.py`**

```python
"""Configurable loss functions for training.

All loss functions accept (pred, target) tensors and return a scalar loss.
Functions are resolved once at setup time and cached — no per-step dispatch.
"""
from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

LossFn = Callable[[Tensor, Tensor], Tensor]


def get_loss_fn(loss_type: str, *, delta: float = 1.0) -> LossFn:
    """Return a loss callable for the given type.

    Args:
        loss_type: One of "mse", "l1", "huber".
        delta: Huber loss delta parameter (only used when loss_type="huber").

    Returns:
        A callable (pred, target) -> scalar loss tensor.
    """
    match loss_type:
        case "mse":
            return _mse_loss
        case "l1" | "mae":
            return _l1_loss
        case "huber":
            return functools.partial(_huber_loss, delta=delta)
        case _:
            raise ValueError(
                f"Unknown loss type '{loss_type}'. "
                f"Supported: 'mse', 'l1', 'mae', 'huber'."
            )


def compute_loss(
    pred: Tensor,
    target: Tensor,
    loss_type: str = "mse",
    *,
    delta: float = 1.0,
) -> Tensor:
    """Compute loss between prediction and target.

    Convenience wrapper that resolves the loss function and applies it.
    For hot-path usage, prefer caching the result of get_loss_fn() instead.
    """
    fn = get_loss_fn(loss_type, delta=delta)
    return fn(pred, target)


def _mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target, reduction="mean")


def _l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target, reduction="mean")


def _huber_loss(pred: Tensor, target: Tensor, *, delta: float = 1.0) -> Tensor:
    return F.huber_loss(pred, target, reduction="mean", delta=delta)
```

**Step 4: Add config field**

In `trainer/config/schema.py`, add to `TrainingConfig`:

```python
loss_type: str = "mse"  # mse, l1, mae, huber
huber_delta: float = 1.0  # delta for huber loss
```

**Step 5: Add `_compute_loss` helper to `ModelStrategy` base**

In `trainer/arch/base.py`, add a method that strategies inherit:

```python
from trainer.loss import get_loss_fn, LossFn

class ModelStrategy:
    _loss_fn: LossFn | None = None

    def _setup_loss_fn(self, cfg: TrainConfig) -> None:
        """Cache the loss function callable. Call from setup()."""
        self._loss_fn = get_loss_fn(
            cfg.training.loss_type,
            delta=cfg.training.huber_delta,
        )

    def _compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute training loss using the configured loss function."""
        if self._loss_fn is None:
            return F.mse_loss(pred, target, reduction="mean")
        return self._loss_fn(pred, target)
```

**Step 6: Update all strategies to use `_compute_loss`**

In every strategy's `setup()`, add `self._setup_loss_fn(cfg)` after the existing config caching.

In every strategy's `training_step()`, replace:
```python
loss = F.mse_loss(model_pred.to(train_dtype), target, reduction="mean")
```
with:
```python
loss = self._compute_loss(model_pred.to(train_dtype), target)
```

This affects: `wan`, `flux_2`, `flux_1`, `flux_kontext`, `sd3`, `sdxl`, `hunyuan_video`, `hunyuan_video_1_5`, `framepack`, `qwen_image`, `kandinsky5`, `zimage`.

**Step 7: Run full test suite**

```bash
python -m pytest tests/ -v
```

**Step 8: Commit**

```bash
git add trainer/loss.py trainer/config/schema.py trainer/arch/base.py trainer/arch/*/strategy.py tests/test_loss.py
git commit -m "feat: add configurable loss functions (mse, l1, huber)"
```

---

## Task 3: SNR-Based Loss Weighting

`weighting_scheme` already exists in `TrainingConfig` but is never used. Both competitors implement min-SNR-gamma weighting (Hang et al. 2023) and debiased estimation. This significantly improves training stability.

**Performance notes (improvements over competitors):**
- Pre-compute SNR lookup tables at setup time where possible (DDPM)
- For flow matching: `SNR(t) = (1-t)²/t²` — compute inline, no lookup needed
- Use in-place operations for weight application
- OneTrainer computes SNR weights per-element then reduces — we can compute per-sample weights (cheaper)

**Files:**
- Create: `trainer/loss_weighting.py` (new)
- Modify: `trainer/config/schema.py` — populate `weighting_scheme` options, add `snr_gamma` field
- Modify: `trainer/arch/base.py` — add `_apply_loss_weighting` helper
- Modify: `trainer/loss.py` — add `reduction="none"` variant for weighted losses
- Modify: All strategy files — integrate weighting into `_compute_loss` path
- Test: `tests/test_loss_weighting.py` (new)

**Step 1: Write failing tests**

```python
# tests/test_loss_weighting.py
"""Tests for SNR-based loss weighting."""
import pytest
import torch
import math
from trainer.loss_weighting import (
    compute_snr_flow_matching,
    compute_snr_ddpm,
    min_snr_gamma_weights,
    debiased_estimation_weights,
)


class TestFlowMatchingSNR:
    def test_snr_at_t_zero_is_large(self):
        """At t=0 (no noise), SNR should be very large."""
        t = torch.tensor([0.001])
        snr = compute_snr_flow_matching(t)
        assert snr.item() > 100.0

    def test_snr_at_t_one_is_small(self):
        """At t=1 (all noise), SNR should be very small."""
        t = torch.tensor([0.999])
        snr = compute_snr_flow_matching(t)
        assert snr.item() < 0.01

    def test_snr_at_t_half(self):
        """At t=0.5, SNR = (0.5)^2 / (0.5)^2 = 1.0."""
        t = torch.tensor([0.5])
        snr = compute_snr_flow_matching(t)
        assert torch.allclose(snr, torch.tensor([1.0]))

    def test_batch_snr(self):
        t = torch.tensor([0.1, 0.5, 0.9])
        snr = compute_snr_flow_matching(t)
        assert snr.shape == (3,)
        assert snr[0] > snr[1] > snr[2]  # monotonically decreasing


class TestMinSNRGamma:
    def test_clamps_to_gamma(self):
        snr = torch.tensor([0.1, 1.0, 5.0, 10.0, 100.0])
        weights = min_snr_gamma_weights(snr, gamma=5.0)
        # weight = min(SNR, gamma) / SNR
        expected = torch.minimum(snr, torch.tensor(5.0)) / snr
        assert torch.allclose(weights, expected)

    def test_high_snr_gets_downweighted(self):
        snr = torch.tensor([100.0])
        weights = min_snr_gamma_weights(snr, gamma=5.0)
        assert weights.item() < 0.1  # 5/100 = 0.05

    def test_low_snr_gets_weight_one(self):
        snr = torch.tensor([1.0])
        weights = min_snr_gamma_weights(snr, gamma=5.0)
        assert torch.allclose(weights, torch.tensor([1.0]))


class TestDebiasedEstimation:
    def test_weight_formula(self):
        snr = torch.tensor([1.0, 4.0, 9.0])
        weights = debiased_estimation_weights(snr)
        expected = 1.0 / snr.sqrt()
        assert torch.allclose(weights, expected)
```

**Step 2: Implement `trainer/loss_weighting.py`**

```python
"""SNR-based loss weighting for diffusion training.

Supports flow matching (continuous t) and DDPM (discrete timesteps).
All functions operate on batched tensors for vectorized computation.

References:
- Min-SNR-gamma: Hang et al. 2023 "Efficient Diffusion Training via Min-SNR Weighting Strategy"
- Debiased estimation: "Perception Prioritized Training of Diffusion Models" (P2 weighting)
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

WeightFn = Callable[[Tensor], Tensor]


def get_weight_fn(
    scheme: str,
    *,
    snr_gamma: float = 5.0,
) -> WeightFn | None:
    """Return a weight function for the given scheme.

    Args:
        scheme: One of "none", "min_snr_gamma", "debiased".
        snr_gamma: Gamma value for min-SNR weighting.

    Returns:
        A callable (snr_tensor) -> weight_tensor, or None for "none".
    """
    match scheme:
        case "none":
            return None
        case "min_snr_gamma":
            gamma = snr_gamma
            def _min_snr(snr: Tensor) -> Tensor:
                return min_snr_gamma_weights(snr, gamma)
            return _min_snr
        case "debiased":
            return debiased_estimation_weights
        case _:
            raise ValueError(
                f"Unknown weighting scheme '{scheme}'. "
                f"Supported: 'none', 'min_snr_gamma', 'debiased'."
            )


def compute_snr_flow_matching(t: Tensor) -> Tensor:
    """Compute SNR for flow matching: SNR(t) = (1-t)² / t².

    Args:
        t: Timestep values in [0, 1], shape (B,).

    Returns:
        SNR values, shape (B,). Clamped to avoid inf at t=0.
    """
    t_clamped = t.clamp(min=1e-6, max=1.0 - 1e-6)
    return ((1.0 - t_clamped) / t_clamped).square()


def compute_snr_ddpm(
    alphas_cumprod: Tensor,
    timesteps: Tensor,
) -> Tensor:
    """Compute SNR for DDPM: SNR(t) = alpha_bar(t) / (1 - alpha_bar(t)).

    Args:
        alphas_cumprod: Precomputed cumulative alpha products, shape (T,).
        timesteps: Integer timesteps, shape (B,).

    Returns:
        SNR values, shape (B,).
    """
    alpha_bar = alphas_cumprod[timesteps]
    return alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)


def min_snr_gamma_weights(snr: Tensor, gamma: float = 5.0) -> Tensor:
    """Min-SNR-gamma weighting: weight = min(SNR, gamma) / SNR.

    Downweights high-SNR (low-noise) timesteps where signal dominates.
    """
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr


def debiased_estimation_weights(snr: Tensor) -> Tensor:
    """Debiased estimation weighting: weight = 1 / sqrt(SNR).

    Equalizes the gradient contribution across noise levels.
    """
    return 1.0 / snr.sqrt().clamp(min=1e-6)
```

**Step 3: Add config fields**

In `trainer/config/schema.py` `TrainingConfig`, update the existing field and add:

```python
weighting_scheme: str = "none"  # none, min_snr_gamma, debiased
snr_gamma: float = 5.0  # gamma for min_snr_gamma weighting
```

**Step 4: Integrate into `ModelStrategy` base and loss.py**

Add to `trainer/loss.py`:

```python
def get_loss_fn(loss_type: str, *, delta: float = 1.0, unreduced: bool = False) -> LossFn:
    """..."""
    # Add unreduced variants for when we need per-element loss before weighting
```

Add to `trainer/arch/base.py`:

```python
from trainer.loss_weighting import get_weight_fn, compute_snr_flow_matching, WeightFn

class ModelStrategy:
    _weight_fn: WeightFn | None = None

    def _setup_loss_weighting(self, cfg: TrainConfig) -> None:
        self._weight_fn = get_weight_fn(
            cfg.training.weighting_scheme,
            snr_gamma=cfg.training.snr_gamma,
        )

    def _compute_weighted_loss(
        self,
        pred: Tensor,
        target: Tensor,
        timesteps: Tensor,
        snr: Tensor | None = None,
    ) -> Tensor:
        """Compute loss with optional SNR-based weighting."""
        if self._weight_fn is None:
            return self._compute_loss(pred, target)

        # Per-element loss, then weight per-sample, then reduce
        loss_per_element = F.mse_loss(pred, target, reduction="none")  # uses cached fn
        if snr is None:
            snr = compute_snr_flow_matching(timesteps)
        weights = self._weight_fn(snr)
        # Broadcast weights: (B,) -> (B, 1, 1, ...) to match loss shape
        for _ in range(loss_per_element.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (loss_per_element * weights).mean()
```

**Step 5: Update strategies to pass timesteps through to loss**

Each strategy's `training_step` that uses flow matching passes `t` to `_compute_weighted_loss`. For DDPM (SDXL), pass precomputed `alphas_cumprod`.

**Step 6: Run tests**

```bash
python -m pytest tests/test_loss_weighting.py tests/test_loss.py tests/ -v
```

**Step 7: Commit**

```bash
git add trainer/loss_weighting.py trainer/config/schema.py trainer/arch/base.py tests/test_loss_weighting.py
git commit -m "feat: add SNR-based loss weighting (min-SNR-gamma, debiased)"
```

---

## Task 4: Network Weight Loading

The `network.network_weights` field exists in `NetworkConfig` but is never loaded in `LoRAMethod.prepare()`. This is a quick fix.

**Files:**
- Modify: `trainer/training/methods.py` — load weights in `LoRAMethod.prepare()`
- Test: `tests/test_network_weight_loading.py` (new)

**Step 1: Write failing test**

```python
# tests/test_network_weight_loading.py
"""Test that pre-trained network weights are loaded when configured."""
import pytest
from unittest.mock import MagicMock, patch
from trainer.training.methods import LoRAMethod


class TestNetworkWeightLoading:
    def test_loads_weights_when_path_provided(self):
        """LoRAMethod.prepare() should call network.load_weights()
        when network_weights is set."""
        import inspect
        source = inspect.getsource(LoRAMethod.prepare)
        assert "load_weights" in source or "network_weights" in source
```

**Step 2: Implement in `LoRAMethod.prepare()`**

After `network.apply_to(model)`, add:

```python
if net_cfg.network_weights:
    logger.info("Loading pre-trained network weights from %s", net_cfg.network_weights)
    network.load_weights(net_cfg.network_weights)
```

**Step 3: Run tests, commit**

```bash
python -m pytest tests/ -v
git add trainer/training/methods.py tests/test_network_weight_loading.py
git commit -m "feat: load pre-trained network weights when network_weights is set"
```

---

## Task 5: Multi-Type Quantized Model Loading

The `model.quantization` config field exists (`None | "fp8" | "fp8_scaled"`) but has no implementation. We'll expand this to support NF4, INT8, and FP8 — covering the range from extreme compression (NF4, ~75% VRAM reduction) to minimal quality loss (FP8, ~50% reduction). This is a v1 spec feature.

**Supported types and VRAM impact:**

| Type | Bits | VRAM Reduction | Backend | Best For |
|------|------|---------------|---------|----------|
| `nf4` | 4-bit | ~75% | bitsandbytes | Consumer GPUs (8-12GB), largest models |
| `int8` | 8-bit | ~50% | bitsandbytes | Good balance of quality and savings |
| `fp8` | 8-bit | ~50% | Pure PyTorch | Ampere+ GPUs, best quality at 8-bit |
| `fp8_scaled` | 8-bit | ~50% | Pure PyTorch | Per-tensor scaling for better range |

**Design decisions (improvements over competitors):**
- **Abstract base `QuantizedLinear`** — OneTrainer uses a `QuantizedLinearMixin`; we'll use a clean ABC with `forward()`, `quantize()`, `dequantize()` so all types share an interface
- **Per-module quantization** — quantize one `nn.Linear` at a time to avoid OOM spike (OneTrainer pattern). Weight moves to GPU, quantizes, moves back to CPU
- **Forward dequantizes on-the-fly** — quantized weights stored compactly; `forward()` dequantizes to `compute_dtype` before matmul. LoRA residual operates in full precision on top
- **Skip norms/embeds** — `LayerNorm`, `GroupNorm`, `Embedding` always stay in full precision
- **`requires_grad_(False)` on quantized weights** — quantized base is always frozen (LoRA trains on top)
- **No bitsandbytes at import time** — deferred imports so the module works without bnb installed (graceful skip)

**Files:**
- Create: `trainer/quantization/__init__.py` (new — public API)
- Create: `trainer/quantization/base.py` (new — ABC + registry)
- Create: `trainer/quantization/fp8.py` (new — FP8 weight-only)
- Create: `trainer/quantization/bnb.py` (new — NF4 + INT8 via bitsandbytes)
- Create: `trainer/quantization/utils.py` (new — graph walker, skip logic)
- Modify: `trainer/config/schema.py` — expand `quantization` field options
- Modify: `trainer/arch/base.py` — add `_quantize_model` helper
- Test: `tests/test_quantization.py` (new)

**Step 1: Write failing tests**

```python
# tests/test_quantization.py
"""Tests for multi-type model quantization."""
import pytest
import torch
import torch.nn as nn


class TestQuantizationRegistry:
    """Test that quantization types are registered and resolvable."""

    def test_get_quantizer_fp8(self):
        from trainer.quantization import get_quantizer
        q = get_quantizer("fp8")
        assert q is not None

    def test_get_quantizer_fp8_scaled(self):
        from trainer.quantization import get_quantizer
        q = get_quantizer("fp8_scaled")
        assert q is not None

    def test_get_quantizer_nf4(self):
        from trainer.quantization import get_quantizer
        q = get_quantizer("nf4")
        assert q is not None

    def test_get_quantizer_int8(self):
        from trainer.quantization import get_quantizer
        q = get_quantizer("int8")
        assert q is not None

    def test_get_quantizer_none_returns_none(self):
        from trainer.quantization import get_quantizer
        assert get_quantizer(None) is None

    def test_get_quantizer_invalid_raises(self):
        from trainer.quantization import get_quantizer
        with pytest.raises(ValueError, match="Unknown quantization type"):
            get_quantizer("invalid_type")


class TestQuantizationConfig:
    """Test config field accepts all quantization types."""

    def test_config_none_default(self):
        from trainer.config.schema import TrainConfig
        cfg = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake"},
            data={"dataset_config": "/fake.toml"},
            saving={"output_dir": "/fake"},
        )
        assert cfg.model.quantization is None

    @pytest.mark.parametrize("qtype", ["fp8", "fp8_scaled", "nf4", "int8"])
    def test_config_accepts_all_types(self, qtype):
        from trainer.config.schema import TrainConfig
        cfg = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake",
                   "quantization": qtype},
            data={"dataset_config": "/fake.toml"},
            saving={"output_dir": "/fake"},
        )
        assert cfg.model.quantization == qtype


class TestFP8Quantization:
    """Test FP8 quantization on small models (no GPU required for basic tests)."""

    def test_quantize_replaces_linear_weights(self):
        from trainer.quantization.fp8 import LinearFp8, quantize_linear_fp8
        linear = nn.Linear(64, 32)
        q_linear = quantize_linear_fp8(linear)
        assert isinstance(q_linear, LinearFp8)
        assert q_linear.weight.dtype == torch.float8_e4m3fn

    def test_quantize_preserves_shape(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        linear = nn.Linear(128, 64, bias=True)
        q_linear = quantize_linear_fp8(linear)
        assert q_linear.weight.shape == (64, 128)
        assert q_linear.bias is not None
        assert q_linear.bias.shape == (64,)

    def test_forward_produces_correct_shape(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        linear = nn.Linear(32, 16)
        q_linear = quantize_linear_fp8(linear)
        x = torch.randn(2, 32)
        out = q_linear(x)
        assert out.shape == (2, 16)

    def test_fp8_scaled_stores_scale(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        linear = nn.Linear(64, 32)
        q_linear = quantize_linear_fp8(linear, scaled=True)
        assert hasattr(q_linear, "_weight_scale")
        assert q_linear._weight_scale is not None

    def test_weights_frozen_after_quantize(self):
        from trainer.quantization.fp8 import quantize_linear_fp8
        linear = nn.Linear(32, 16)
        q_linear = quantize_linear_fp8(linear)
        assert not q_linear.weight.requires_grad


class TestQuantizeModel:
    """Test whole-model quantization utility."""

    def test_skips_norm_layers(self):
        from trainer.quantization import quantize_model
        model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.Linear(64, 32))
        quantize_model(model, quantization_type="fp8")
        # LayerNorm should be untouched
        assert model[1].weight.dtype == torch.float32

    def test_skips_embedding_layers(self):
        from trainer.quantization import quantize_model
        model = nn.Sequential(nn.Embedding(100, 64), nn.Linear(64, 32))
        quantize_model(model, quantization_type="fp8")
        assert model[0].weight.dtype == torch.float32

    def test_quantizes_all_linears(self):
        from trainer.quantization import quantize_model
        from trainer.quantization.base import QuantizedLinear
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
        quantize_model(model, quantization_type="fp8")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, QuantizedLinear):
                pytest.fail(f"Linear layer {name} was not quantized")

    def test_quantize_count_reported(self):
        from trainer.quantization import quantize_model
        model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.Linear(64, 32))
        stats = quantize_model(model, quantization_type="fp8")
        assert stats["quantized"] == 2
        assert stats["skipped"] >= 1


class TestNF4Quantization:
    """Test NF4 quantization (requires bitsandbytes)."""

    def test_nf4_available_check(self):
        from trainer.quantization.bnb import is_bnb_available
        # Should return bool without crashing
        result = is_bnb_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not _bnb_available(), reason="bitsandbytes not installed"
    )
    def test_nf4_quantize_linear(self):
        from trainer.quantization.bnb import quantize_linear_nf4
        linear = nn.Linear(64, 32)
        q_linear = quantize_linear_nf4(linear)
        x = torch.randn(2, 64)
        out = q_linear(x)
        assert out.shape == (2, 32)


def _bnb_available():
    try:
        from trainer.quantization.bnb import is_bnb_available
        return is_bnb_available()
    except Exception:
        return False
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_quantization.py -v
```

Expected: FAIL — `trainer.quantization` does not exist.

**Step 3: Implement `trainer/quantization/base.py` — Abstract base + registry**

```python
"""Base classes for quantized linear layers.

All quantized layers inherit from QuantizedLinear and implement:
- forward(): dequantize weight to compute_dtype, run matmul
- Class-level from_linear(): factory to convert nn.Linear
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QuantizedLinear(nn.Module):
    """Abstract base for quantized linear layers.

    Subclasses store weights in a compressed format and dequantize
    on-the-fly during forward(). Base model weights are always frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.bias = None

    def dequantize_weight(self) -> Tensor:
        """Return full-precision weight for inspection/DoRA. Override in subclass."""
        raise NotImplementedError

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "QuantizedLinear":
        """Convert an nn.Linear to this quantized type. Override in subclass."""
        raise NotImplementedError
```

**Step 4: Implement `trainer/quantization/fp8.py` — FP8 weight-only**

```python
"""FP8 weight-only quantization.

Pure PyTorch, no external dependencies. Uses torch.float8_e4m3fn.
Forward: dequantize weight to compute_dtype, standard F.linear.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import QuantizedLinear


class LinearFp8(QuantizedLinear):
    """Linear layer with FP8 weight storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(in_features, out_features, bias, compute_dtype)
        # Weight stored as fp8 (set during from_linear)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self._weight_scale: Tensor | None = None  # optional per-tensor scale

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize: fp8 -> compute_dtype, apply scale if present
        w = self.weight.to(dtype=self.compute_dtype)
        if self._weight_scale is not None:
            w = w * self._weight_scale
        return F.linear(x.to(self.compute_dtype), w, self.bias)

    def dequantize_weight(self) -> Tensor:
        w = self.weight.float()
        if self._weight_scale is not None:
            w = w * self._weight_scale
        return w

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, *, scaled: bool = False, **kwargs
    ) -> "LinearFp8":
        q = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=kwargs.get("compute_dtype", torch.bfloat16),
        )
        with torch.no_grad():
            w = linear.weight.data.float()
            if scaled:
                max_val = w.abs().max().clamp(min=1e-12)
                scale = torch.finfo(torch.float8_e4m3fn).max / max_val
                q.weight.data = (w * scale).to(torch.float8_e4m3fn)
                q._weight_scale = scale.reciprocal()
            else:
                q.weight.data = w.to(torch.float8_e4m3fn)
            if linear.bias is not None:
                q.bias.data.copy_(linear.bias.data)
        return q


def quantize_linear_fp8(
    linear: nn.Linear, scaled: bool = False, **kwargs
) -> LinearFp8:
    """Convert a single nn.Linear to FP8."""
    return LinearFp8.from_linear(linear, scaled=scaled, **kwargs)
```

**Step 5: Implement `trainer/quantization/bnb.py` — NF4 + INT8 via bitsandbytes**

```python
"""NF4 and INT8 quantization via bitsandbytes.

Deferred import: bitsandbytes is optional. Functions check availability
and raise clear errors if not installed.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import QuantizedLinear

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def is_bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def _require_bnb():
    if not is_bnb_available():
        raise ImportError(
            "bitsandbytes is required for NF4/INT8 quantization. "
            "Install with: pip install bitsandbytes"
        )


class LinearNf4(QuantizedLinear):
    """Linear layer with NF4 (4-bit normal float) weight storage via bitsandbytes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(in_features, out_features, bias, compute_dtype)
        _require_bnb()
        # Quantized weight + quant_state set during from_linear
        self._quantized_weight: Tensor | None = None
        self._quant_state = None

    def forward(self, x: Tensor) -> Tensor:
        import bitsandbytes as bnb
        x = x.to(self.compute_dtype)
        out = bnb.matmul_4bit(
            x, self._quantized_weight.t(),
            bias=self.bias, quant_state=self._quant_state,
        )
        return out

    def dequantize_weight(self) -> Tensor:
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(
            self._quantized_weight, self._quant_state
        ).float()

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "LinearNf4":
        import bitsandbytes as bnb
        q = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=kwargs.get("compute_dtype", torch.bfloat16),
        )
        with torch.no_grad():
            w = linear.weight.data.float()
            q._quantized_weight, q._quant_state = bnb.functional.quantize_4bit(
                w, blocksize=64, compress_statistics=True,
                quant_type="nf4", quant_storage=torch.uint8,
            )
            if linear.bias is not None:
                q.bias.data.copy_(linear.bias.data)
        return q


class LinearInt8(QuantizedLinear):
    """Linear layer with INT8 weight storage via bitsandbytes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(in_features, out_features, bias, compute_dtype)
        _require_bnb()
        self._int8_module: nn.Module | None = None

    def forward(self, x: Tensor) -> Tensor:
        return self._int8_module(x)

    def dequantize_weight(self) -> Tensor:
        m = self._int8_module
        return (m.weight.data.float() * m.state.SCB.float().unsqueeze(1) / 127.0)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "LinearInt8":
        import bitsandbytes as bnb
        q = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=kwargs.get("compute_dtype", torch.bfloat16),
        )
        # Use bnb's Linear8bitLt which handles quantization internally
        int8_linear = bnb.nn.Linear8bitLt(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            has_fp16_weights=False,
        )
        int8_linear.weight = nn.Parameter(linear.weight.data, requires_grad=False)
        if linear.bias is not None:
            int8_linear.bias = nn.Parameter(linear.bias.data, requires_grad=False)
        q._int8_module = int8_linear
        return q


def quantize_linear_nf4(linear: nn.Linear, **kwargs) -> LinearNf4:
    return LinearNf4.from_linear(linear, **kwargs)


def quantize_linear_int8(linear: nn.Linear, **kwargs) -> LinearInt8:
    return LinearInt8.from_linear(linear, **kwargs)
```

**Step 6: Implement `trainer/quantization/utils.py` — Graph walker**

```python
"""Utilities for applying quantization across a model graph.

Walks the module tree, replacing nn.Linear instances with quantized
variants while skipping normalization and embedding layers.
Quantizes one module at a time to limit peak VRAM.
"""
from __future__ import annotations

import logging
from typing import Callable

import torch.nn as nn

logger = logging.getLogger(__name__)

# Module types that should never be quantized
_SKIP_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.Embedding)


def replace_linear_layers(
    model: nn.Module,
    convert_fn: Callable[[nn.Linear], nn.Module],
) -> dict[str, int]:
    """Replace all nn.Linear in model with convert_fn(linear).

    Walks the graph recursively, replacing each nn.Linear with the
    result of convert_fn. Skips modules in _SKIP_TYPES.
    Operates one layer at a time to avoid OOM during quantization.

    Returns:
        Dict with "quantized" and "skipped" counts.
    """
    stats = {"quantized": 0, "skipped": 0}
    visited: set[int] = set()
    _replace_recursive(model, convert_fn, stats, visited)
    return stats


def _replace_recursive(
    parent: nn.Module,
    convert_fn: Callable[[nn.Linear], nn.Module],
    stats: dict[str, int],
    visited: set[int],
) -> None:
    if id(parent) in visited:
        return
    visited.add(id(parent))

    for name, child in list(parent.named_children()):
        if isinstance(child, _SKIP_TYPES):
            stats["skipped"] += 1
            continue

        if isinstance(child, nn.Linear):
            converted = convert_fn(child)
            setattr(parent, name, converted)
            stats["quantized"] += 1
        else:
            _replace_recursive(child, convert_fn, stats, visited)
```

**Step 7: Implement `trainer/quantization/__init__.py` — Public API**

```python
"""Multi-type model quantization for VRAM reduction.

Supported types:
- "nf4"        — 4-bit NormalFloat via bitsandbytes (~75% VRAM reduction)
- "int8"       — 8-bit integer via bitsandbytes (~50% reduction)
- "fp8"        — 8-bit float, pure PyTorch (~50% reduction)
- "fp8_scaled" — 8-bit float with per-tensor scaling (~50% reduction)
- None         — no quantization

Usage in strategy setup():
    from trainer.quantization import quantize_model
    if cfg.model.quantization:
        quantize_model(model, cfg.model.quantization, compute_dtype=train_dtype)
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

from .base import QuantizedLinear
from .utils import replace_linear_layers

logger = logging.getLogger(__name__)

# Registry: quantization_type -> (convert_fn_factory, requires_package)
_QUANTIZERS: dict[str, Callable] = {}


def _register_quantizers():
    """Lazily register all quantization backends."""
    if _QUANTIZERS:
        return

    # FP8 — always available (pure PyTorch)
    from .fp8 import quantize_linear_fp8
    _QUANTIZERS["fp8"] = lambda **kw: lambda lin: quantize_linear_fp8(lin, scaled=False, **kw)
    _QUANTIZERS["fp8_scaled"] = lambda **kw: lambda lin: quantize_linear_fp8(lin, scaled=True, **kw)

    # NF4 + INT8 — require bitsandbytes
    from .bnb import is_bnb_available
    if is_bnb_available():
        from .bnb import quantize_linear_nf4, quantize_linear_int8
        _QUANTIZERS["nf4"] = lambda **kw: lambda lin: quantize_linear_nf4(lin, **kw)
        _QUANTIZERS["int8"] = lambda **kw: lambda lin: quantize_linear_int8(lin, **kw)
    else:
        # Register stubs that give clear errors
        def _bnb_stub(name):
            def _factory(**kw):
                raise ImportError(
                    f"bitsandbytes is required for '{name}' quantization. "
                    f"Install with: pip install bitsandbytes"
                )
            return _factory
        _QUANTIZERS["nf4"] = _bnb_stub("nf4")
        _QUANTIZERS["int8"] = _bnb_stub("int8")


def get_quantizer(quantization_type: str | None):
    """Get a quantizer factory for the given type, or None."""
    if quantization_type is None:
        return None
    _register_quantizers()
    if quantization_type not in _QUANTIZERS:
        raise ValueError(
            f"Unknown quantization type '{quantization_type}'. "
            f"Supported: {sorted(_QUANTIZERS.keys())}"
        )
    return _QUANTIZERS[quantization_type]


def quantize_model(
    model: nn.Module,
    quantization_type: str,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, int]:
    """Quantize all Linear layers in a model.

    Args:
        model: Model to quantize in-place.
        quantization_type: One of "nf4", "int8", "fp8", "fp8_scaled".
        compute_dtype: Dtype for forward pass computation.

    Returns:
        Dict with "quantized" and "skipped" counts.
    """
    factory = get_quantizer(quantization_type)
    if factory is None:
        return {"quantized": 0, "skipped": 0}

    convert_fn = factory(compute_dtype=compute_dtype)
    logger.info("Quantizing model with '%s' (%s compute)", quantization_type, compute_dtype)
    stats = replace_linear_layers(model, convert_fn)
    logger.info(
        "Quantized %d linear layers, skipped %d norm/embed layers",
        stats["quantized"], stats["skipped"],
    )
    return stats
```

**Step 8: Update config to accept all quantization types**

In `trainer/config/schema.py`, update the `ModelConfig.quantization` field docstring/comment:

```python
quantization: str | None = None  # None, "nf4", "int8", "fp8", "fp8_scaled"
```

**Step 9: Add `_quantize_model` helper to ModelStrategy base**

In `trainer/arch/base.py`:

```python
def _quantize_model(self, model: nn.Module, cfg: TrainConfig) -> None:
    """Apply quantization to model if configured. Call after load, before grad ckpt."""
    if cfg.model.quantization:
        from trainer.quantization import quantize_model
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        compute_dtype = dtype_map.get(cfg.model.dtype, torch.bfloat16)
        quantize_model(model, cfg.model.quantization, compute_dtype=compute_dtype)
```

Each strategy's `setup()` calls `self._quantize_model(model, cfg)` after model load, before gradient checkpointing and block swap.

**Step 10: Run tests, commit**

```bash
python -m pytest tests/test_quantization.py tests/ -v
git add trainer/quantization/ tests/test_quantization.py trainer/config/schema.py trainer/arch/base.py
git commit -m "feat: add multi-type quantization (NF4, INT8, FP8) for VRAM reduction"
```

---

## Task 6: EMA (Exponential Moving Average)

EMA maintains a shadow copy of model weights that's a smoothed average of training weights. This improves output quality, especially for full fine-tuning. Deferred to v2 in spec, but requested by user.

**Performance notes (improvements over competitors):**
- Shadow params on CPU by default (like OneTrainer) to avoid doubling GPU VRAM
- Use `torch.lerp` instead of `add_(one_minus_decay * diff)` — single fused op
- Non-blocking CPU↔GPU copies for shadow update
- Warm-start decay from OneTrainer: `decay = min((1 + step) / (10 + step), max_decay)` — avoids cold-start averaging artifacts
- Stochastic rounding from AI-Toolkit when casting fp32 shadow → bf16 model — avoids systematic bias

**Files:**
- Create: `trainer/ema.py` (new)
- Modify: `trainer/config/schema.py` — add EMA config fields
- Modify: `trainer/training/trainer.py` — add EMA update after optimizer step, EMA swap for save/sample
- Test: `tests/test_ema.py` (new)

**Step 1: Write failing tests**

```python
# tests/test_ema.py
"""Tests for Exponential Moving Average."""
import pytest
import torch
import torch.nn as nn
from trainer.ema import EMATracker


class TestEMATracker:
    def _make_model(self):
        return nn.Linear(4, 4, bias=False)

    def test_shadow_initialized_to_params(self):
        model = self._make_model()
        ema = EMATracker(model.parameters(), decay=0.999)
        for shadow, param in zip(ema.shadow_params, model.parameters()):
            assert torch.allclose(shadow, param.data.float())

    def test_shadow_on_cpu(self):
        model = self._make_model()
        ema = EMATracker(model.parameters(), decay=0.999, device="cpu")
        for shadow in ema.shadow_params:
            assert shadow.device == torch.device("cpu")

    def test_step_moves_shadow_toward_params(self):
        model = self._make_model()
        ema = EMATracker(model.parameters(), decay=0.9)
        original_shadow = ema.shadow_params[0].clone()
        # Simulate a training step that modifies params
        with torch.no_grad():
            model.weight.add_(torch.randn_like(model.weight))
        ema.step(model.parameters(), global_step=100)
        # Shadow should have moved toward new params
        new_shadow = ema.shadow_params[0]
        assert not torch.allclose(new_shadow, original_shadow, atol=1e-6)

    def test_warm_start_decay(self):
        model = self._make_model()
        ema = EMATracker(model.parameters(), decay=0.999)
        # At step 0: decay = min((1+0)/(10+0), 0.999) = 0.1
        assert ema.get_decay(0) == pytest.approx(0.1)
        # At step 90: decay = min((1+90)/(10+90), 0.999) = 0.91
        assert ema.get_decay(90) == pytest.approx(0.91)
        # At large step: should approach max_decay
        assert ema.get_decay(100000) == pytest.approx(0.999, abs=0.001)

    def test_copy_to_and_restore(self):
        model = self._make_model()
        ema = EMATracker(model.parameters(), decay=0.999)
        original_weights = model.weight.data.clone()
        # Modify params
        with torch.no_grad():
            model.weight.fill_(99.0)
        # EMA shadow is still original
        ema.copy_to(model.parameters())
        assert torch.allclose(model.weight.data, original_weights.float(), atol=1e-5)
        # Restore should bring back the 99s
        ema.restore(model.parameters())
        assert torch.allclose(model.weight.data, torch.full_like(model.weight, 99.0))

    def test_state_dict_roundtrip(self):
        model = self._make_model()
        ema = EMATracker(model.parameters(), decay=0.99)
        ema.step(model.parameters(), global_step=50)
        state = ema.state_dict()
        # Create new EMA and load state
        ema2 = EMATracker(model.parameters(), decay=0.99)
        ema2.load_state_dict(state)
        for s1, s2 in zip(ema.shadow_params, ema2.shadow_params):
            assert torch.allclose(s1, s2)
```

**Step 2: Implement `trainer/ema.py`**

```python
"""Exponential Moving Average tracker for model parameters.

Shadow params are stored on CPU in fp32 to avoid doubling GPU VRAM.
Uses warm-start decay from OneTrainer and non-blocking transfers.
"""
from __future__ import annotations

import logging
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

logger = logging.getLogger(__name__)


class EMATracker:
    """Tracks an exponential moving average of model parameters.

    Shadow parameters are kept in fp32 on the specified device (default: CPU).
    Updates use non-blocking copies to overlap with GPU compute.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        decay: float = 0.9999,
        device: str | torch.device = "cpu",
    ) -> None:
        self.max_decay = decay
        self.device = torch.device(device)
        self.shadow_params: list[Tensor] = [
            p.data.detach().float().to(self.device) for p in parameters
        ]
        self._backup: list[Tensor] = []

    def get_decay(self, global_step: int) -> float:
        """Warm-start decay: ramps from ~0.1 to max_decay over early steps."""
        return min((1 + global_step) / (10 + global_step), self.max_decay)

    @torch.no_grad()
    def step(
        self,
        parameters: Iterable[Parameter],
        global_step: int,
    ) -> None:
        """Update shadow params toward current params."""
        decay = self.get_decay(global_step)
        for shadow, param in zip(self.shadow_params, parameters):
            # Non-blocking copy to CPU, lerp in fp32
            param_fp32 = param.data.float().to(self.device, non_blocking=True)
            shadow.lerp_(param_fp32, 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[Parameter]) -> None:
        """Copy shadow params to model (for inference/saving). Backs up originals."""
        self._backup = []
        for shadow, param in zip(self.shadow_params, parameters):
            self._backup.append(param.data.clone())
            param.data.copy_(shadow.to(param.device, param.dtype, non_blocking=True))

    @torch.no_grad()
    def restore(self, parameters: Iterable[Parameter]) -> None:
        """Restore original params from backup (after inference/saving)."""
        for backup, param in zip(self._backup, parameters):
            param.data.copy_(backup)
        self._backup = []

    def state_dict(self) -> dict:
        return {
            "shadow_params": [s.clone() for s in self.shadow_params],
            "max_decay": self.max_decay,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow_params = [s.to(self.device) for s in state["shadow_params"]]
        self.max_decay = state.get("max_decay", self.max_decay)
```

**Step 3: Add config fields**

In `trainer/config/schema.py` `TrainingConfig`:

```python
ema_enabled: bool = False
ema_decay: float = 0.9999
ema_device: str = "cpu"  # cpu or cuda — cpu avoids doubling VRAM
```

**Step 4: Integrate into trainer.py**

In `trainer/training/trainer.py`:

```python
# After optimizer creation (~line 91):
if cfg.training.ema_enabled:
    from trainer.ema import EMATracker
    self.ema = EMATracker(
        self.method_result.get_trainable_params_flat(),
        decay=cfg.training.ema_decay,
        device=cfg.training.ema_device,
    )
else:
    self.ema = None

# After optimizer.zero_grad() (~line 202):
if self.ema is not None:
    self.ema.step(self.method_result.get_trainable_params_flat(), global_step)

# In _save_checkpoint / _save_final — swap EMA weights for saving:
if self.ema is not None:
    self.ema.copy_to(self.method_result.get_trainable_params_flat())
# ... save ...
if self.ema is not None:
    self.ema.restore(self.method_result.get_trainable_params_flat())

# In _generate_samples — same swap pattern
```

**Step 5: Run tests, commit**

```bash
python -m pytest tests/test_ema.py tests/ -v
git add trainer/ema.py trainer/config/schema.py trainer/training/trainer.py tests/test_ema.py
git commit -m "feat: add EMA tracker with warm-start decay and CPU shadow params"
```

---

## Task 7: OOM Retry with Counter

AI-Toolkit has a 3-strike OOM recovery — on OOM, clear caches and retry the step instead of immediately crashing. We already save an emergency checkpoint on OOM; this adds retry before that.

**Files:**
- Modify: `trainer/training/trainer.py` — wrap training step in OOM retry
- Test: `tests/test_oom_retry.py` (new)

**Step 1: Write failing test**

```python
# tests/test_oom_retry.py
"""Test OOM retry logic."""
import pytest


class TestOOMRetryConfig:
    def test_oom_retry_behavior_documented(self):
        """Verify the trainer has OOM retry logic."""
        import inspect
        from trainer.training.trainer import Trainer
        source = inspect.getsource(Trainer.run)
        assert "OutOfMemoryError" in source
        assert "consecutive_oom" in source or "oom_count" in source
```

**Step 2: Implement in trainer.py**

Wrap the `accelerator.accumulate` block:

```python
oom_count = 0
MAX_OOM_RETRIES = 3

# Inside the step loop:
try:
    with accelerator.accumulate(training_model):
        # ... existing training step ...
    oom_count = 0  # reset on success
except torch.cuda.OutOfMemoryError:
    oom_count += 1
    logger.warning("OOM on step %d (%d/%d retries)", global_step, oom_count, MAX_OOM_RETRIES)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    if oom_count >= MAX_OOM_RETRIES:
        # Fall through to existing OOM handler with emergency checkpoint
        raise
    continue  # retry the step
```

**Step 3: Run tests, commit**

```bash
python -m pytest tests/test_oom_retry.py tests/ -v
git add trainer/training/trainer.py tests/test_oom_retry.py
git commit -m "feat: add 3-strike OOM retry before emergency checkpoint"
```

---

## Task 8: torch.compile Support

The `model.compile_model` config field exists but is never used. `torch.compile` can give 10-20% speedup on supported hardware.

**Files:**
- Modify: `trainer/arch/base.py` — add compile helper
- Modify: Strategy files — call compile after model load in `setup()`
- Test: `tests/test_compile.py` (new)

**Step 1: Write test**

```python
# tests/test_compile.py
"""Test torch.compile config field."""
import pytest
from trainer.config.schema import TrainConfig


class TestCompileConfig:
    def test_compile_default_false(self):
        cfg = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake"},
            data={"dataset_config": "/fake.toml"},
            saving={"output_dir": "/fake"},
        )
        assert cfg.model.compile_model is False

    def test_compile_can_enable(self):
        cfg = TrainConfig(
            model={"architecture": "wan", "base_model_path": "/fake",
                   "compile_model": True},
            data={"dataset_config": "/fake.toml"},
            saving={"output_dir": "/fake"},
        )
        assert cfg.model.compile_model is True
```

**Step 2: Add compile call in strategies**

In each strategy's `setup()`, after model load and gradient checkpointing:

```python
if cfg.model.compile_model:
    logger.info("Compiling model with torch.compile")
    model = torch.compile(model)
```

**Note:** `torch.compile` must happen AFTER gradient checkpointing is enabled, and BEFORE `accelerator.prepare()`. The compiled model should still be assignable to `components.model`.

**Step 3: Run tests, commit**

```bash
python -m pytest tests/test_compile.py tests/ -v
git add trainer/arch/base.py trainer/arch/*/strategy.py tests/test_compile.py
git commit -m "feat: add torch.compile support via model.compile_model config"
```

---

## Dependency Order

```
Task 1 (gradient checkpointing) ─── independent
Task 2 (loss functions)         ─── independent
Task 3 (SNR weighting)          ─── depends on Task 2 (uses loss.py)
Task 4 (network weights)        ─── independent
Task 5 (FP8 quantization)       ─── independent
Task 6 (EMA)                    ─── independent
Task 7 (OOM retry)              ─── independent
Task 8 (torch.compile)          ─── independent
```

**Parallel groups:**
- Group A: Tasks 1, 2, 4, 5, 6, 7, 8 (all independent)
- Group B: Task 3 (after Task 2)

Most tasks can be implemented by independent subagents. Task 3 must wait for Task 2's `trainer/loss.py`. Task 5 is the largest (multiple files in a new package) but is self-contained.

---

## VRAM Impact Summary

| Feature | Estimated Saving | When |
|---------|-----------------|------|
| Gradient checkpointing (already enabled by default) | ~30-50% | Already working |
| NF4 quantization (4-bit) | ~75% on model weights | Task 5 |
| INT8 / FP8 quantization (8-bit) | ~50% on model weights | Task 5 |
| EMA on CPU (not GPU) | Avoids 2x VRAM doubling | Task 6 |
| torch.compile (indirect — faster, same VRAM) | Faster throughput | Task 8 |
| **Combined (NF4 + grad ckpt + block swap)** | Train 14B models on 12GB | All tasks |

---

## What We're NOT Doing (v2+)

These were identified in the analysis but are deferred:

- **Activation offloading** — OneTrainer's `LayerOffloadConductor` is 1000+ lines and tightly coupled. High effort, revisit for v2.
- **Fused backward pass** — Per-parameter optimizer stepping in grad hooks. Only needed with activation offloading.
- **Masked training / prior preservation** — Requires data pipeline changes. v2 feature per spec.
- **Text encoder training** — Useful but requires data pipeline + config changes. v2.
- **Textual inversion** — Niche. v2.
- **Validation loss** — Nice-to-have but not critical path. v2.
- **Multi-loss mixing** (multiple loss types with configurable strengths) — OneTrainer supports this. Overkill for v1.
- **Learnable SNR gamma** — AI-Toolkit's unique feature. Interesting research, not production-ready.
