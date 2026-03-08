"""Core interface for model architectures. One subclass per architecture."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trainer.config.schema import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainStepOutput:
    """Structured output from a training step."""
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelComponents:
    """Bundle of loaded model parts.

    Mutable -- Trainer updates .model after accelerator.prepare().
    The 'extra' dict holds arch-specific objects by convention:
        Wan:          {"wan_config": ..., "task": "t2v-14B"}
        HunyuanVideo: {"rope_cos": Tensor, "rope_sin": Tensor}
        Flux:         {"img_ids_template": Tensor}
    """
    model: nn.Module
    vae: nn.Module | None = None
    text_encoders: list[nn.Module] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class ModelStrategy:
    """How a model architecture participates in training.

    Concrete base class. One subclass per architecture. The Trainer
    interacts with models exclusively through this interface.

    Required (must override): architecture, setup, training_step
    Optional: supports_video, encode_text, prepare_latents, generate_sample
    Lifecycle hooks: on_before/after_accelerate_prepare, on_before_training_step,
                     on_before/after_sampling
    """

    def __init__(self, config: TrainConfig):
        self.config = config

    # --- Required ---

    @property
    def architecture(self) -> str:
        """Unique identifier, e.g. 'wan', 'hunyuan_video', 'flux_2'."""
        raise NotImplementedError

    def setup(self) -> ModelComponents:
        """Load all model components (denoiser, VAE, text encoders).

        Uses self.config for paths, dtype, etc.
        Raises ModelLoadError on failure.
        """
        raise NotImplementedError

    def training_step(
        self,
        components: ModelComponents,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> TrainStepOutput:
        """Full forward pass: batch -> loss. Strategy owns everything inside:
        noise sampling, timestep selection, conditioning, denoiser call, loss.
        Trainer handles backward pass."""
        raise NotImplementedError

    # --- Optional ---

    @property
    def supports_video(self) -> bool:
        return False

    def encode_text(
        self, text_encoders: list[nn.Module], prompts: list[str], device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Encode prompts into conditioning tensors. For text encoder caching."""
        raise NotImplementedError(f"encode_text not implemented for {self.architecture}")

    def prepare_latents(self, vae: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixels to latent space. For latent caching."""
        raise NotImplementedError(f"prepare_latents not implemented for {self.architecture}")

    def generate_sample(self, components: ModelComponents, prompt: str, **kwargs) -> Any:
        """Generate a sample for preview. Returns PIL Image, video path, or tensor."""
        raise NotImplementedError(f"Sampling not implemented for {self.architecture}")

    # --- Lifecycle hooks (no-op defaults) ---

    def on_before_accelerate_prepare(
        self, components: ModelComponents, accelerator: Any,
    ) -> dict[str, Any]:
        """Return {"device_placement": bool}. Override for block swap setup."""
        return {"device_placement": True}

    def on_after_accelerate_prepare(
        self, components: ModelComponents, accelerator: Any,
    ) -> None:
        """For block swap move, torch.compile, etc."""
        pass

    def on_before_training_step(self, components: ModelComponents) -> None:
        """For block swap: prepare_block_swap_before_forward."""
        pass

    def on_before_sampling(self, components: ModelComponents) -> None:
        """Switch to inference mode."""
        pass

    def on_after_sampling(self, components: ModelComponents) -> None:
        """Switch back to training mode."""
        pass

    # --- Text encoder training ---

    _train_text_encoder: bool = False
    _text_encoder_lr: float | None = None
    _te_gradient_checkpointing: bool = True

    def encode_text_for_training(
        self, components: ModelComponents, captions: list[str], device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Encode raw captions into conditioning tensors during training.

        Override in strategies that support text encoder training.
        Falls back to raising NotImplementedError.

        Args:
            components: Model components (includes text_encoders).
            captions: List of raw caption strings.
            device: Target device for outputs.

        Returns:
            Dict of conditioning tensors (same keys as TE cache).
        """
        raise NotImplementedError(
            f"Text encoder training not implemented for {self.architecture}. "
            f"Override encode_text_for_training() in the strategy."
        )

    def _setup_text_encoder_training(self, cfg: TrainConfig) -> None:
        """Cache TE training config values. Call from setup()."""
        self._train_text_encoder = cfg.training.train_text_encoder
        self._text_encoder_lr = cfg.training.text_encoder_lr
        self._te_gradient_checkpointing = cfg.training.text_encoder_gradient_checkpointing

    # --- Shared utilities (call from setup/training_step) ---

    _loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None
    _unreduced_loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None
    _weight_fn: Callable[[Tensor], Tensor] | None = None
    # Default noise offset type - overridden by setup() when cfg is available.
    _noise_offset_type: str = "simple"

    def _setup_loss_fn(self, cfg: TrainConfig) -> None:
        """Cache the loss function callables. Call from setup()."""
        from trainer.loss import get_loss_fn, get_unreduced_loss_fn
        self._loss_fn = get_loss_fn(
            cfg.training.loss_type,
            delta=cfg.training.huber_delta,
        )
        self._unreduced_loss_fn = get_unreduced_loss_fn(
            cfg.training.loss_type,
            delta=cfg.training.huber_delta,
        )

    def _setup_loss_weighting(self, cfg: TrainConfig) -> None:
        """Cache the loss weighting function. Call from setup()."""
        from trainer.loss_weighting import get_weight_fn
        self._weight_fn = get_weight_fn(
            cfg.training.weighting_scheme,
            snr_gamma=cfg.training.snr_gamma,
            p2_gamma=cfg.training.p2_gamma,
        )

    def _compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        loss_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute training loss using the configured loss function.

        Args:
            pred:        Model predictions, shape (B, ...).
            target:      Ground-truth targets, same shape as pred.
            loss_weight: Optional per-sample dataset weights, shape (B,).
                         Applied as a scalar multiplier via loss_weight.mean().
                         None (default) is equivalent to all-ones weights.
        """
        if self._loss_fn is None:
            loss = F.mse_loss(pred, target, reduction="mean")
        else:
            loss = self._loss_fn(pred, target)
        if loss_weight is not None:
            loss = loss * loss_weight.mean()
        return loss

    def _compute_weighted_loss(
        self,
        pred: Tensor,
        target: Tensor,
        timesteps: Tensor,
        snr: Tensor | None = None,
        loss_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute loss with optional SNR-based weighting.

        If no weighting scheme is configured, falls back to _compute_loss.

        Args:
            pred:        Model predictions, shape (B, ...).
            target:      Ground-truth targets, same shape as pred.
            timesteps:   Sampled timesteps for SNR computation, shape (B,).
            snr:         Pre-computed SNR values (optional; derived from timesteps if None).
            loss_weight: Optional per-sample dataset weights, shape (B,).
                         Applied after SNR weighting as a scalar multiplier.
                         None (default) is equivalent to all-ones weights.
        """
        if self._weight_fn is None:
            return self._compute_loss(pred, target, loss_weight=loss_weight)

        from trainer.loss_weighting import compute_snr_flow_matching

        # Per-element loss then weight per-sample
        if self._unreduced_loss_fn is not None:
            loss_per_element = self._unreduced_loss_fn(pred, target)
        else:
            loss_per_element = F.mse_loss(pred, target, reduction="none")

        if snr is None:
            snr = compute_snr_flow_matching(timesteps)
        weights = self._weight_fn(snr)
        # Broadcast weights: (B,) -> (B, 1, 1, ...) to match loss shape
        for _ in range(loss_per_element.dim() - 1):
            weights = weights.unsqueeze(-1)
        loss = (loss_per_element * weights).mean()
        if loss_weight is not None:
            loss = loss * loss_weight.mean()
        return loss

    def _compute_masked_loss(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        mask_weight: float = 1.0,
        normalize_by_area: bool = True,
    ) -> Tensor:
        """Compute loss with spatial masking.

        Masked regions (mask=1) get full weight scaled by mask_weight.
        Unmasked regions (mask=0) get weight=1 (they still contribute to training).

        Args:
            pred:              Model predictions, shape (B, C, ...).
            target:            Ground truth, same shape as pred.
            mask:              Binary/soft mask, shape (B, 1, ...) or broadcastable.
                               Values in [0, 1]. 1 = masked region (focus training here).
            mask_weight:       Scaling factor applied to masked regions. Values >1
                               increase emphasis on masked areas; values in (0,1] reduce it.
            normalize_by_area: If True, normalize loss by effective mask area so the
                               loss magnitude does not change with mask size.

        Returns:
            Scalar loss tensor.
        """
        if self._unreduced_loss_fn is not None:
            loss_per_element = self._unreduced_loss_fn(pred, target)
        else:
            loss_per_element = F.mse_loss(pred, target, reduction="none")

        # Weight: masked regions get mask_weight, unmasked regions get 1.0
        weight = mask * mask_weight + (1.0 - mask)
        weighted = loss_per_element * weight

        if normalize_by_area:
            # Normalize by effective weight sum to keep loss scale stable regardless
            # of how much of the image is masked.  +1e-8 prevents division by zero.
            # Use numel - mask.sum() instead of (1.0 - mask).sum() to avoid
            # allocating a full-size intermediate tensor.
            mask_sum = mask.sum()
            denom = mask_sum * mask_weight + (mask.numel() - mask_sum) + 1e-8
            return weighted.sum() / denom
        return weighted.mean()

    def _quantize_model(self, model: nn.Module, cfg: TrainConfig) -> None:
        """Apply quantization to model if configured. Call after load, before grad ckpt."""
        if cfg.model.quantization:
            from trainer.quantization import quantize_model
            dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
            compute_dtype = dtype_map.get(cfg.model.dtype, torch.bfloat16)
            quantize_model(model, cfg.model.quantization, compute_dtype=compute_dtype)

    def _maybe_compile_model(self, model: nn.Module, cfg: TrainConfig) -> nn.Module:
        """Apply torch.compile if configured. Call from setup() after grad ckpt."""
        if cfg.model.compile_model:
            logger.info("Compiling model with torch.compile")
            model = torch.compile(model)
        return model

    def _setup_weight_bouncing(
        self, model: nn.Module, cfg: TrainConfig, device: torch.device
    ) -> None:
        """Apply weight bouncing if configured. Call from setup() after model load.

        Weight bouncing keeps nn.Linear weights on CPU pinned memory and transfers
        them to ``device`` only during each layer's forward/backward pass, reducing
        GPU VRAM to approximately one layer's worth of weights at a time.

        Args:
            model:  The model whose Linear layers will be converted in-place.
            cfg:    Training config; checks ``cfg.model.weight_bouncing``.
            device: GPU device to bounce weights to during forward/backward.
        """
        if cfg.model.weight_bouncing:
            from trainer.util.weight_bouncing import apply_weight_bouncing
            count = apply_weight_bouncing(model, device)
            logger.info("Weight bouncing enabled: %d layers converted", count)

    # --- Timestep & noise utilities (shared across flow-matching strategies) ---

    @staticmethod
    def _sample_t(
        batch_size: int,
        device: torch.device,
        method: str = "uniform",
        min_t: float = 0.0,
        max_t: float = 1.0,
        sigmoid_scale: float = 1.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 1.0,
    ) -> Tensor:
        """Sample interpolation coefficient t in [min_t, max_t].

        Returns a float tensor of shape (batch_size,). Strategies scale to
        their model's expected timestep range (e.g. t*1000+1 for Wan,
        t*1000 for SD3, raw t for Flux).

        Args:
            batch_size:    Number of samples.
            device:        Target device.
            method:        "uniform", "sigmoid", "logit_normal", or "shift".
            min_t:         Lower bound for t.
            max_t:         Upper bound for t.
            sigmoid_scale: Scale factor for sigmoid/shift methods.
            logit_mean:    Mean for logit_normal method.
            logit_std:     Std for logit_normal method.
            flow_shift:    Pre-computed exp(discrete_flow_shift). Only used by
                           the "shift" method; ignored by other methods.
        """
        if method == "uniform":
            t = torch.rand((batch_size,), device=device)
        elif method == "sigmoid":
            t = torch.sigmoid(sigmoid_scale * torch.randn((batch_size,), device=device))
        elif method == "logit_normal":
            t = torch.sigmoid(
                torch.normal(logit_mean, logit_std, size=(batch_size,), device=device)
            )
        elif method == "shift":
            z = torch.randn((batch_size,), device=device) * sigmoid_scale
            t_base = torch.sigmoid(z)
            t = (t_base * flow_shift) / (1.0 + (flow_shift - 1.0) * t_base)
        else:
            raise ValueError(f"Unknown timestep sampling method: {method}")

        # In-place clamp: t is a fresh tensor we just created, safe to mutate.
        return t.mul_(max_t - min_t).add_(min_t).clamp_(min_t, max_t)

    @staticmethod
    def _apply_noise_offset(
        noise: Tensor,
        offset: float,
        t: Tensor | None = None,
        offset_type: str = "simple",
    ) -> None:
        """Apply channel-wise noise offset in-place. Handles 4D (image) and 5D (video).

        Adds a per-channel random bias scaled by ``offset`` to the noise tensor.
        This encourages the model to learn overall brightness/color shifts.

        Args:
            noise:       Noise tensor of shape (B, C, ...) - 4D or 5D.
            offset:      Noise offset magnitude. If <= 0, this is a no-op.
            t:           Per-sample interpolation coefficient, shape (B,).
                         Required for ``offset_type="generalized"``, ignored otherwise.
            offset_type: "simple" (constant offset) or "generalized" (offset scales
                         with sqrt(t), from "Generalized Diffusion Model with Adjusted
                         Offset Noise"). Falls back to "simple" when ``t`` is None.
        """
        if offset <= 0:
            return
        ndim = noise.ndim
        shape = (noise.shape[0], noise.shape[1]) + (1,) * (ndim - 2)
        if offset_type == "generalized" and t is not None:
            # psi(t) = offset * sqrt(t), one scale per batch sample
            # Broadcast t: (B,) -> (B, 1, 1, ...) to match noise dims
            effective_scale = offset * t.sqrt()
            # effective_scale: (B,) - reshape to (B, 1, 1, ...) for broadcasting
            scale_shape = (noise.shape[0],) + (1,) * (ndim - 1)
            effective_scale = effective_scale.view(scale_shape)
            channel_offset = torch.randn(*shape, device=noise.device, dtype=noise.dtype)
            noise.add_(channel_offset * effective_scale)
        else:
            noise.add_(
                torch.randn(*shape, device=noise.device, dtype=noise.dtype),
                alpha=offset,
            )

    @staticmethod
    def _apply_dynamic_shift(
        t: Tensor,
        seq_len: int,
        shift_base: float,
        shift_max: float,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
    ) -> Tensor:
        """Apply resolution-dependent timestep shifting (Flux/SD3 style).

        Higher-resolution images get shifted toward higher noise levels to
        compensate for the increased number of tokens.

        The shift parameter mu is linearly interpolated between shift_base
        (at base_seq_len) and shift_max (at max_seq_len), then applied as a
        sigmoid-like warp: ``t' = (mu * t) / (1 + (mu - 1) * t)``.

        Args:
            t:            Sampled timesteps, shape (B,), values in [0, 1].
            seq_len:      Token sequence length for the current batch.
            shift_base:   Target mu at base_seq_len.
            shift_max:    Target mu at max_seq_len.
            base_seq_len: Sequence length corresponding to shift_base.
            max_seq_len:  Sequence length corresponding to shift_max.

        Returns:
            Shifted timesteps, shape (B,), values in [0, 1].
        """
        m = (shift_max - shift_base) / (max_seq_len - base_seq_len)
        b = shift_base - m * base_seq_len
        mu = seq_len * m + b
        return (mu * t) / (1 + (mu - 1) * t)

    @staticmethod
    def _apply_progressive_blend(t: Tensor, step: int, warmup_steps: int) -> Tensor:
        """Blend timesteps from uniform toward their original distribution.

        At step 0: pure uniform distribution.
        At step >= warmup_steps: full target distribution (no modification).
        Between: linear interpolation between uniform and the target distribution.

        This avoids shocking the model with extreme timestep imbalance at the
        start of training and instead gradually introduces the full distribution.

        Args:
            t:             Sampled timesteps from the target distribution, shape (B,).
            step:          Current training step (0-indexed).
            warmup_steps:  Number of steps over which to blend.

        Returns:
            Blended timesteps, shape (B,). Same tensor (no copy) if step >= warmup_steps.
        """
        if step >= warmup_steps or warmup_steps <= 0:
            return t
        blend = step / warmup_steps  # 0.0 -> 1.0
        uniform = torch.rand_like(t)
        return blend * t + (1 - blend) * uniform

    @staticmethod
    def _rescale_zero_terminal_snr(alphas_cumprod: Tensor) -> Tensor:
        """Rescale noise schedule so the final timestep has SNR=0.

        Applies the rescaling from Lin et al. 2024, "Common Diffusion Noise
        Schedules and Sample Steps are Flawed" (https://arxiv.org/abs/2305.08891).

        Only relevant for DDPM-style schedules (epsilon / v-prediction). Flow-
        matching architectures already reach SNR=0 at t=1 by construction.

        The procedure:
          1. Decompose alpha_bar into its sqrt factors.
          2. Shift both factors so that sqrt_one_minus[-1] == 0.
          3. Recompose alpha_bar and clamp the last element to exactly 0.0.

        Args:
            alphas_cumprod: Precomputed cumulative alpha products, shape (T,).
                            Values must be in (0, 1].

        Returns:
            Rescaled alphas_cumprod, shape (T,), with the final entry == 0.0.
        """
        sqrt_alpha = alphas_cumprod.sqrt()
        sqrt_one_minus = (1.0 - alphas_cumprod).sqrt()

        # Shift so terminal sqrt_one_minus lands at 0
        last_sqrt_one_minus = sqrt_one_minus[-1]
        sqrt_one_minus = sqrt_one_minus - last_sqrt_one_minus

        # Rescale sqrt_alpha to keep the ratio consistent
        # The rescaling factor makes the last sqrt_alpha == sqrt(1 - 0^2) == 1,
        # but since the last alpha_bar must be 0, we derive the factor from the
        # original terminal value: scale = sqrt(1 - last_sqrt_one_minus^2).
        scale = (1.0 - last_sqrt_one_minus ** 2).sqrt()
        sqrt_alpha = sqrt_alpha * scale

        # Recompose: alpha_bar = sqrt_alpha^2
        result = sqrt_alpha ** 2
        # Clamp final entry to exactly 0 to prevent floating-point residuals
        result[-1] = 0.0
        return result
