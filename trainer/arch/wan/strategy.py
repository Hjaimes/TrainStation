"""Wan 2.1 model strategy. Ported from Musubi_Tuner's wan_train_network.py."""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from trainer.arch.base import ModelStrategy, ModelComponents, TrainStepOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


# ---------------------------------------------------------------------------
# WanStrategy
# ---------------------------------------------------------------------------

class WanStrategy(ModelStrategy):
    """Strategy for Wan 2.1 text-to-video / image-to-video architectures."""

    @property
    def architecture(self) -> str:
        return "wan"

    @property
    def supports_video(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load WanModel from checkpoint, detect dtype, return ModelComponents.

        Config fields used:
            model.base_model_path   - path to transformer .safetensors
            model.dtype             - training dtype (bf16, fp16, fp32)
            model.attn_mode         - attention backend (sdpa, flash, xformers, etc.)
            model.split_attn        - whether to use split attention
            model.quantization      - None, "fp8", "fp8_scaled"
            model.block_swap_count  - number of blocks to swap CPU↔GPU
            model.gradient_checkpointing - enable gradient checkpointing
            model.model_kwargs      - extra kwargs, e.g. {"task": "t2v-14B"}
        """
        from trainer.arch.wan.components.configs import WAN_CONFIGS
        from trainer.arch.wan.components.model import (
            detect_wan_sd_dtype,
            load_wan_model,
        )

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"

        # Resolve task config (e.g. "t2v-14B", "t2v-1.3B", "i2v-14B")
        task = cfg.model.model_kwargs.get("task", "t2v-14B")
        if task not in WAN_CONFIGS:
            raise ValueError(
                f"Unknown Wan task '{task}'. Available: {list(WAN_CONFIGS)}"
            )
        wan_config = WAN_CONFIGS[task]

        # Detect dtype from checkpoint (for non-fp8, ensures correct weight dtype)
        if fp8_scaled:
            dit_weight_dtype = None  # fp8_scaled handles its own dtype
        else:
            detected_dtype = detect_wan_sd_dtype(dit_path)
            dit_weight_dtype = detected_dtype if train_dtype == detected_dtype else train_dtype

        # Determine loading device
        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        # Load model
        model = load_wan_model(
            config=wan_config,
            device=device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=fp8_scaled,
        )

        # Apply quantization before gradient checkpointing (quantization changes layer types)
        self._quantize_model(model, cfg)

        # Enable gradient checkpointing
        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Enable block swap if requested
        if blocks_to_swap > 0:
            model.enable_block_swap(
                blocks_to_swap,
                device,
                supports_backward=True,
            )
            model.move_to_device_except_swap_blocks(device)

        # Pre-compute constants that never change during training
        patch_size = wan_config.patch_size
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        noise_offset_val = cfg.training.noise_offset
        use_gradient_checkpointing = cfg.model.gradient_checkpointing

        # Pre-compute flow shift constant for "shift" timestep method
        dfs = cfg.training.discrete_flow_shift
        flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        # Store strategy state - accessed every training step, so cache everything
        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._wan_config = wan_config
        self._task = task
        self._patch_volume = patch_volume
        self._noise_offset_val = noise_offset_val
        self._noise_offset_type = cfg.training.noise_offset_type
        self._use_gradient_checkpointing = use_gradient_checkpointing
        self._flow_shift = flow_shift
        # Pre-extract timestep config to avoid repeated Pydantic attribute access
        self._ts_method = cfg.training.timestep_sampling
        self._ts_min = cfg.training.min_timestep
        self._ts_max = cfg.training.max_timestep
        self._ts_sigmoid_scale = cfg.training.sigmoid_scale
        self._ts_logit_mean = cfg.training.logit_mean
        self._ts_logit_std = cfg.training.logit_std

        self._mask_weight = cfg.data.mask_weight
        self._normalize_masked_loss = cfg.data.normalize_masked_area_loss

        self._setup_loss_fn(cfg)
        self._setup_loss_weighting(cfg)

        return ModelComponents(
            model=model,
            extra={
                "wan_config": wan_config,
                "task": task,
                "patch_size": patch_size,
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_before_accelerate_prepare(
        self, components: ModelComponents, accelerator: Any,
    ) -> dict[str, Any]:
        """Disable device placement when using block swap."""
        return {"device_placement": self._blocks_to_swap == 0}

    def on_after_accelerate_prepare(
        self, components: ModelComponents, accelerator: Any,
    ) -> None:
        """After accelerate wraps the model, re-place blocks and prepare swap."""
        if self._blocks_to_swap > 0:
            model = accelerator.unwrap_model(components.model)
            model.move_to_device_except_swap_blocks(self._device)
            model.prepare_block_swap_before_forward()

    def on_before_training_step(self, components: ModelComponents) -> None:
        """Prepare block swap before each forward pass."""
        if self._blocks_to_swap > 0:
            model = components.model
            # Unwrap if wrapped by accelerator
            if hasattr(model, "module"):
                model = model.module
            model.prepare_block_swap_before_forward()

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def _sample_timesteps(self, bsz: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timesteps for Wan flow matching training.

        Returns:
            t: Float [B] in [min_t, max_t] - interpolation coefficient.
            timesteps: Float [B] in [1, 1001] - Wan model's expected range.
        """
        t = self._sample_t(
            bsz, device,
            method=self._ts_method,
            min_t=self._ts_min,
            max_t=self._ts_max,
            sigmoid_scale=self._ts_sigmoid_scale,
            logit_mean=self._ts_logit_mean,
            logit_std=self._ts_logit_std,
            flow_shift=self._flow_shift,
        )
        timesteps = t * 1000.0 + 1.0
        return t, timesteps

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------

    def training_step(
        self,
        components: ModelComponents,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> TrainStepOutput:
        """Flow matching training step for Wan.

        1. Extract latents and conditioning from batch.
        2. Sample noise and timesteps.
        3. Create noisy latents: noisy = (1-t) * latents + t * noise
        4. Forward pass through WanModel.
        5. Compute MSE loss against target = noise - latents.
        """
        # Use cached values instead of repeated attribute lookups
        device = self._device
        train_dtype = self._train_dtype

        # --- Extract batch data ---
        latents = batch["latents"].to(device=device, dtype=train_dtype)
        # T5 context: list of variable-length tensors (varlen key)
        raw_t5 = batch["t5"]
        if isinstance(raw_t5, list):
            context = [t.to(device=device, dtype=train_dtype) for t in raw_t5]
        else:
            context = [t.to(device=device, dtype=train_dtype) for t in raw_t5.unbind(0)]

        # --- Noise (in-place fill avoids allocation when shape is stable) ---
        noise = torch.empty_like(latents).normal_()

        # --- Timesteps ---
        bsz = latents.shape[0]
        t, timesteps = self._sample_timesteps(bsz, device)

        # Apply noise offset in-place if configured
        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # --- Create noisy latents ---
        # Cast t to train_dtype first, then reshape (one dispatch instead of two)
        t_expanded = t.to(dtype=train_dtype).view(-1, 1, 1, 1, 1)
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * noise

        # --- Compute seq_len from latent shape and pre-computed patch volume ---
        seq_len = (latents.shape[2] * latents.shape[3] * latents.shape[4]) // self._patch_volume

        # --- Forward pass ---
        # WanModel expects x as iterable of [C, F, H, W] tensors (iterates batch dim).
        # Note: gradient checkpointing uses use_reentrant=False in WanAttentionBlock,
        # so requires_grad_(True) on inputs is NOT needed.
        model_pred = components.model(
            noisy_model_input,
            t=timesteps,
            context=context,
            seq_len=seq_len,
        )

        # Model returns List[Tensor] - stack into [B, C, F, H, W]
        model_pred = torch.stack(model_pred, dim=0)

        # --- Loss ---
        # Flow matching target: velocity = noise - latents (computed inline to avoid
        # holding a named full-size intermediate tensor)
        dataset_weight = batch.get("dataset_weight")
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device=device, dtype=train_dtype)
            loss = self._compute_masked_loss(
                model_pred.to(train_dtype), noise - latents, loss_mask,
                mask_weight=self._mask_weight, normalize_by_area=self._normalize_masked_loss,
            )
            if dataset_weight is not None:
                loss = loss * dataset_weight.mean()
        else:
            loss = self._compute_loss(model_pred.to(train_dtype), noise - latents, loss_weight=dataset_weight)

        return TrainStepOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "timestep_mean": t.mean().detach(),
            },
        )
