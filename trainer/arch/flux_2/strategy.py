"""Flux 2 model strategy. Ported from Musubi_Tuner's flux_2_train_network.py."""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from trainer.arch.base import ModelStrategy, ModelComponents, TrainStepOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype helpers (shared with Wan strategy)
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
# Flux2Strategy
# ---------------------------------------------------------------------------

class Flux2Strategy(ModelStrategy):
    """Strategy for Flux 2 image generation architectures (dev, klein-4b/9b, etc.).

    Dual-stream architecture with 128-channel latents, 4-D position IDs, and
    Mistral3 / Qwen3 text encoders. Image-only (supports_video = False).
    """

    @property
    def architecture(self) -> str:
        return "flux_2"

    @property
    def supports_video(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load Flux2Model from checkpoint, detect dtype, return ModelComponents.

        Config fields used:
            model.base_model_path    - path to DiT .safetensors
            model.dtype              - training dtype (bf16, fp16, fp32)
            model.attn_mode          - attention backend (sdpa, flash, etc.)
            model.split_attn         - whether to use split attention
            model.quantization       - None, "fp8", "fp8_scaled"
            model.block_swap_count   - number of blocks to swap CPU↔GPU
            model.gradient_checkpointing - enable gradient checkpointing
            model.model_kwargs       - extra kwargs, e.g. {"model_version": "dev"}
        """
        from trainer.arch.flux_2.components.configs import FLUX2_CONFIGS
        from trainer.arch.flux_2.components.model import (
            detect_flux2_weight_dtype,
            load_flux2_model,
        )

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"

        # Resolve model variant ("dev", "klein-4b", "klein-9b", etc.)
        model_version = cfg.model.model_kwargs.get("model_version", "dev")
        if model_version not in FLUX2_CONFIGS:
            raise ValueError(
                f"Unknown Flux 2 model_version '{model_version}'. "
                f"Available: {list(FLUX2_CONFIGS)}"
            )
        flux2_config = FLUX2_CONFIGS[model_version]

        # Detect weight dtype from checkpoint
        if fp8_scaled:
            dit_weight_dtype = None
        else:
            detected_dtype = detect_flux2_weight_dtype(dit_path)
            dit_weight_dtype = detected_dtype if (detected_dtype is not None and train_dtype == detected_dtype) else train_dtype

        # Determine loading device (CPU for block swap to save VRAM)
        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        # Load model
        model = load_flux2_model(
            config=flux2_config,
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
            model.enable_block_swap(blocks_to_swap, device, supports_backward=True)
            model.move_to_device_except_swap_blocks(device)

        # Pre-compute and cache all constants used in the hot path
        noise_offset_val = cfg.training.noise_offset
        dfs = cfg.training.discrete_flow_shift
        flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        # Cache everything so training_step never does Pydantic attribute access
        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._flux2_config = flux2_config
        self._model_version = model_version
        self._noise_offset_val = noise_offset_val
        self._noise_offset_type = cfg.training.noise_offset_type
        self._flow_shift = flow_shift
        self._guidance_scale = cfg.training.guidance_scale

        # Timestep sampling config
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
                "flux2_config": flux2_config,
                "model_version": model_version,
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
            if hasattr(model, "module"):
                model = model.module
            model.prepare_block_swap_before_forward()

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def _sample_timesteps(self, bsz: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timesteps for Flux 2 flow-matching training.

        Returns:
            t:         Float [B] in [min_t, max_t] - interpolation coefficient.
            timesteps: Float [B] in [0, 1] - Flux 2 model receives timesteps in [0, 1].
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
        # Flux 2 model receives timesteps in [0, 1] (not scaled to 1000)
        return t, t

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------

    def training_step(
        self,
        components: ModelComponents,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> TrainStepOutput:
        """Flow-matching training step for Flux 2.

        Batch format:
            latents:  ``(B, 128, H, W)`` - 128-channel packed latents.
            ctx_vec:  ``(B, L, D)``       - text embeddings from Mistral3/Qwen3.

        Pipeline:
        1. Pack spatial dims: ``(B, 128, H, W)`` → ``(B, HW, 128)`` + position IDs.
        2. Sample noise and timesteps.
        3. Noisy latents: ``x_t = (1-t)*x_0 + t*noise``.
        4. Forward through Flux2Model.
        5. MSE loss against flow-matching target ``noise - latents``.
        """
        device = self._device
        train_dtype = self._train_dtype

        # --- Extract batch ---
        latents = batch["latents"].to(device=device, dtype=train_dtype)
        # Latents are (B, 128, H, W) - packed 128-channel format
        packed_latent_h = latents.shape[2]
        packed_latent_w = latents.shape[3]

        ctx_vec = batch["ctx_vec"].to(device=device, dtype=train_dtype)  # (B, L, D)

        # --- Noise ---
        noise = torch.empty_like(latents).normal_()

        # --- Timesteps ---
        bsz = latents.shape[0]
        t, timesteps = self._sample_timesteps(bsz, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # --- Noisy latents ---
        t_expanded = t.to(train_dtype).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * noise

        # --- Pack latents: (B, 128, H, W) → (B, HW, 128) + position IDs ---
        from trainer.arch.flux_2.components.utils import prc_img, prc_txt

        noisy_packed, img_ids = prc_img(noisy_model_input)   # (B, HW, 128), (B, HW, 4)
        ctx, ctx_ids = prc_txt(ctx_vec)                       # (B, L, D),    (B, L, 4)

        # Move IDs to device (prc_* moves to x.device, but ensure consistency)
        img_ids = img_ids.to(device=device)
        ctx_ids = ctx_ids.to(device=device)

        # Guidance vector: use 1.0 for LoRA/fine-tune training (not CFG)
        guidance_vec: torch.Tensor | None = None
        if self._flux2_config.use_guidance_embed:
            guidance_vec = torch.ones(bsz, device=device, dtype=train_dtype)

        # --- Forward pass ---
        # Flux2Model returns (B, HW, 128) - a single tensor (not a list like Wan)
        model_pred = components.model(
            x=noisy_packed,
            x_ids=img_ids,
            timesteps=timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance_vec,
        )

        # Unpack: (B, HW, 128) → (B, 128, H, W)
        model_pred = rearrange(
            model_pred,
            "b (h w) c -> b c h w",
            h=packed_latent_h,
            w=packed_latent_w,
        )

        # --- Flow-matching loss ---
        # Target: velocity = noise - latents
        target = noise - latents
        dataset_weight = batch.get("dataset_weight")
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device=device, dtype=train_dtype)
            loss = self._compute_masked_loss(
                model_pred.to(train_dtype), target, loss_mask,
                mask_weight=self._mask_weight, normalize_by_area=self._normalize_masked_loss,
            )
            if dataset_weight is not None:
                loss = loss * dataset_weight.mean()
        else:
            loss = self._compute_loss(model_pred.to(train_dtype), target, loss_weight=dataset_weight)

        return TrainStepOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "timestep_mean": t.mean().detach(),
            },
        )
