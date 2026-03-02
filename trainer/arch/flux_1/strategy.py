"""Flux 1 model strategy. Flow matching with dual-stream transformer.

Flux 1 uses:
- 16-channel latents packed 2x2 to 64-channel tokens
- 3D RoPE (16, 56, 56) — image-only, no temporal dimension
- T5-XXL text encoder + CLIP-L pooled conditioning
- Per-block AdaLayerNorm modulation
- GEGLU activation
- 19 double + 38 single stream blocks
- Flow matching (rectified flow) training objective
"""
from __future__ import annotations

import logging
import math
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


class Flux1Strategy(ModelStrategy):
    """Strategy for Flux 1 image generation (dual-stream, flow matching).

    Dual-stream architecture with 16-channel latents packed to 64, 3D RoPE,
    T5+CLIP text encoders. Image-only. Supports block swap.
    """

    @property
    def architecture(self) -> str:
        return "flux_1"

    @property
    def supports_video(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load Flux1Transformer from checkpoint, detect dtype, return ModelComponents.

        Config fields used:
            model.base_model_path        — path to DiT .safetensors
            model.dtype                  — training dtype (bf16, fp16, fp32)
            model.attn_mode              — attention backend
            model.split_attn             — whether to use split attention
            model.quantization           — None, "fp8_scaled"
            model.block_swap_count       — number of blocks to swap CPU<->GPU
            model.gradient_checkpointing — enable gradient checkpointing
            model.model_kwargs           — extra kwargs, e.g. {"model_version": "dev"}
        """
        from trainer.arch.flux_1.components.configs import FLUX1_CONFIGS
        from trainer.arch.flux_1.components.model import detect_flux1_weight_dtype, load_flux1_model

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"

        model_version = cfg.model.model_kwargs.get("model_version", "dev")
        if model_version not in FLUX1_CONFIGS:
            raise ValueError(
                f"Unknown Flux 1 model_version '{model_version}'. "
                f"Available: {list(FLUX1_CONFIGS)}"
            )
        flux1_config = FLUX1_CONFIGS[model_version]

        if fp8_scaled:
            dit_weight_dtype = None
        else:
            detected_dtype = detect_flux1_weight_dtype(dit_path)
            dit_weight_dtype = (
                detected_dtype
                if (detected_dtype is not None and train_dtype == detected_dtype)
                else train_dtype
            )

        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        model = load_flux1_model(
            config=flux1_config,
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

        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        if blocks_to_swap > 0:
            model.enable_block_swap(blocks_to_swap, device, supports_backward=True)
            model.move_to_device_except_swap_blocks(device)

        # Pre-compute and cache all constants used in the hot path
        noise_offset_val = cfg.training.noise_offset
        dfs = cfg.training.discrete_flow_shift
        flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._flux1_config = flux1_config
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
            extra={"flux1_config": flux1_config, "model_version": model_version},
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
        """Sample timesteps for Flux 1 flow-matching training.

        Returns:
            t:         Float [B] in [min_t, max_t] — interpolation coefficient.
            timesteps: Float [B] in [0, 1] — Flux 1 model receives timesteps in [0, 1].
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
        # Flux 1 model receives timesteps in [0, 1] (not scaled to 1000)
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
        """Flow-matching training step for Flux 1.

        Batch format:
            latents:  (B, 16, H, W) — 16-channel raw latents.
            ctx_vec:  (B, L, 4096)  — T5-XXL text embeddings.
            pooled:   (B, 768)      — CLIP-L pooled text embedding (optional).

        Pipeline:
        1. Pack spatial dims: (B, 16, H, W) -> (B, HW/4, 64) + 3D position IDs.
        2. Sample noise and timesteps.
        3. Noisy latents: x_t = (1-t)*x_0 + t*noise.
        4. Forward through Flux1Transformer.
        5. Unpack: (B, HW/4, 64) -> (B, 16, H, W).
        6. MSE loss against flow-matching target (noise - latents).
        """
        from trainer.arch.flux_1.components.utils import (
            pack_latents,
            unpack_latents,
            prepare_img_ids,
            prepare_txt_ids,
        )

        device = self._device
        train_dtype = self._train_dtype

        # Extract batch
        latents = batch["latents"].to(device=device, dtype=train_dtype)
        # Latents are (B, 16, H, W) — raw 16-channel
        packed_latent_h = latents.shape[2] // 2  # packed spatial height (H/2)
        packed_latent_w = latents.shape[3] // 2  # packed spatial width  (W/2)

        ctx_vec = batch["ctx_vec"].to(device=device, dtype=train_dtype)  # (B, L, 4096)

        # Optional CLIP-L pooled text embedding
        pooled_text: torch.Tensor | None = None
        if "pooled" in batch:
            pooled_text = batch["pooled"].to(device=device, dtype=train_dtype)

        # Noise (in-place normal_ for efficiency)
        noise = torch.empty_like(latents).normal_()

        # Timesteps
        bsz = latents.shape[0]
        t, timesteps = self._sample_timesteps(bsz, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # Noisy latents (interpolate in raw space before packing)
        t_expanded = t.to(train_dtype).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * noise

        # Pack: (B, 16, H, W) -> (B, HW/4, 64) + 3D position IDs
        noisy_packed = pack_latents(noisy_model_input)   # (B, h*w, 64)
        img_ids = prepare_img_ids(packed_latent_h, packed_latent_w).to(device=device)
        txt_ids = prepare_txt_ids(ctx_vec.shape[1]).to(device=device)

        # Expand IDs for batch dimension
        img_ids = img_ids.expand(bsz, -1, -1)  # (B, HW, 3)
        txt_ids = txt_ids.expand(bsz, -1, -1)  # (B, L, 3)

        # Guidance vector: use 1.0 for fine-tuning (not CFG); schnell gets None
        guidance_vec: torch.Tensor | None = None
        if self._flux1_config.use_guidance_embed:
            guidance_vec = torch.ones(bsz, device=device, dtype=train_dtype)

        # Gradient checkpointing: ensure inputs require grad
        if batch.get("gradient_checkpointing", False):
            noisy_packed = noisy_packed.requires_grad_(True)
            ctx_vec = ctx_vec.requires_grad_(True)

        # Forward pass — Flux1Transformer returns (B, HW, 64)
        model_pred = components.model(
            x=noisy_packed,
            x_ids=img_ids,
            timesteps=timesteps,
            ctx=ctx_vec,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
            pooled_text=pooled_text,
        )

        # Unpack: (B, HW, 64) -> (B, 16, H, W)
        model_pred = unpack_latents(model_pred, packed_latent_h, packed_latent_w)

        # Flow-matching loss: target = velocity = noise - latents
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
