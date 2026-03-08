"""FramePack model strategy.

Ported from Musubi_Tuner's fpack_train_network.py, following the WanStrategy template.

Key differences from WanStrategy:
- Always I2V: batch always contains image_embeddings (CLIP/SigLIP features)
- Packed temporal context: batch contains 1x/2x/4x clean latents + indices
- Guidance embedding: embedded into timestep conditioning (distilled guidance scale)
- Forward pass signature differs significantly from Wan (packed model API)
- VAE dtype fixed at float16, DiT dtype fixed at bfloat16
"""
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
# FramePackStrategy
# ---------------------------------------------------------------------------

class FramePackStrategy(ModelStrategy):
    """Strategy for FramePack (HunyuanVideo-packed) I2V architecture.

    Packed temporal format:
    - Batch contains: latents (noisy), latent_indices, latents_clean (1x),
      clean_latent_indices, optionally latents_clean_2x/4x and their indices
    - Image conditioning: image_embeddings (CLIP/SigLIP features)
    - Text conditioning: llama_vec + llama_attention_mask + clip_l_pooler
    - Flow matching: target = noise - latents (standard direction)
    """

    @property
    def architecture(self) -> str:
        return "framepack"

    @property
    def supports_video(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load FramePack DiT model from checkpoint.

        Config fields used:
            model.base_model_path       - path to packed DiT .safetensors
            model.dtype                 - training dtype (bf16 default)
            model.attn_mode             - attention backend (sdpa, flash, etc.)
            model.split_attn            - per-sample split attention
            model.quantization          - None, "fp8_scaled"
            model.block_swap_count      - block swap CPU<->GPU count
            model.gradient_checkpointing - enable gradient checkpointing
            model.model_kwargs.guidance_scale - distilled guidance scale
        """
        from trainer.arch.framepack.components.configs import FRAMEPACK_CONFIGS
        from trainer.arch.framepack.components.model import load_packed_model

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"

        fp_config = FRAMEPACK_CONFIGS["framepack"]

        # Determine loading device
        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        # Load model
        model = load_packed_model(
            device=device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            loading_device=loading_device,
            fp8_scaled=fp8_scaled,
            split_attn=split_attn,
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

        # Guidance scale (embedded as conditioning, scaled to [0, 1000] range)
        guidance_scale = cfg.model.model_kwargs.get(
            "guidance_scale", fp_config.default_guidance_scale
        )

        # Pre-compute flow shift constant for "shift" timestep method
        dfs = cfg.training.discrete_flow_shift
        flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        # Cache all config values - no Pydantic access in hot path
        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._fp_config = fp_config
        self._guidance_scale = guidance_scale
        self._noise_offset_val = cfg.training.noise_offset
        self._noise_offset_type = cfg.training.noise_offset_type
        self._flow_shift = flow_shift
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
                "fp_config": fp_config,
                "guidance_scale": guidance_scale,
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
        """Sample timesteps for FramePack flow matching training.

        Returns:
            t: Float [B] in [min_t, max_t] - interpolation coefficient.
            timesteps: Float [B] in [1, 1001] - model's expected timestep range.
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
        """Flow matching training step for FramePack.

        Expected batch keys:
            latents                  - [B, 16, T, H, W] noisy target latents
            latent_indices           - [B, T] temporal position indices
            latents_clean            - [B, 16, T1, H, W] 1x clean context latents
            clean_latent_indices     - [B, T1] indices for 1x latents
            latents_clean_2x         - (optional) [B, 16, T2, H, W]
            clean_latent_2x_indices  - (optional) [B, T2]
            latents_clean_4x         - (optional) [B, 16, T4, H, W]
            clean_latent_4x_indices  - (optional) [B, T4]
            llama_vec                - [B, L, 4096] LLaMA text features
            llama_attention_mask     - [B, L] text attention mask
            clip_l_pooler            - [B, 768] CLIP-L pooled text embedding
            image_embeddings         - [B, L_img, 1152] SigLIP image features

        Flow matching loss:
            target = noise - latents
            noisy = (1 - t) * latents + t * noise
            loss = MSE(model_pred, target)
        """
        device = self._device
        train_dtype = self._train_dtype

        # --- Extract batch tensors ---
        latents = batch["latents"].to(device=device, dtype=train_dtype)

        # Text conditioning
        llama_vec = batch["llama_vec"].to(device=device, dtype=train_dtype)
        llama_attention_mask = batch["llama_attention_mask"].to(device=device)
        clip_l_pooler = batch["clip_l_pooler"].to(device=device, dtype=train_dtype)

        # Image conditioning (always present for FramePack I2V)
        image_embeddings = batch["image_embeddings"].to(device=device, dtype=train_dtype)

        # Temporal indices
        latent_indices = batch["latent_indices"].to(device=device)

        # 1x clean latents (conditioning frame)
        latents_clean = batch["latents_clean"].to(device=device, dtype=train_dtype)
        clean_latent_indices = batch["clean_latent_indices"].to(device=device)

        # Optional multi-scale clean latents
        latents_clean_2x = None
        clean_latent_2x_indices = None
        if "latents_clean_2x" in batch and "clean_latent_2x_indices" in batch:
            latents_clean_2x = batch["latents_clean_2x"].to(device=device, dtype=train_dtype)
            clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(device=device)

        latents_clean_4x = None
        clean_latent_4x_indices = None
        if "latents_clean_4x" in batch and "clean_latent_4x_indices" in batch:
            latents_clean_4x = batch["latents_clean_4x"].to(device=device, dtype=train_dtype)
            clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(device=device)

        # --- Noise ---
        noise = torch.empty_like(latents).normal_()

        # --- Timesteps ---
        bsz = latents.shape[0]
        t, timesteps = self._sample_timesteps(bsz, device)

        # Apply noise offset in-place if configured
        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # --- Create noisy latents ---
        t_expanded = t.to(dtype=train_dtype).view(-1, 1, 1, 1, 1)
        noisy_model_input = (1.0 - t_expanded) * latents + t_expanded * noise

        # --- Distilled guidance embedding ---
        # FramePack uses embedded guidance (not CFG at inference) - 
        # guidance scale is fixed and embedded as a conditioning vector.
        guidance = torch.tensor(
            [self._guidance_scale * 1000.0] * bsz,
            device=device,
            dtype=train_dtype,
        )

        # --- Forward pass ---
        model_output = components.model(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=llama_vec,
            encoder_attention_mask=llama_attention_mask,
            pooled_projections=clip_l_pooler,
            guidance=guidance,
            latent_indices=latent_indices,
            clean_latents=latents_clean,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=latents_clean_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=latents_clean_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            image_embeddings=image_embeddings,
            return_dict=False,
        )
        model_pred = model_output[0]  # (hidden_states,) tuple

        # --- Flow matching loss: target = noise - latents ---
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
