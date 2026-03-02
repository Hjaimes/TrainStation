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
    (0..999) and alpha_bar noise schedule instead of continuous t in [0,1].
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

        # Apply quantization before gradient checkpointing (quantization changes layer types)
        self._quantize_model(model, cfg)

        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Pre-compute alpha_bar schedule on device
        alphas_cumprod = compute_alphas_cumprod(sdxl_config.num_train_timesteps).to(device)

        # Optionally rescale so the final timestep has SNR=0 (Lin et al. 2024).
        # Flow-matching models already satisfy this; SDXL (DDPM) does not by default.
        if cfg.training.zero_terminal_snr:
            alphas_cumprod = self._rescale_zero_terminal_snr(alphas_cumprod)
            logger.info("Applied zero-terminal-SNR rescaling to SDXL noise schedule.")

        # Cache everything for training_step
        self._device = device
        self._train_dtype = train_dtype
        self._sdxl_config = sdxl_config
        self._alphas_cumprod = alphas_cumprod
        self._noise_offset_val = cfg.training.noise_offset
        self._ts_min = int(cfg.training.min_timestep * sdxl_config.num_train_timesteps)
        self._ts_max = int(cfg.training.max_timestep * sdxl_config.num_train_timesteps)

        self._mask_weight = cfg.data.mask_weight
        self._normalize_masked_loss = cfg.data.normalize_masked_area_loss

        self._setup_loss_fn(cfg)
        self._setup_loss_weighting(cfg)

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
        from trainer.arch.sdxl.components.utils import build_time_ids, get_velocity

        device = self._device
        train_dtype = self._train_dtype
        config = self._sdxl_config

        latents = batch["latents"].to(device=device, dtype=train_dtype)
        ctx_vec = batch["ctx_vec"].to(device=device, dtype=train_dtype)
        bsz = latents.shape[0]

        # Pooled embeddings
        if "pooled_vec" in batch:
            pooled = batch["pooled_vec"].to(device=device, dtype=train_dtype)
        else:
            pooled = torch.zeros(bsz, 1280, device=device, dtype=train_dtype)

        # Noise
        noise = torch.empty_like(latents).normal_()
        self._apply_noise_offset(noise, self._noise_offset_val)

        # Discrete timesteps
        timesteps = torch.randint(self._ts_min, self._ts_max, (bsz,), device=device, dtype=torch.long)

        # DDPM noisy latents
        alpha_bar_t = self._alphas_cumprod[timesteps].to(train_dtype)
        sqrt_alpha_bar = alpha_bar_t.sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1, 1)
        noisy_latents = sqrt_alpha_bar * latents + sqrt_one_minus_alpha_bar * noise

        # Time IDs
        h_pixels = latents.shape[2] * 8
        w_pixels = latents.shape[3] * 8
        time_ids = build_time_ids(
            original_size=(h_pixels, w_pixels),
            crop_coords=(0, 0),
            target_size=(h_pixels, w_pixels),
            dtype=train_dtype,
        ).to(device).unsqueeze(0).expand(bsz, -1)

        added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

        # Forward
        model_pred = components.model(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=ctx_vec,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Loss
        if config.prediction_type == "epsilon":
            target = noise
        else:
            target = get_velocity(latents, noise, alpha_bar_t.view(-1, 1, 1, 1))

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
                "timestep_mean": timesteps.float().mean().detach(),
            },
        )
