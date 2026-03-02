"""SD3 model strategy. Flow matching with MMDiT architecture."""
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


class SD3Strategy(ModelStrategy):
    """Strategy for SD3 image generation (MMDiT, flow matching).

    Custom SD3Transformer2DModel with JointTransformerBlock for bidirectional
    text+image attention. Image-only, 16-channel latents, patch size 2.

    Supports:
    - sd3-medium  (24 joint blocks, no single blocks)
    - sd3.5-medium (24 joint + 12 single blocks)
    - sd3.5-large  (38 joint + 12 single blocks)

    Flow matching: noisy = (1-t)*x0 + t*noise, target = noise - x0.
    Timesteps are sampled in [0, 1] and scaled to [0, 1000] for the model.
    """

    @property
    def architecture(self) -> str:
        return "sd3"

    @property
    def supports_video(self) -> bool:
        return False

    def setup(self) -> ModelComponents:
        """Load SD3 model from checkpoint and cache hot-path constants.

        Config fields used:
            model.base_model_path     — path to transformer .safetensors
            model.dtype               — training dtype (bf16, fp16, fp32)
            model.gradient_checkpointing — enable gradient checkpointing
            model.model_kwargs        — {"model_version": "sd3-medium"} etc.
            training.noise_offset     — additive noise offset
            training.discrete_flow_shift — flow shift exponent (0 = no shift)
            training.timestep_sampling — uniform, sigmoid, logit_normal, shift
            training.min_timestep     — lower bound for t in [0, 1]
            training.max_timestep     — upper bound for t in [0, 1]
            training.sigmoid_scale    — scale for sigmoid/shift method
            training.logit_mean       — mean for logit_normal method
            training.logit_std        — std for logit_normal method
        """
        from trainer.arch.sd3.components.configs import SD3_CONFIGS
        from trainer.arch.sd3.components.model import load_sd3_model

        cfg = self.config
        train_dtype = _resolve_dtype(cfg.model.dtype)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_version = cfg.model.model_kwargs.get("model_version", "sd3-medium")
        if model_version not in SD3_CONFIGS:
            raise ValueError(
                f"Unknown SD3 model_version '{model_version}'. "
                f"Available: {list(SD3_CONFIGS)}"
            )
        sd3_config = SD3_CONFIGS[model_version]

        model = load_sd3_model(
            config=sd3_config,
            device=device,
            path=cfg.model.base_model_path,
            dtype=train_dtype,
        )

        # Apply quantization before gradient checkpointing (quantization changes layer types)
        self._quantize_model(model, cfg)

        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Cache hot-path constants — avoid Pydantic attribute access in training loop
        self._device = device
        self._train_dtype = train_dtype
        self._sd3_config = sd3_config
        self._noise_offset_val = cfg.training.noise_offset
        self._noise_offset_type = cfg.training.noise_offset_type

        # Timestep sampling config
        dfs = cfg.training.discrete_flow_shift
        self._flow_shift = math.exp(dfs) if dfs != 0 else 1.0
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
            extra={"sd3_config": sd3_config, "model_version": model_version},
        )

    def _sample_timesteps(self, bsz: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timesteps for SD3 flow-matching training.

        Returns:
            t:         Float [B] in [min_t, max_t] — interpolation coefficient.
            timesteps: Float [B] in [0, 1000] — SD3 model-facing timestep.
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
        # SD3 model expects timesteps scaled to [0, 1000]
        timesteps = t * 1000.0
        return t, timesteps

    def training_step(
        self,
        components: ModelComponents,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> TrainStepOutput:
        """Flow-matching training step for SD3.

        Batch format:
            latents:    (B, 16, H/8, W/8) — 16-channel latents
            ctx_vec:    (B, L, 4096)      — T5-XXL text embeddings
            pooled_vec: (B, 2048)         — concatenated CLIP-L + CLIP-G pooled (optional)

        Pipeline:
        1. Sample noise and timesteps t in [0, 1].
        2. Compute noisy latents: (1-t)*latents + t*noise.
        3. Forward: model(noisy, ctx_vec, timestep*1000, pooled).
        4. Flow-matching target = noise - latents.
        5. Loss = MSE(pred, target).
        """
        device = self._device
        train_dtype = self._train_dtype

        latents = batch["latents"].to(device=device, dtype=train_dtype)
        ctx_vec = batch["ctx_vec"].to(device=device, dtype=train_dtype)
        bsz = latents.shape[0]

        # Pooled embeddings (CLIP-L 768 + CLIP-G 1280 = 2048)
        if "pooled_vec" in batch:
            pooled = batch["pooled_vec"].to(device=device, dtype=train_dtype)
        else:
            pooled = torch.zeros(
                bsz, self._sd3_config.pooled_projection_dim,
                device=device, dtype=train_dtype,
            )

        # Noise: in-place fill avoids allocation vs torch.randn_like
        noise = torch.empty_like(latents).normal_()

        # Timesteps
        t, timesteps = self._sample_timesteps(bsz, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # Flow-matching noisy latents: x_t = (1-t)*x0 + t*noise
        t_expanded = t.to(train_dtype).view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise

        # Forward pass
        model_pred = components.model(
            hidden_states=noisy_latents,
            encoder_hidden_states=ctx_vec,
            timestep=timesteps,
            pooled_projections=pooled,
        )

        # Flow-matching loss: target velocity = noise - latents (not just noise)
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
