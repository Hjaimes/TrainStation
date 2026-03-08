"""HunyuanVideo model strategy.

Follows the exact pattern of trainer/arch/wan/strategy.py.
All config values are cached in setup() - no Pydantic attribute access in hot path.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from trainer.arch.base import ModelComponents, ModelStrategy, TrainStepOutput

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}. Choose from: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


class HunyuanVideoStrategy(ModelStrategy):
    """Strategy for HunyuanVideo (HYVideo-T/2-cfgdistill) training."""

    @property
    def architecture(self) -> str:
        return "hunyuan_video"

    @property
    def supports_video(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load HunyuanVideoTransformer3DModel from checkpoint.

        Config fields used:
            model.base_model_path       - path to transformer .safetensors
            model.dtype                 - training dtype (bf16, fp16, fp32)
            model.attn_mode             - attention backend (torch, flash, sageattn, etc.)
            model.split_attn            - whether to use split attention
            model.block_swap_count      - number of blocks to swap CPU↔GPU
            model.gradient_checkpointing - enable gradient checkpointing
            model.model_kwargs          - extra kwargs, e.g. {"guidance_scale": 7.0}
        """
        from .components.configs import HUNYUAN_VIDEO_CONFIG
        from .components.embeddings import get_rotary_pos_embed_by_shape
        from .components.model import load_hunyuan_video_model

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        blocks_to_swap = cfg.model.block_swap_count

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        model = load_hunyuan_video_model(
            dit_path=dit_path,
            config=HUNYUAN_VIDEO_CONFIG,
            device=device,
            dtype=train_dtype,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
        )

        # Apply quantization before gradient checkpointing (quantization changes layer types)
        self._quantize_model(model, cfg)

        # Gradient checkpointing
        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Block swap setup
        if blocks_to_swap > 0:
            model.enable_block_swap(blocks_to_swap, device, supports_backward=True)
            model.move_to_device_except_swap_blocks(device)

        # ---- Cache ALL hot-path values ----
        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._hv_config = HUNYUAN_VIDEO_CONFIG
        self._patch_size = HUNYUAN_VIDEO_CONFIG.patch_size
        # Guidance scale from model_kwargs (default 1.0 = no guidance)
        self._guidance_scale = float(
            cfg.model.model_kwargs.get("guidance_scale", 1.0)
        )

        # Noise
        self._noise_offset_val = cfg.training.noise_offset
        self._noise_offset_type = cfg.training.noise_offset_type

        # Timestep sampling
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
            extra={
                "hv_config": HUNYUAN_VIDEO_CONFIG,
                "patch_size": self._patch_size,
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_before_accelerate_prepare(
        self, components: ModelComponents, accelerator: Any,
    ) -> dict[str, Any]:
        return {"device_placement": self._blocks_to_swap == 0}

    def on_after_accelerate_prepare(
        self, components: ModelComponents, accelerator: Any,
    ) -> None:
        if self._blocks_to_swap > 0:
            model = accelerator.unwrap_model(components.model)
            model.move_to_device_except_swap_blocks(self._device)
            model.prepare_block_swap_before_forward()

    def on_before_training_step(self, components: ModelComponents) -> None:
        if self._blocks_to_swap > 0:
            model = components.model
            if hasattr(model, "module"):
                model = model.module
            model.prepare_block_swap_before_forward()

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def _sample_timesteps(self, bsz: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timesteps for HunyuanVideo flow matching training.

        Returns:
            t: [B] float in [min_t, max_t] - interpolation coefficient.
            timesteps: [B] float in [1, 1001] - model's expected range.
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
        """Flow matching training step for HunyuanVideo.

        Batch keys expected:
            latents      : [B, 16, F, H, W] - pre-encoded video latents
            text_states  : [B, S, 4096] - LLM (Llama) text embeddings
            text_mask    : [B, S] - attention mask for text_states
            text_states_2: [B, 768] - CLIP pooled embeddings

        Returns:
            TrainStepOutput with scalar loss and detached metrics.
        """
        device = self._device
        train_dtype = self._train_dtype

        # ---- Extract batch ----
        latents = batch["latents"].to(device=device, dtype=train_dtype)
        text_states = batch["text_states"].to(device=device, dtype=train_dtype)
        text_mask = batch["text_mask"].to(device=device)
        text_states_2 = batch["text_states_2"].to(device=device, dtype=train_dtype)

        # ---- Noise ----
        noise = torch.empty_like(latents).normal_()

        # ---- Timesteps ----
        bsz = latents.shape[0]
        t, timesteps = self._sample_timesteps(bsz, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # ---- Noisy latents (flow matching interpolation) ----
        t_expanded = t.to(dtype=train_dtype).view(-1, 1, 1, 1, 1)
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * noise

        # ---- RoPE position embeddings (computed from actual latent shape) ----
        from .components.embeddings import get_rotary_pos_embed_by_shape
        hv_cfg = self._hv_config
        latent_size = list(latents.shape[2:])  # [F, H, W]
        freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(
            patch_size=hv_cfg.patch_size,
            hidden_size=hv_cfg.hidden_size,
            heads_num=hv_cfg.heads_num,
            rope_dim_list=hv_cfg.rope_dim_list,
            rope_theta=hv_cfg.rope_theta,
            latents_size=latent_size,
        )
        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)

        # ---- Guidance tensor (cfg-distilled model) ----
        guidance = None
        if hv_cfg.guidance_embed:
            # guidance is provided in units of 1000 (like timesteps)
            guidance = torch.full(
                (bsz,), self._guidance_scale * 1000.0,
                device=device, dtype=train_dtype,
            )

        # ---- Forward pass ----
        model_pred = components.model(
            noisy_model_input,
            t=timesteps,
            text_states=text_states,
            text_mask=text_mask,
            text_states_2=text_states_2,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            guidance=guidance,
        )

        # ---- Loss: flow matching target = noise - latents ----
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
