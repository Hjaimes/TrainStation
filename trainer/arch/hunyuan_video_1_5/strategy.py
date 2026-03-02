"""HunyuanVideo 1.5 model strategy.

Ported from Musubi_Tuner's hv_1_5_train_network.py and hunyuan_video_1_5_models.py.

Key differences from HunyuanVideo (original):
- 54 double blocks, 0 single blocks
- patch_size = [1, 1, 1] — no patching, so seq_len = T*H*W exactly
- No guidance embedding → forward() has NO guidance parameter
- Single ModelOffloader (double blocks only)
- Text encoders: Qwen2.5-VL + ByT5 (embeddings come pre-cached in batch)
- Flow matching target: noise - latents (standard direction)
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
        raise ValueError(f"Unknown dtype '{name}'. Valid: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def _pad_varlen(
    seq_list: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and return (padded_tensor, bool_mask).

    Different prompts have different token counts; pads to the batch maximum.

    Args:
        seq_list: List of tensors with shape [L_i, D].
        device: Target device.
        dtype: Target dtype.

    Returns:
        padded: [B, max_L, D]
        mask:   [B, max_L] bool — True for valid positions.
    """
    lengths = [t.shape[0] for t in seq_list]
    max_len = max(lengths)
    padded = []
    for t in seq_list:
        if t.shape[0] < max_len:
            t = F.pad(t, (0, 0, 0, max_len - t.shape[0]))
        padded.append(t)
    stacked = torch.stack(padded, dim=0).to(device=device, dtype=dtype)

    mask = torch.zeros(len(seq_list), max_len, device=device, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return stacked, mask


class HunyuanVideo15Strategy(ModelStrategy):
    """Training strategy for HunyuanVideo 1.5.

    Batch keys expected:
        latents       — [B, 16, T, H, W] VAE latent video
        latents_image — [B, 17, T, H, W] I2V conditioning (16 + 1 mask); or absent for T2V
        vl_embed      — list of [L_i, 3584] Qwen2.5-VL embeddings
        byt5_embed    — list of [L_i, 1472] ByT5 embeddings
        siglip        — [B, V, 1152] SigLIP image embeddings (I2V, optional)
    """

    @property
    def architecture(self) -> str:
        return "hunyuan_video_1_5"

    @property
    def supports_video(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load HunyuanVideo 1.5 transformer from checkpoint.

        Config fields used:
            model.base_model_path     — path to DiT .safetensors
            model.dtype               — training dtype (bf16, fp16, fp32)
            model.attn_mode           — attention backend
            model.split_attn          — split-attention flag
            model.quantization        — None or "fp8_scaled"
            model.block_swap_count    — number of double blocks to swap CPU↔GPU
            model.gradient_checkpointing — enable gradient checkpointing
            model.model_kwargs.task   — "t2v" (default) or "i2v"
        """
        from trainer.arch.hunyuan_video_1_5.components.model import (
            HunyuanVideo15Transformer,
            detect_hunyuan_video_1_5_sd_dtype,
            load_hunyuan_video_1_5_model,
        )

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"
        task_type = cfg.model.model_kwargs.get("task", "t2v")
        blocks_to_swap = cfg.model.block_swap_count

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        # Determine weight dtype
        if fp8_scaled:
            dit_weight_dtype = None
        else:
            detected = detect_hunyuan_video_1_5_sd_dtype(dit_path)
            dit_weight_dtype = detected if train_dtype == detected else train_dtype

        model = load_hunyuan_video_1_5_model(
            device=device,
            task_type=task_type,
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

        # Cache all config values used in training_step to avoid repeated
        # Pydantic attribute lookups in the hot loop.
        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._task_type = task_type
        self._noise_offset_val = cfg.training.noise_offset
        self._noise_offset_type = cfg.training.noise_offset_type
        self._ts_method = cfg.training.timestep_sampling
        self._ts_min = cfg.training.min_timestep
        self._ts_max = cfg.training.max_timestep
        self._ts_sigmoid_scale = cfg.training.sigmoid_scale
        self._ts_logit_mean = cfg.training.logit_mean
        self._ts_logit_std = cfg.training.logit_std

        dfs = cfg.training.discrete_flow_shift
        self._flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        # VAE_LATENT_CHANNELS = 16 for HV 1.5
        self._latent_channels = 16

        self._mask_weight = cfg.data.mask_weight
        self._normalize_masked_loss = cfg.data.normalize_masked_area_loss

        self._setup_loss_fn(cfg)
        self._setup_loss_weighting(cfg)

        return ModelComponents(
            model=model,
            extra={
                "task_type": task_type,
                "patch_size": [1, 1, 1],
                "latent_channels": self._latent_channels,
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
        """Sample timesteps for HunyuanVideo 1.5 flow matching training.

        Returns:
            t: [B] float in [min_t, max_t] — interpolation coefficient.
            timesteps: [B] float in [1, 1001] — model's expected range.
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
        batch: dict,
        step: int,
    ) -> TrainStepOutput:
        """Flow-matching training step for HunyuanVideo 1.5.

        Steps:
        1. Extract latents [B, 16, T, H, W] and conditioning from batch.
        2. Build I2V conditioning tensor (or zeros for T2V).
        3. Sample noise and timesteps.
        4. Create noisy latents via linear interpolation.
        5. Forward: model(concat(noisy, cond), t, text_states, ...) — NO guidance param.
        6. MSE loss against target = noise - latents.
        """
        device = self._device
        dtype = self._train_dtype
        lat_ch = self._latent_channels

        # --- Latents ---
        latents = batch["latents"].to(device=device, dtype=dtype)  # [B, 16, T, H, W]
        B, C, T, H, W = latents.shape
        assert C == lat_ch, f"Expected {lat_ch} latent channels, got {C}"

        # --- I2V conditioning (cond_latents): [B, 17, T, H, W] ---
        # First 16 channels = first-frame latent, channel 16 = conditioning mask.
        cond_latents = batch.get("latents_image", None)
        if cond_latents is None:
            # T2V mode: all-zero conditioning (no image)
            cond_latents = torch.zeros(
                (B, lat_ch + 1, T, H, W), device=device, dtype=dtype
            )
        else:
            cond_latents = cond_latents.to(device=device, dtype=dtype)

        # --- Noise ---
        noise = torch.empty_like(latents).normal_()

        # --- Timesteps ---
        t, timesteps = self._sample_timesteps(B, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # --- Noisy latents (flow matching interpolation) ---
        t_expanded = t.to(dtype).view(B, 1, 1, 1, 1)
        noisy = (1.0 - t_expanded) * latents + t_expanded * noise

        # --- Concatenate noisy + cond along channel dim: [B, 16+17, T, H, W] ---
        model_input = torch.cat([noisy, cond_latents], dim=1)

        # --- Text conditioning (variable-length, pad to batch max) ---
        raw_vl = batch["vl_embed"]      # list of [L_i, 3584]
        raw_byt5 = batch["byt5_embed"]  # list of [L_i, 1472]

        if isinstance(raw_vl, torch.Tensor):
            raw_vl = list(raw_vl.unbind(0))
        if isinstance(raw_byt5, torch.Tensor):
            raw_byt5 = list(raw_byt5.unbind(0))

        vl_embed, vl_mask = _pad_varlen(raw_vl, device, dtype)         # [B, L_vl, 3584]
        byt5_embed, byt5_mask = _pad_varlen(raw_byt5, device, dtype)   # [B, L_b, 1472]

        # Vision states (SigLIP, I2V optional)
        vision_states = batch.get("siglip", None)
        if vision_states is not None:
            vision_states = vision_states.to(device=device, dtype=dtype)

        # Gradient checkpointing requires inputs to have requires_grad=True
        if self.config.model.gradient_checkpointing:
            model_input.requires_grad_(True)
            vl_embed.requires_grad_(True)
            byt5_embed.requires_grad_(True)
            if vision_states is not None:
                vision_states.requires_grad_(True)

        # --- Forward pass (NO guidance parameter) ---
        model_pred = components.model(
            hidden_states=model_input,
            timestep=timesteps,
            text_states=vl_embed,
            encoder_attention_mask=vl_mask,
            vision_states=vision_states,
            byt5_text_states=byt5_embed,
            byt5_text_mask=byt5_mask,
            rotary_pos_emb_cache=None,
        )

        # --- Flow-matching loss: predict velocity = noise - latents ---
        target = noise - latents
        dataset_weight = batch.get("dataset_weight")
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device=device, dtype=dtype)
            loss = self._compute_masked_loss(
                model_pred.to(dtype), target, loss_mask,
                mask_weight=self._mask_weight, normalize_by_area=self._normalize_masked_loss,
            )
            if dataset_weight is not None:
                loss = loss * dataset_weight.mean()
        else:
            loss = self._compute_loss(model_pred.to(dtype), target, loss_weight=dataset_weight)

        return TrainStepOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "timestep_mean": t.mean().detach(),
            },
        )
