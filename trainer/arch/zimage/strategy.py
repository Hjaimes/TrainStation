"""Z-Image model strategy. Ported from Musubi_Tuner's zimage_train_network.py.

Key differences from Wan:
  - Image only: latents are 4D [B, C, H, W], frame dim F=1 added on-the-fly.
  - Reversed timestep: model receives (1000 - t) / 1000 (NOT raw t/1000).
  - Flow matching target: latents - noise (sign-reversed vs. Wan's noise - latents).
  - Text encoder: Qwen3-4B embeddings (cached; not encoded during training).
  - Sequence padding: (image_seq + text_seq) must be a multiple of SEQ_MULTI_OF=32.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from trainer.arch.base import ModelStrategy, ModelComponents, TrainStepOutput
from trainer.arch.zimage.components.configs import SEQ_MULTI_OF

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
# ZImageStrategy
# ---------------------------------------------------------------------------

class ZImageStrategy(ModelStrategy):
    """Strategy for the Z-Image image generation architecture.

    Architecture: ZImageTransformer2DModel + AutoencoderKL (16ch) + Qwen3-4B TE.
    Training: flow matching with reversed timestep schedule.
    """

    @property
    def architecture(self) -> str:
        return "zimage"

    @property
    def supports_video(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load ZImageTransformer2DModel from checkpoint, return ModelComponents.

        Config fields used:
            model.base_model_path   - path to transformer .safetensors
            model.dtype             - training dtype (bf16, fp16, fp32)
            model.attn_mode         - attention backend
            model.split_attn        - split attention (no attn mask)
            model.quantization      - None or "fp8_scaled"
            model.block_swap_count  - CPU↔GPU block swapping
            model.gradient_checkpointing - enable gradient checkpointing
            model.model_kwargs      - extra kwargs, e.g. {"use_16bit_attn": True}
        """
        from trainer.arch.zimage.components.configs import ZIMAGE_DEFAULT_CONFIG
        from trainer.arch.zimage.components.model import (
            detect_zimage_sd_dtype,
            load_zimage_model,
        )

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"
        use_16bit_attn = cfg.model.model_kwargs.get("use_16bit_attn", True)

        zimage_config = ZIMAGE_DEFAULT_CONFIG

        # Determine weight dtype
        if fp8_scaled:
            dit_weight_dtype = None
        else:
            detected_dtype = detect_zimage_sd_dtype(dit_path)
            dit_weight_dtype = (
                detected_dtype if train_dtype == detected_dtype else train_dtype
            )

        # Determine loading device
        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        # Load model
        model = load_zimage_model(
            config=zimage_config,
            device=device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=fp8_scaled,
            use_16bit_for_attention=use_16bit_attn,
        )

        # Apply quantization before gradient checkpointing (quantization changes layer types)
        self._quantize_model(model, cfg)

        # Enable gradient checkpointing
        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Enable block swap
        if blocks_to_swap > 0:
            model.enable_block_swap(blocks_to_swap, device, supports_backward=True)
            model.move_to_device_except_swap_blocks(device)

        # Pre-compute patch size (spatial patch, always 2 for Z-Image)
        patch_size = zimage_config.patch_size[0]   # int: 2

        # Pre-compute flow shift constant
        dfs = cfg.training.discrete_flow_shift
        flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        # Cache all config values accessed in training_step()
        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._zimage_config = zimage_config
        self._patch_size = patch_size
        self._split_attn = split_attn
        self._noise_offset_val = cfg.training.noise_offset
        self._noise_offset_type = cfg.training.noise_offset_type
        self._use_gradient_checkpointing = cfg.model.gradient_checkpointing
        self._flow_shift = flow_shift
        # Timestep config
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
                "zimage_config": zimage_config,
                "patch_size": patch_size,
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
        """Sample timesteps for Z-Image flow matching training.

        Returns:
            t:         Float [B] in [min_t, max_t] - interpolation coefficient.
            timesteps: Float [B] in [1, 1001] - Z-Image reversed timestep: model
                       receives (1000 - timesteps) / 1000.
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
        """Flow matching training step for Z-Image.

        Steps:
          1. Extract latents [B, C, H, W] and add dummy frame dim → [B, C, 1, H, W].
          2. Prepare Qwen3 embeddings (already cached in batch).
          3. Sample noise and timesteps.
          4. Create noisy latents: noisy = (1-t) * latents + t * noise.
          5. Apply reversed timestep: t_model = (1000 - raw_timestep) / 1000.
          6. Forward pass: model(noisy, t_model, cap_feats, cap_mask).
          7. Squeeze frame dim, compute MSE: target = latents - noise.
        """
        device = self._device
        train_dtype = self._train_dtype

        # --- Extract batch ---
        latents = batch["latents"].to(device=device, dtype=train_dtype)

        # Z-Image latents are [B, C, H, W] - add frame dim for the model
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)  # [B, C, 1, H, W]
        # If already 5D (F=1 from cache), pass through as-is

        bsz = latents.shape[0]

        # --- Prepare text embeddings ---
        # Batch provides "llm_embed": list of per-sample tensors [L, D]
        # We need to pad/stack to [B, max_L, D] and build cap_mask if not split_attn
        raw_embeds = batch["llm_embed"]
        if isinstance(raw_embeds, torch.Tensor):
            # Already stacked [B, L, D]
            cap_feats = raw_embeds.to(device=device, dtype=train_dtype)
            cap_mask = batch.get("llm_mask")
            if cap_mask is not None:
                cap_mask = cap_mask.to(device=device)
        else:
            # List of [L_i, D] tensors (variable length)
            txt_seq_lens = [x.shape[0] for x in raw_embeds]
            max_len = max(txt_seq_lens)

            # Pad total seq len to multiple of SEQ_MULTI_OF (only needed without split_attn)
            if not self._split_attn and bsz > 1:
                # Compute spatial seq len from latent shape
                H, W = latents.shape[3], latents.shape[4]
                image_seq_len = (H // self._patch_size) * (W // self._patch_size)
                total = image_seq_len + max_len
                padded_total = math.ceil(total / SEQ_MULTI_OF) * SEQ_MULTI_OF
                max_len = int(padded_total) - image_seq_len

            cap_feats_list = [
                F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in raw_embeds
            ]
            cap_feats = torch.stack(cap_feats_list, dim=0)  # [B, max_L, D]
            cap_feats = cap_feats.to(device=device, dtype=train_dtype)

            if not self._split_attn and bsz > 1:
                cap_mask = torch.zeros(
                    bsz, max_len, dtype=torch.bool, device=device,
                )
                for i, length in enumerate(txt_seq_lens):
                    cap_mask[i, :length] = True
            else:
                cap_mask = None

        # --- Noise ---
        noise = torch.empty_like(latents).normal_()

        # --- Timesteps ---
        t, timesteps = self._sample_timesteps(bsz, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # --- Noisy latents ---
        t_expanded = t.to(dtype=train_dtype).view(-1, 1, 1, 1, 1)
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * noise

        # --- Reversed timestep (Z-Image specific) ---
        # Model expects t in [0, 1] where 0 = clean, 1 = noise - opposite of standard
        t_model = ((1000.0 - timesteps) / 1000.0).to(dtype=train_dtype)

        # --- Forward pass ---
        model_pred = components.model(
            x=noisy_model_input,
            t=t_model,
            cap_feats=cap_feats,
            cap_mask=cap_mask,
        )
        # model_pred: [B, C, F, H, W] - squeeze frame dim
        model_pred = model_pred.squeeze(2)           # [B, C, H, W]

        # --- Loss ---
        # Z-Image target = latents - noise (sign-reversed relative to Wan)
        latents_2d = latents.squeeze(2)              # [B, C, H, W]
        noise_2d = noise.squeeze(2)                  # [B, C, H, W]
        target = latents_2d - noise_2d
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
