"""Kandinsky 5 model strategy. Ported from Musubi_Tuner's kandinsky5_train_network.py."""
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
# VAE stub (training uses pre-cached latents; real VAE only needed for sampling)
# ---------------------------------------------------------------------------

class _VaeStub:
    """Placeholder VAE that satisfies the trainer's ModelComponents.vae interface."""

    def requires_grad_(self, *_, **__):
        return self

    def eval(self):
        return self

    def to(self, *_, **__):
        return self


# ---------------------------------------------------------------------------
# Kandinsky5Strategy
# ---------------------------------------------------------------------------

class Kandinsky5Strategy(ModelStrategy):
    """Strategy for Kandinsky 5 text-to-video / image-to-video architectures.

    Key design points:
    - Per-sample loop: each sample in the batch is processed independently
      because the model operates on single sequences without batch dimensions.
    - Channels-last format: model expects (F, H, W, C) input tensors.
    - Flow matching with target = noise - latents.
    - Visual conditioning: video models append [latent | cond | mask] channels.
    - Task-based configs: 8 variants in TASK_CONFIGS.
    """

    @property
    def architecture(self) -> str:
        return "kandinsky5"

    @property
    def supports_video(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load DiffusionTransformer3D from checkpoint, return ModelComponents.

        Config fields used:
            model.base_model_path     - path to DiT .safetensors
            model.dtype               - training dtype (bf16, fp16, fp32)
            model.attn_mode           - attention engine (sdpa, flash, etc.)
            model.quantization        - None or "fp8_scaled"
            model.block_swap_count    - number of blocks to swap CPU↔GPU
            model.gradient_checkpointing - enable gradient checkpointing
            model.model_kwargs        - must contain "task" key (e.g. "k5-lite-t2v-5s-sd")
        """
        from trainer.arch.kandinsky5.components.configs import TASK_CONFIGS
        from trainer.arch.kandinsky5.components.model import DiffusionTransformer3D

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)

        # Resolve task config
        task = cfg.model.model_kwargs.get("task", "k5-lite-t2v-5s-sd")
        if task not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown Kandinsky 5 task '{task}'. Available: {list(TASK_CONFIGS)}"
            )
        task_conf = TASK_CONFIGS[task]

        # Determine loading device
        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        # Build model config dict from task params
        dit_p = task_conf.dit_params
        dit_conf: dict = {
            "in_visual_dim": dit_p.in_visual_dim,
            "out_visual_dim": dit_p.out_visual_dim,
            "in_text_dim": dit_p.in_text_dim,
            "in_text_dim2": dit_p.in_text_dim2,
            "time_dim": dit_p.time_dim,
            "patch_size": tuple(dit_p.patch_size),
            "model_dim": dit_p.model_dim,
            "ff_dim": dit_p.ff_dim,
            "num_text_blocks": dit_p.num_text_blocks,
            "num_visual_blocks": dit_p.num_visual_blocks,
            "axes_dims": tuple(dit_p.axes_dims),
            "visual_cond": dit_p.visual_cond,
            "attention_engine": cfg.model.attn_mode,
        }
        if dit_p.instruct_type is not None:
            dit_conf["instruct_type"] = dit_p.instruct_type

        model = DiffusionTransformer3D(**dit_conf)
        model = model.to(loading_device)

        # Load checkpoint (best-effort; skip if path doesn't exist for tests)
        import os
        if dit_path and os.path.isfile(dit_path):
            logger.info(f"Loading Kandinsky5 DiT from {dit_path}")
            try:
                from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
                sd: dict[str, torch.Tensor] = {}
                with MemoryEfficientSafeOpen(dit_path) as f:
                    for key in f.keys():
                        sd[key] = f.get_tensor(key, device=loading_device)
                info = model.load_state_dict(sd, strict=False, assign=True)
                logger.info(f"Kandinsky5 DiT loaded: {info}")
                del sd
            except Exception as exc:
                logger.warning(f"Could not load DiT weights from {dit_path}: {exc}")
        else:
            logger.info(
                "Kandinsky5 DiT: no checkpoint path provided or file not found - "
                "using randomly-initialised weights (test / dry-run mode)."
            )

        # Cast to training dtype
        model = model.to(train_dtype)

        # Apply quantization before gradient checkpointing (quantization changes layer types)
        self._quantize_model(model, cfg)

        # Enable gradient checkpointing
        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        # Enable block swap if requested
        if blocks_to_swap > 0:
            model.enable_block_swap(blocks_to_swap, device, supports_backward=True)
            model.move_to_device_except_swap_blocks(device)
        else:
            model = model.to(device)

        # Attach attention config to model (checked in _build_sparse_params)
        import types
        model.attention = types.SimpleNamespace(**task_conf.attention.__dict__)

        # ------------------------------------------------------------------
        # Cache all hot-path state as self._* attributes so training_step
        # does not touch Pydantic objects.
        # ------------------------------------------------------------------
        patch_size = tuple(dit_p.patch_size)
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]

        self._device = device
        self._train_dtype = train_dtype
        self._task = task
        self._task_conf = task_conf
        self._blocks_to_swap = blocks_to_swap
        self._patch_size = patch_size
        self._patch_volume = patch_volume
        self._scale_factor = tuple(task_conf.scale_factor)
        self._visual_cond = dit_p.visual_cond
        self._attn_type = task_conf.attention.type
        # Attention conf for sparse mask (nabla)
        self._attn_conf = task_conf.attention
        # Sparse mask cache: (T, H//8, W//8, device) -> Tensor
        self._nabla_mask_cache: dict = {}
        # Pre-extract training config
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

        self._mask_weight = cfg.data.mask_weight
        self._normalize_masked_loss = cfg.data.normalize_masked_area_loss

        self._setup_loss_fn(cfg)
        self._setup_loss_weighting(cfg)

        return ModelComponents(
            model=model,
            vae=_VaeStub(),  # type: ignore[arg-type]
            extra={
                "task": task,
                "task_conf": task_conf,
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
        """Sample timesteps for Kandinsky5 flow-matching training.

        Returns:
            t: Float [B] in [min_t, max_t] - interpolation coefficient.
            timesteps: Float [B] scaled to [1, 1001] - passed to the model.
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
    # Sparse attention helpers
    # ------------------------------------------------------------------

    def _build_sparse_params(
        self,
        x: torch.Tensor,
        device: torch.device,
    ) -> dict | None:
        """Build (and cache) nabla sparse-attention params for the current visual grid.

        x is the per-sample input after channels-last reshape: (F, H, W, C).
        Returns None if the task does not use nabla attention.
        """
        attn_conf = self._attn_conf
        if attn_conf.type != "nabla":
            return None

        patch_size = self._patch_size
        if patch_size[0] != 1:
            raise ValueError(
                f"NABLA requires temporal patch size == 1 (got {patch_size[0]})"
            )

        duration, height, width = x.shape[:3]
        T = duration // patch_size[0]
        H = height // patch_size[1]
        W = width // patch_size[2]
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError(
                f"NABLA requires H//patch and W//patch divisible by 8; "
                f"got H={H}, W={W}"
            )

        from trainer.arch.kandinsky5.components.utils import fast_sta_nabla

        sta_key = (T, H // 8, W // 8, device)
        sta_mask = self._nabla_mask_cache.get(sta_key)
        if sta_mask is None:
            sta_mask = fast_sta_nabla(
                T, H // 8, W // 8,
                attn_conf.wT, attn_conf.wH, attn_conf.wW,
                device=device,
            )
            self._nabla_mask_cache[sta_key] = sta_mask

        return {
            "sta_mask": sta_mask.unsqueeze(0).unsqueeze(0),
            "attention_type": attn_conf.type,
            "to_fractal": True,
            "P": attn_conf.P,
            "wT": attn_conf.wT,
            "wW": attn_conf.wW,
            "wH": attn_conf.wH,
            "add_sta": attn_conf.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(attn_conf, "method", "topcdf"),
        }

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------

    def training_step(
        self,
        components: ModelComponents,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> TrainStepOutput:
        """Per-sample flow-matching training step for Kandinsky 5.

        Steps:
        1. Extract latents (B, C, F, H, W) and text conditioning.
        2. Sample noise and timesteps.
        3. Create noisy latents: noisy = (1-t)*latent + t*noise.
        4. Per-sample forward pass through DiffusionTransformer3D.
        5. MSE loss: target = noise - latent.
        """
        device = self._device
        train_dtype = self._train_dtype

        # --- Batch extraction ---
        latents = batch["latents"].to(device=device, dtype=train_dtype)
        # text_embeds: [B, seq, dim]; pooled_embed: [B, dim]
        text_embeds = batch["text_embeds"].to(device=device, dtype=train_dtype)
        pooled_embeds = batch["pooled_embed"].to(device=device, dtype=train_dtype)
        attention_mask = batch.get("attention_mask")

        bsz = latents.shape[0]

        # --- Noise ---
        noise = torch.empty_like(latents).normal_()

        # --- Timesteps ---
        t, timesteps = self._sample_timesteps(bsz, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # --- Noisy latents: (1-t)*x + t*noise ---
        # Expand t to match latents shape: (B,) -> (B, 1, 1, ...) with same ndim.
        t_view = (bsz,) + (1,) * (latents.dim() - 1)
        t_expanded = t.to(dtype=train_dtype).view(*t_view)
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * noise

        # --- Per-sample loop ---
        patch_size = self._patch_size
        scale_factor = self._scale_factor
        model = components.model

        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        for b in range(bsz):
            latent_b = latents[b]           # C, F, H, W  or  C, H, W
            noise_b = noise[b]
            noisy_b = noisy_model_input[b]
            text_b = text_embeds[b]         # seq, dim  or  F*seq, dim
            pooled_b = pooled_embeds[b]     # dim  or  F, dim

            mask_b = None
            if attention_mask is not None:
                mask_b = attention_mask[b].to(device=device)
                if mask_b.dtype != torch.bool:
                    mask_b = mask_b.bool()
                mask_b_flat = mask_b.flatten()
                if mask_b_flat.shape[0] == text_b.shape[0]:
                    text_b = text_b[mask_b_flat]
                elif mask_b_flat.sum().item() == text_b.shape[0]:
                    text_b = text_b
                mask_b = None  # consumed - don't pass again to avoid shape mismatch

            # Determine layout: 4-D = video (C, F, H, W), 3-D = image (C, H, W)
            if latent_b.dim() == 4:
                duration = latent_b.shape[-3]
                height, width = latent_b.shape[-2:]
                # Convert (C, F, H, W) -> (F, H, W, C) for the model
                x = noisy_b.permute(1, 2, 3, 0)

                # Append visual conditioning channels if model expects them
                if self._visual_cond:
                    visual_cond = torch.zeros_like(x)
                    visual_cond_mask = torch.zeros(
                        (*x.shape[:-1], 1), device=device, dtype=train_dtype,
                    )
                    x = torch.cat([x, visual_cond, visual_cond_mask], dim=-1)

                # Expand text embeddings to cover all frames
                if text_b.dim() == 2:
                    text_b = text_b.unsqueeze(0).expand(duration, -1, -1).reshape(-1, text_b.shape[-1])
                if pooled_b.dim() == 1:
                    pooled_b = pooled_b.unsqueeze(0).expand(duration, -1)

            else:
                # Image: (C, H, W) -> (1, H, W, C)
                duration = 1
                height, width = latent_b.shape[-2:]
                x = noisy_b.permute(1, 2, 0).unsqueeze(0)
                if self._visual_cond:
                    visual_cond = torch.zeros_like(x)
                    visual_cond_mask = torch.zeros(
                        (*x.shape[:-1], 1), device=device, dtype=train_dtype,
                    )
                    x = torch.cat([x, visual_cond, visual_cond_mask], dim=-1)
                if pooled_b.dim() == 1:
                    pooled_b = pooled_b.unsqueeze(0)

            sparse_params = self._build_sparse_params(x, device)

            visual_rope_pos = [
                torch.arange(duration, device=device),
                torch.arange(height // patch_size[1], device=device),
                torch.arange(width // patch_size[2], device=device),
            ]
            text_rope_pos = torch.arange(text_b.shape[0], device=device)

            t_b = timesteps[b].to(dtype=train_dtype)
            if t_b.dim() > 0:
                t_b = t_b.flatten()[0]
            t_b = t_b.unsqueeze(0)

            model_pred = model(
                x,
                text_b,
                pooled_b,
                t_b,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=scale_factor,
                sparse_params=sparse_params,
                attention_mask=mask_b,
            )

            # model output: (F, H, W, C) -> (F, C, H, W) to align with target
            model_pred = model_pred.permute(0, 3, 1, 2)

            # Target: flow matching velocity = noise - latent
            target = noise_b - latent_b
            if target.dim() == 4:
                # (C, F, H, W) -> (F, C, H, W)
                target = target.permute(1, 0, 2, 3)
            else:
                target = target.unsqueeze(0)

            preds.append(model_pred)
            targets.append(target)

        # Stack across samples and compute MSE loss
        all_preds = torch.stack(preds, dim=0)    # (B, F, C, H, W)
        all_targets = torch.stack(targets, dim=0)  # (B, F, C, H, W)
        dataset_weight = batch.get("dataset_weight")
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device=device, dtype=train_dtype)
            loss = self._compute_masked_loss(
                all_preds.to(train_dtype), all_targets, loss_mask,
                mask_weight=self._mask_weight, normalize_by_area=self._normalize_masked_loss,
            )
            if dataset_weight is not None:
                loss = loss * dataset_weight.mean()
        else:
            loss = self._compute_loss(all_preds.to(train_dtype), all_targets, loss_weight=dataset_weight)

        return TrainStepOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "timestep_mean": t.mean().detach(),
            },
        )
