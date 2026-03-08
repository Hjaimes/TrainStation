"""Flux Kontext model strategy.

Ported from Musubi_Tuner's flux_kontext_train_network.py.

Flux Kontext is an image-editing architecture built on top of the original
FLUX.1 dual-stream transformer. The key concept is that a *control image*
is injected into the denoising process by concatenating its packed latents
with the noisy target latents along the sequence dimension:

    img_input = cat([noisy_target_packed, control_packed], dim=1)  # (B, L_n+L_c, 64)

The model predicts a velocity field for the full (target + control) sequence.
After the forward pass, the control portion is sliced off and only the target
portion is used for the flow-matching loss:

    pred_target = model_pred[:, :L_n]     # slice off control
    loss = MSE(pred_target, noise - target_latents)

``control_lengths`` is a list of per-sample control token counts forwarded
through every attention block so the model can distinguish noise vs. control
tokens within its attention mechanism.
"""
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
# FluxKontextStrategy
# ---------------------------------------------------------------------------

class FluxKontextStrategy(ModelStrategy):
    """Strategy for Flux Kontext image-editing architecture.

    Architecture: dual-stream (DoubleStreamBlock + SingleStreamBlock) Flux transformer
    with control-image injection via sequence concatenation.

    Text encoders: T5-XXL (context) + CLIP-L (pooled vector conditioning).
    These are cached offline; training receives pre-computed embeddings.

    Supports: block swap, gradient checkpointing, FP8 quantisation.
    Image-only (supports_video = False).
    """

    @property
    def architecture(self) -> str:
        return "flux_kontext"

    @property
    def supports_video(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load FluxKontextModel from checkpoint, detect dtype, return ModelComponents.

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
        from trainer.arch.flux_kontext.components.configs import FLUX_KONTEXT_CONFIGS
        from trainer.arch.flux_kontext.components.model import (
            detect_flux_kontext_weight_dtype,
            load_flux_kontext_model,
        )

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"

        # Resolve model variant ("dev" is the only current variant)
        model_version = cfg.model.model_kwargs.get("model_version", "dev")
        if model_version not in FLUX_KONTEXT_CONFIGS:
            raise ValueError(
                f"Unknown Flux Kontext model_version '{model_version}'. "
                f"Available: {list(FLUX_KONTEXT_CONFIGS)}"
            )
        flux_kontext_config = FLUX_KONTEXT_CONFIGS[model_version]

        # Detect weight dtype from checkpoint
        if fp8_scaled:
            dit_weight_dtype = None
        else:
            detected_dtype = detect_flux_kontext_weight_dtype(dit_path)
            dit_weight_dtype = (
                detected_dtype
                if (detected_dtype is not None and train_dtype == detected_dtype)
                else train_dtype
            )

        # Determine loading device (CPU for block swap to save VRAM)
        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        # Load model
        model = load_flux_kontext_model(
            config=flux_kontext_config,
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

        # --- Pre-compute and cache all hot-path constants ---
        noise_offset_val = cfg.training.noise_offset
        dfs = cfg.training.discrete_flow_shift
        flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        # Store strategy state - all read by training_step, never touch Pydantic in hot path
        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._flux_kontext_config = flux_kontext_config
        self._model_version = model_version
        self._noise_offset_val = noise_offset_val
        self._noise_offset_type = cfg.training.noise_offset_type
        self._flow_shift = flow_shift

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
                "flux_kontext_config": flux_kontext_config,
                "model_version": model_version,
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks (block swap)
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
        """Sample timesteps for Flux Kontext flow-matching training.

        Returns:
            t:         Float ``(B,)`` in [min_t, max_t] - interpolation coefficient.
            timesteps: Float ``(B,)`` in [0, 1] - Flux Kontext model receives [0, 1].
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
        # Flux Kontext model receives timesteps in [0, 1] (not scaled to 1000)
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
        """Flow-matching training step for Flux Kontext.

        Batch format:
            latents:         ``(B, 16, H, W)`` - target latents.
            latents_control: ``(B, 16, H_c, W_c)`` - control image latents.
            t5_vec:          ``(B, T, 4096)`` - T5-XXL text embeddings.
            clip_l_pooler:   ``(B, 768)`` - CLIP-L pooled text embedding.

        Pipeline:
        1. Pack target latents: ``(B, 16, H, W)`` → ``(B, L_n, 64)``.
        2. Pack control latents: ``(B, 16, H_c, W_c)`` → ``(B, L_c, 64)``.
        3. Sample noise and timesteps.
        4. Noisy target: ``x_t = (1-t)*target + t*noise``.
        5. Concatenate noisy target + control: ``img_input = cat([x_t, ctrl], dim=1)``.
        6. Build position IDs for full concatenated sequence.
        7. Forward through FluxKontextModel with control_lengths.
        8. Slice prediction: ``pred_target = model_pred[:, :L_n]``.
        9. Unpack and compute MSE loss: target = noise - latents.
        """
        from trainer.arch.flux_kontext.components.utils import (
            pack_latents,
            prepare_img_ids,
            prepare_txt_ids,
        )

        device = self._device
        train_dtype = self._train_dtype

        # --- Extract batch ---
        latents = batch["latents"].to(device=device, dtype=train_dtype)         # (B, 16, H, W)
        control_latents = batch["latents_control"].to(device=device, dtype=train_dtype)  # (B, 16, H_c, W_c)
        t5_vec = batch["t5_vec"].to(device=device, dtype=train_dtype)           # (B, T, 4096)
        clip_l_pooler = batch["clip_l_pooler"].to(device=device, dtype=train_dtype)  # (B, 768)

        bsz = latents.shape[0]

        # --- Noise (in-place to avoid extra allocation) ---
        noise = torch.empty_like(latents).normal_()

        # --- Timesteps ---
        t, timesteps = self._sample_timesteps(bsz, device)

        self._apply_noise_offset(noise, self._noise_offset_val, t=t, offset_type=self._noise_offset_type)

        # --- Noisy target latents ---
        t_expanded = t.to(train_dtype).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise  # (B, 16, H, W)

        # --- Pack latents: (B, 16, H, W) → (B, L, 64) ---
        packed_h = latents.shape[2] // 2
        packed_w = latents.shape[3] // 2
        noisy_packed = pack_latents(noisy_latents)   # (B, L_n, 64)

        packed_ctrl_h = control_latents.shape[2] // 2
        packed_ctrl_w = control_latents.shape[3] // 2
        ctrl_packed = pack_latents(control_latents)  # (B, L_c, 64)

        # --- control_lengths: [L_c, L_c, ...] for each batch sample ---
        # Uniform batch: all samples have the same control length.
        # (Variable-length batches are handled in _attention_with_control.)
        ctrl_seq_len = ctrl_packed.shape[1]
        control_lengths: list[int] = [ctrl_seq_len] * bsz

        # --- Concatenate noisy target + control for the model input ---
        img_input = torch.cat([noisy_packed, ctrl_packed], dim=1)   # (B, L_n+L_c, 64)

        # --- Position IDs ---
        img_ids = prepare_img_ids(bsz, packed_h, packed_w, is_ctrl=False).to(device)       # (B, L_n, 3)
        ctrl_ids = prepare_img_ids(bsz, packed_ctrl_h, packed_ctrl_w, is_ctrl=True).to(device)  # (B, L_c, 3)
        img_input_ids = torch.cat([img_ids, ctrl_ids], dim=1)  # (B, L_n+L_c, 3)

        txt_ids = prepare_txt_ids(bsz, t5_vec.shape[1]).to(device)  # (B, T, 3)

        # Guidance: use 1.0 for LoRA/fine-tune training (not CFG)
        guidance_vec = torch.ones(bsz, device=device, dtype=train_dtype)

        # Gradient checkpointing: ensure inputs require grad
        if self.config.model.gradient_checkpointing:
            noisy_packed = noisy_packed.requires_grad_(True)
            ctrl_packed = ctrl_packed.requires_grad_(True)
            t5_vec = t5_vec.requires_grad_(True)
            clip_l_pooler = clip_l_pooler.requires_grad_(True)
            img_input = torch.cat([noisy_packed, ctrl_packed], dim=1)

        # --- Forward pass ---
        model_pred = components.model(
            img=img_input,
            img_ids=img_input_ids,
            txt=t5_vec,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=clip_l_pooler,
            guidance=guidance_vec,
            control_lengths=control_lengths,
        )
        # model_pred: (B, L_n+L_c, 64) - includes control portion

        # --- Slice prediction: remove control tokens ---
        noisy_seq_len = noisy_packed.shape[1]
        pred_target = model_pred[:, :noisy_seq_len]  # (B, L_n, 64)

        # --- Unpack prediction: (B, L_n, 64) → (B, 16, H, W) ---
        pred_target = rearrange(
            pred_target,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=packed_h, w=packed_w, ph=2, pw=2,
        )

        # --- Flow-matching loss ---
        # Target velocity: v = noise - latents
        target = noise - latents
        dataset_weight = batch.get("dataset_weight")
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device=device, dtype=train_dtype)
            loss = self._compute_masked_loss(
                pred_target.to(train_dtype), target, loss_mask,
                mask_weight=self._mask_weight, normalize_by_area=self._normalize_masked_loss,
            )
            if dataset_weight is not None:
                loss = loss * dataset_weight.mean()
        else:
            loss = self._compute_loss(pred_target.to(train_dtype), target, loss_weight=dataset_weight)

        return TrainStepOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "timestep_mean": t.mean().detach(),
            },
        )
