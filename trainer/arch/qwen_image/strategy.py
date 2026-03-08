"""QwenImage model strategy. Ported from Musubi_Tuner's qwen_image_train_network.py."""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from trainer.arch.base import ModelStrategy, ModelComponents, TrainStepOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype helpers (shared with WanStrategy)
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
# Latent packing helpers
# ---------------------------------------------------------------------------

def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack latents into a flat sequence with 2×2 spatial patchification.

    Handles three input formats:
      - [B, C, 1, H, W] (t2i/edit single frame)
      - [B, C, H, W]    (2D, treated as single frame)
      - [B, L, C, H, W] with L > 1 (layered, multiple frames)

    Output: [B, S, C*4] where S = (num_frames × H/2 × W/2) and C*4 is the
    patchified channel dim (16 latent ch × 4 = 64 = model in_channels).

    The 2×2 pixel-unshuffle packs a 2×2 spatial neighborhood into the channel
    dim, matching the QwenImage model's expected in_channels=64.
    """
    batch_size = latents.shape[0]

    if latents.ndim == 4 or (latents.ndim == 5 and latents.shape[2] == 1):
        # Single frame: [B, C, 1, H, W] or [B, C, H, W]
        if latents.ndim == 5:
            latents = latents.squeeze(2)  # [B, C, H, W]
        num_channels = latents.shape[1]
        height = latents.shape[-2]
        width = latents.shape[-1]
        # Pixel-unshuffle: [B, C, H, W] -> [B, H/2*W/2, C*4]
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    elif latents.ndim == 5:
        # Layered: [B, L, C, H, W]
        num_layers = latents.shape[1]
        num_channels = latents.shape[2]
        height = latents.shape[3]
        width = latents.shape[4]
        # Pixel-unshuffle per layer: [B, L, C, H, W] -> [B, L*H/2*W/2, C*4]
        latents = latents.view(batch_size, num_layers, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        latents = latents.reshape(batch_size, num_layers * (height // 2) * (width // 2), num_channels * 4)
    else:
        raise ValueError(f"pack_latents: expected 4D or 5D tensor, got {latents.ndim}D")

    return latents


def unpack_latents(
    latents: torch.Tensor,
    lat_h: int,
    lat_w: int,
    num_frames: int = 1,
) -> torch.Tensor:
    """Unpack model output [B, S, C*4] back to latent space [B, C, F, H, W].

    lat_h, lat_w are the 2×2-downsampled grid sizes (H/2, W/2).
    For single frame: [B, H/2*W/2, C*4] -> [B, C, 1, H, W]
    For layered: [B, F*H/2*W/2, C*4] -> [B, C, F, H, W]
    """
    b, seq, c_patch = latents.shape
    patch_h = lat_h  # already in patch grid space
    patch_w = lat_w
    num_channels = c_patch // 4  # reverse the 2×2 pixel-unshuffle

    # Reshape to [B, F, ph, pw, C, 2, 2]
    latents = latents.reshape(b, num_frames, patch_h, patch_w, num_channels, 2, 2)
    # Permute to [B, F, C, ph, 2, pw, 2] then reshape to [B, F, C, H, W]
    latents = latents.permute(0, 1, 4, 2, 5, 3, 6)
    latents = latents.reshape(b, num_frames, num_channels, patch_h * 2, patch_w * 2)
    # [B, F, C, H, W] -> [B, C, F, H, W]
    latents = latents.permute(0, 2, 1, 3, 4)
    return latents


# ---------------------------------------------------------------------------
# QwenImageStrategy
# ---------------------------------------------------------------------------

class QwenImageStrategy(ModelStrategy):
    """Strategy for the QwenImage text-to-image / editing / layered architecture.

    Three modes:
        t2i     - text-to-image (default)
        edit    - image editing with one or more control images
        edit-2511 - edit with zero_cond_t enabled (newer variant)
        layered - multi-layer generation with 3D RoPE and additional t-cond

    Config keys used:
        model.base_model_path   - path to DiT safetensors
        model.dtype             - training dtype (bf16, fp16, fp32)
        model.attn_mode         - attention backend
        model.split_attn        - split attention for memory efficiency
        model.quantization      - None or "fp8_scaled"
        model.block_swap_count  - blocks to swap CPU↔GPU
        model.gradient_checkpointing
        model.model_kwargs      - {"mode": "t2i"|"edit"|"edit-2511"|"layered",
                                   "num_layers": 60}
    """

    @property
    def architecture(self) -> str:
        return "qwen_image"

    @property
    def supports_video(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> ModelComponents:
        """Load QwenImageTransformer2DModel from checkpoint, return ModelComponents."""
        from trainer.arch.qwen_image.components.configs import QWEN_IMAGE_CONFIGS
        from trainer.arch.qwen_image.components.model import load_qwen_image_model

        cfg = self.config
        dit_path = cfg.model.base_model_path
        train_dtype = _resolve_dtype(cfg.model.dtype)
        attn_mode = cfg.model.attn_mode
        split_attn = cfg.model.split_attn
        fp8_scaled = cfg.model.quantization == "fp8_scaled"

        # Resolve mode
        mode = cfg.model.model_kwargs.get("mode", "t2i")
        if mode not in QWEN_IMAGE_CONFIGS:
            raise ValueError(
                f"Unknown QwenImage mode '{mode}'. Available: {list(QWEN_IMAGE_CONFIGS)}"
            )
        qwen_config = QWEN_IMAGE_CONFIGS[mode]
        num_layers = cfg.model.model_kwargs.get("num_layers", 60)

        # Determine mode-specific flags
        zero_cond_t = qwen_config.zero_cond_t
        use_additional_t_cond = qwen_config.use_additional_t_cond
        use_layer3d_rope = qwen_config.use_layer3d_rope
        is_layered = qwen_config.mode == "layered"
        is_edit = qwen_config.mode == "edit"

        # Warn if model path and mode look mismatched
        if is_edit and "edit" not in dit_path:
            logger.warning(
                f"Mode is '{mode}' (edit) but model path '{dit_path}' does not contain 'edit'. "
                "Verify you have the correct checkpoint."
            )
        elif not is_edit and not is_layered and "edit" in dit_path:
            logger.warning(
                f"Mode is '{mode}' (t2i) but model path '{dit_path}' contains 'edit'. "
                "Verify you have the correct checkpoint."
            )

        # Weight dtype
        dit_weight_dtype = None if fp8_scaled else train_dtype

        # Device selection
        blocks_to_swap = cfg.model.block_swap_count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loading_device = torch.device("cpu") if blocks_to_swap > 0 else device

        model = load_qwen_image_model(
            device=device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            zero_cond_t=zero_cond_t,
            use_additional_t_cond=use_additional_t_cond,
            use_layer3d_rope=use_layer3d_rope,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=fp8_scaled,
            num_layers=num_layers,
        )

        # Apply quantization before gradient checkpointing (quantization changes layer types)
        self._quantize_model(model, cfg)

        if cfg.model.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        if blocks_to_swap > 0:
            model.enable_block_swap(blocks_to_swap, device, supports_backward=True)
            model.move_to_device_except_swap_blocks(device)

        # Cache all config values needed in the hot-path training_step
        noise_offset_val = cfg.training.noise_offset
        dfs = cfg.training.discrete_flow_shift
        flow_shift = math.exp(dfs) if dfs != 0 else 1.0

        self._blocks_to_swap = blocks_to_swap
        self._device = device
        self._train_dtype = train_dtype
        self._qwen_config = qwen_config
        self._mode = mode
        self._is_edit = is_edit
        self._is_layered = is_layered
        self._use_additional_t_cond = use_additional_t_cond
        self._noise_offset_val = noise_offset_val
        self._flow_shift = flow_shift
        self._gradient_checkpointing = cfg.model.gradient_checkpointing
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
                "qwen_config": qwen_config,
                "mode": mode,
                "is_edit": is_edit,
                "is_layered": is_layered,
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
        """Sample timesteps for QwenImage flow-matching training.

        Returns:
            t:         Float [B] in [min_t, max_t] - interpolation coefficient.
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
    # training_step
    # ------------------------------------------------------------------

    def training_step(
        self,
        components: ModelComponents,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> TrainStepOutput:
        """Flow-matching training step for QwenImage.

        Batch expected keys:
            latents:    [B, C, 1, H, W] (t2i) or [B, C, L, H, W] (layered)
            vl_embed:   list of [S, D] tensors - one per sample (variable length)

        Optional batch keys:
            latents_control_0 ... latents_control_N: edit control images
            img_shapes: pre-computed img_shapes list (or built from latents shape)
            txt_seq_lens: pre-computed text sequence lengths

        Forward flow:
            1. Extract latents and text conditioning.
            2. Pack latents into sequence: [B, H*W, C].
            3. Sample noise and timestep.
            4. Noisy input: (1-t)*latents + t*noise (packed).
            5. Forward through QwenImageTransformer2DModel.
            6. Unpack model output back to spatial layout.
            7. Flow-matching loss: MSE(pred, noise - latents).
        """
        device = self._device
        train_dtype = self._train_dtype

        # --- Extract latents ---
        latents = batch["latents"].to(device=device, dtype=train_dtype)
        # latents: [B, C, F, H, W]  F=1 for t2i/edit, F≥2 for layered

        if self._is_layered:
            if latents.shape[2] < 2:
                raise ValueError(
                    f"Layered mode requires F≥2 latents but got shape {latents.shape}"
                )
            # Permute to [B, L, C, H, W] for per-layer handling
            latents = latents.permute(0, 2, 1, 3, 4)  # B, L, C, H, W

        lat_h = latents.shape[-2]
        lat_w = latents.shape[-1]

        # --- Text conditioning ---
        # vl_embed: list of [S_i, D] per sample
        raw_vl = batch["vl_embed"]
        if isinstance(raw_vl, list):
            vl_list = [t.to(device=device, dtype=train_dtype) for t in raw_vl]
        else:
            vl_list = [t.to(device=device, dtype=train_dtype) for t in raw_vl.unbind(0)]
        txt_seq_lens = [v.shape[0] for v in vl_list]

        # Pad to max length and stack: [B, max_S, D]
        max_s = max(txt_seq_lens)
        vl_embed = torch.stack([
            F.pad(v, (0, 0, 0, max_s - v.shape[0])) for v in vl_list
        ], dim=0)

        # Attention mask when batch > 1 and not using split attention
        bsz = latents.shape[0]
        if bsz > 1 and not self._qwen_config.__dict__.get("split_attn", False):
            vl_mask = torch.zeros(bsz, max_s, dtype=torch.bool, device=device)
            for i, slen in enumerate(txt_seq_lens):
                vl_mask[i, :slen] = True
        else:
            vl_mask = None

        # --- Pack latents into sequence ---
        # For t2i/edit: latents is [B, C, 1, H, W] -> packed [B, H*W, C]
        # For layered: latents is [B, L, C, H, W] -> transpose -> [B, C, L, H, W] -> packed
        if self._is_layered:
            latents_bcfhw = latents.transpose(1, 2)  # B, C, L, H, W
        else:
            latents_bcfhw = latents  # B, C, 1, H, W

        packed = pack_latents(latents_bcfhw)  # B, F*H*W, C
        img_seq_len = packed.shape[1]

        # --- Noise ---
        noise_packed = torch.empty_like(packed).normal_()
        if self._noise_offset_val > 0:
            # NOTE: _apply_noise_offset() from the base class cannot be used here.
            # That method assumes (B, C, ...) layout and produces (B, C, 1, ...) offsets.
            # Packed latents are (B, S, C); we need a (B, 1, C) channel-wise offset.
            offset = torch.randn(bsz, 1, packed.shape[-1], device=device, dtype=train_dtype)
            noise_packed.add_(offset, alpha=self._noise_offset_val)

        # --- Timesteps ---
        t, timesteps = self._sample_timesteps(bsz, device)

        # --- Noisy input ---
        # t: [B] -> [B, 1, 1] for packed [B, S, C]
        t_expanded = t.to(dtype=train_dtype).view(-1, 1, 1)
        noisy_input = (1 - t_expanded) * packed + t_expanded * noise_packed

        # --- Edit: append control latents ---
        control_chunks: list = []
        if self._is_edit:
            i = 0
            while f"latents_control_{i}" in batch:
                lc = batch[f"latents_control_{i}"].to(device=device, dtype=train_dtype)
                control_chunks.append(pack_latents(lc))
                i += 1
            if control_chunks:
                noisy_input = torch.cat([noisy_input, *control_chunks], dim=1)

        elif self._is_layered:
            # First target layer acts as condition in layered mode
            cond_layer = latents[:, 0:1]  # B, 1, C, H, W
            cond_layer_bcfhw = cond_layer.transpose(1, 2)  # B, C, 1, H, W
            noisy_input = torch.cat([noisy_input, pack_latents(cond_layer_bcfhw)], dim=1)

        # --- img_shapes ---
        # Determine from batch or from latent shape
        if "img_shapes" in batch:
            img_shapes = batch["img_shapes"]
        else:
            # Default: 1 frame, lat_h//2 and lat_w//2 (after 2×2 patch)
            base_shape = (1, lat_h // 2, lat_w // 2)
            if self._is_layered:
                n_layers = latents.shape[1]
                img_shapes = [[base_shape] * n_layers]
            elif self._is_edit and control_chunks:
                img_shapes = [[(1, lat_h // 2, lat_w // 2)] + [(1, lat_h // 2, lat_w // 2)] * len(control_chunks)]
            else:
                img_shapes = [[base_shape]]

        # --- is_rgb conditioning for layered mode ---
        is_rgb = (
            torch.zeros(bsz, dtype=torch.long, device=device)
            if self._use_additional_t_cond
            else None
        )

        # --- Timestep scaling (model expects [0, 1]) ---
        model_timestep = timesteps / 1000.0

        # --- Forward pass ---
        noisy_input = noisy_input.to(device=device, dtype=train_dtype)
        vl_embed = vl_embed.to(device=device, dtype=train_dtype)
        if vl_mask is not None:
            vl_mask = vl_mask.to(device)

        if self._gradient_checkpointing:
            noisy_input.requires_grad_(True)
            vl_embed.requires_grad_(True)

        model_pred = components.model(
            hidden_states=noisy_input,
            encoder_hidden_states=vl_embed,
            encoder_hidden_states_mask=vl_mask,
            timestep=model_timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            additional_t_cond=is_rgb,
        )

        # Trim control tokens for edit/layered (keep only target sequence)
        if (self._is_edit or self._is_layered) and model_pred.shape[1] > img_seq_len:
            model_pred = model_pred[:, :img_seq_len]

        # --- Unpack model output from sequence to spatial layout ---
        # model_pred: [B, S, patch_size²*out_channels] = [B, S, 64]
        # Unpack to [B, out_channels=16, F, H, W] for loss comparison
        patch_h = lat_h // 2  # patch grid height (after 2×2 pixel-unshuffle)
        patch_w = lat_w // 2  # patch grid width
        if self._is_layered:
            num_frames = latents.shape[1]
        else:
            num_frames = 1

        model_pred = unpack_latents(model_pred, patch_h, patch_w, num_frames)

        # --- Unpack latents and noise for loss ---
        # latents are in packed space [B, S, 64]; unpack back to [B, C, F, H, W]
        latents_spatial = unpack_latents(packed, patch_h, patch_w, num_frames)
        noise_spatial = unpack_latents(noise_packed, patch_h, patch_w, num_frames)

        # --- Loss ---
        # Flow matching target = noise - latents (velocity field direction)
        target = noise_spatial - latents_spatial
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
