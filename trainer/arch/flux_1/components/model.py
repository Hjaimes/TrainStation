"""Flux 1 transformer model: Flux1Transformer.

Dual-stream architecture with:
- 16ch latents packed 2x2 to 64ch input tokens
- T5-XXL (4096-dim) text encoder + CLIP-L pooled (768-dim) conditioning
- 3D RoPE (16, 56, 56) - image-only, no temporal dimension
- 19 Flux1DoubleStreamBlocks + 38 Flux1SingleStreamBlocks
- Per-block AdaLayerNorm modulation (not global shared like Flux 2)
- GEGLU activation
- Optional guidance embedding (dev only)
- Block swap support following the same pattern as Flux 2

Ported and adapted from Musubi_Tuner reference with improvements:
- print() -> logger
- logging.basicConfig() removed
- Block swap uses the same ModelOffloader as Flux 2 / Wan
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .configs import Flux1Config
from .blocks import Flux1DoubleStreamBlock, Flux1SingleStreamBlock
from .embeddings import Flux1RoPE, TimestepEmbedding, GuidanceEmbedding, MLPEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP8 optimization keys
# ---------------------------------------------------------------------------

FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "rope", "time_embed", "guidance_embed", "modulation"]


# ---------------------------------------------------------------------------
# Final layer
# ---------------------------------------------------------------------------

class _LastLayer(nn.Module):
    """Final AdaLN-Zero projection: norm -> scale/shift -> linear."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        org_dtype = x.dtype
        mod = self.adaLN_modulation(vec.float())
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x.float()) + shift
        return self.linear(x).to(org_dtype)


# ---------------------------------------------------------------------------
# Flux1Transformer
# ---------------------------------------------------------------------------

class Flux1Transformer(nn.Module):
    """Flux 1 dual-stream transformer.

    Args:
        config: Flux1Config specifying architecture dimensions.
    """

    def __init__(self, config: Flux1Config) -> None:
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels   # 64
        self.out_channels = config.in_channels  # 64
        self.hidden_size = config.hidden_size   # 3072
        self.num_heads = config.num_attention_heads  # 24

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} must be divisible by "
                f"num_attention_heads {config.num_attention_heads}"
            )

        # Validate that rope_axes sum to head_dim
        head_dim = config.hidden_size // config.num_attention_heads
        if sum(config.rope_axes) != head_dim:
            raise ValueError(
                f"rope_axes sum {sum(config.rope_axes)} != head_dim {head_dim}. "
                f"rope_axes={config.rope_axes}"
            )

        # Input projections
        self.img_in = nn.Linear(config.in_channels, config.hidden_size, bias=True)     # 64 -> 3072
        self.txt_in = nn.Linear(config.context_dim, config.hidden_size, bias=True)     # 4096 -> 3072

        # Timestep and guidance conditioning
        self.time_embed = TimestepEmbedding(config.hidden_size)
        self.pooled_embed = MLPEmbedder(config.pooled_dim, config.hidden_size)  # CLIP-L 768 -> 3072

        self.use_guidance_embed = config.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_embed = GuidanceEmbedding(config.hidden_size)

        # 3D RoPE (image-only, 3 axes)
        self.rope = Flux1RoPE(axes_dim=config.rope_axes)

        # Double-stream blocks (19)
        self.double_blocks = nn.ModuleList([
            Flux1DoubleStreamBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.num_double_blocks)
        ])

        # Single-stream blocks (38)
        self.single_blocks = nn.ModuleList([
            Flux1SingleStreamBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.num_single_blocks)
        ])

        # Final output
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.final_layer = _LastLayer(config.hidden_size, self.out_channels)

        # Block-swap state (None = disabled)
        self._blocks_to_swap: int | None = None
        self._offloader_double: Any = None
        self._offloader_single: Any = None
        self._num_double_blocks = len(self.double_blocks)
        self._num_single_blocks = len(self.single_blocks)

    # --- Properties ---

    def get_model_type(self) -> str:
        return "flux_1"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # --- Gradient checkpointing ---

    def enable_gradient_checkpointing(self) -> None:
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.enable_gradient_checkpointing()
        logger.info("Flux1: gradient checkpointing enabled")

    def disable_gradient_checkpointing(self) -> None:
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.disable_gradient_checkpointing()
        logger.info("Flux1: gradient checkpointing disabled")

    # --- Block swap ---

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ) -> None:
        """Enable CPU<->GPU block swapping for memory-constrained training.

        Distributes swap slots proportionally between double and single blocks.
        """
        from trainer.arch.wan.components.utils import ModelOffloader

        self._blocks_to_swap = num_blocks
        n_dbl = self._num_double_blocks
        n_sgl = self._num_single_blocks

        if num_blocks <= 0:
            double_to_swap = single_to_swap = 0
        elif n_dbl == 0:
            double_to_swap, single_to_swap = 0, num_blocks
        elif n_sgl == 0:
            double_to_swap, single_to_swap = num_blocks, 0
        else:
            swap_ratio = n_sgl / n_dbl
            double_to_swap = int(round(num_blocks / (1.0 + swap_ratio / 2.0)))
            single_to_swap = int(round(double_to_swap * swap_ratio))

            # Prevent swapping too many blocks (keep at least 2 on GPU)
            if n_dbl * 2 < n_sgl:
                while double_to_swap >= 1 and double_to_swap > n_dbl - 2:
                    double_to_swap -= 1
                    single_to_swap += 2
            else:
                while single_to_swap >= 2 and single_to_swap > n_sgl - 2:
                    single_to_swap -= 2
                    double_to_swap += 1

            if double_to_swap == 0 and single_to_swap == 0:
                if n_sgl >= n_dbl:
                    single_to_swap = 1
                else:
                    double_to_swap = 1

        if double_to_swap > n_dbl - 2 or single_to_swap > n_sgl - 2:
            raise ValueError(
                f"Block swap too large: requested double={double_to_swap}, single={single_to_swap}; "
                f"available double<={n_dbl - 2}, single<={n_sgl - 2}"
            )

        self._offloader_double = ModelOffloader(
            "double", self.double_blocks, n_dbl, double_to_swap,
            supports_backward, device, use_pinned_memory,
        )
        self._offloader_single = ModelOffloader(
            "single", self.single_blocks, n_sgl, single_to_swap,
            supports_backward, device, use_pinned_memory,
        )
        logger.info(
            "Flux1: block swap enabled - total=%d, double=%d, single=%d",
            num_blocks, double_to_swap, single_to_swap,
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        """Move all model parameters to device except swap blocks."""
        if self._blocks_to_swap:
            saved_double = self.double_blocks
            saved_single = self.single_blocks
            self.double_blocks = nn.ModuleList()
            self.single_blocks = nn.ModuleList()

        self.to(device)

        if self._blocks_to_swap:
            self.double_blocks = saved_double
            self.single_blocks = saved_single

    def prepare_block_swap_before_forward(self) -> None:
        """Pre-fetch blocks to device before the forward pass."""
        if not self._blocks_to_swap:
            return
        self._offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self._offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    # --- Forward ---

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None = None,
        pooled_text: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for Flux 1 transformer.

        Args:
            x:           (B, HW, 64) - packed image tokens (16ch -> 64ch).
            x_ids:       (B, HW, 3)  - 3D image position IDs [ch, y, x].
            timesteps:   (B,)        - float timesteps in [0, 1].
            ctx:         (B, L, 4096) - T5-XXL text embeddings.
            ctx_ids:     (B, L, 3)   - 3D text position IDs (all zeros).
            guidance:    (B,)        - guidance scale, or None if unused (schnell).
            pooled_text: (B, 768)    - CLIP-L pooled text embedding, or None.

        Returns:
            (B, HW, 64) - predicted velocity field (packed, same shape as x).
        """
        num_txt_tokens = ctx.shape[1]

        # 1. Project image and text to hidden dim
        img = self.img_in(x)    # (B, HW, 3072)
        txt = self.txt_in(ctx)  # (B, L, 3072)

        # 2. Compute conditioning vector: timestep + pooled text [+ guidance]
        vec = self.time_embed(timesteps)  # (B, 3072)
        if pooled_text is not None:
            vec = vec + self.pooled_embed(pooled_text)  # (B, 3072)
        if self.use_guidance_embed and guidance is not None:
            vec = vec + self.guidance_embed(guidance)   # (B, 3072)

        # 3. Compute RoPE for image and text positions
        img_rope = self.rope(x_ids)    # (B, HW, head_dim)
        txt_rope = self.rope(ctx_ids)  # (B, L, head_dim)

        # 4. Double-stream blocks
        for block_idx, block in enumerate(self.double_blocks):
            if self._blocks_to_swap:
                self._offloader_double.wait_for_block(block_idx)

            img, txt = block(img, txt, vec, img_rope, txt_rope)

            if self._blocks_to_swap:
                self._offloader_double.submit_move_blocks_forward(self.double_blocks, block_idx)

        # 5. Concatenate txt + img for single-stream blocks
        combined = torch.cat([txt, img], dim=1)               # (B, L+HW, 3072)
        combined_rope = torch.cat([txt_rope, img_rope], dim=1)  # (B, L+HW, head_dim)

        # 6. Single-stream blocks
        for block_idx, block in enumerate(self.single_blocks):
            if self._blocks_to_swap:
                self._offloader_single.wait_for_block(block_idx)

            combined = block(combined, vec, combined_rope)

            if self._blocks_to_swap:
                self._offloader_single.submit_move_blocks_forward(self.single_blocks, block_idx)

        # Move to vec's device (handles CPU offloading edge case)
        combined = combined.to(vec.device)

        # 7. Extract image tokens (discard text tokens)
        img_out = combined[:, num_txt_tokens:]  # (B, HW, 3072)

        # 8. Final norm + output projection
        return self.final_layer(img_out, vec)   # (B, HW, 64)


# ---------------------------------------------------------------------------
# Weight dtype detection
# ---------------------------------------------------------------------------

def detect_flux1_weight_dtype(path: str) -> torch.dtype | None:
    """Inspect the first tensor in the checkpoint to detect its dtype.

    Returns None if the path doesn't exist or can't be read.
    """
    if not os.path.exists(path):
        return None
    try:
        import safetensors.torch
        with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
            keys = f.keys()
            first_key = next(iter(keys), None)
            if first_key is None:
                return None
            return f.get_tensor(first_key).dtype
    except Exception as exc:
        logger.warning("Could not detect dtype from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_flux1_model(
    config: Flux1Config,
    device: torch.device,
    dit_path: str,
    attn_mode: str,
    split_attn: bool = False,
    loading_device: torch.device | None = None,
    dit_weight_dtype: torch.dtype | None = None,
    fp8_scaled: bool = False,
) -> Flux1Transformer:
    """Build and load a Flux1Transformer from a safetensors checkpoint.

    Args:
        config:           Flux1Config with architecture dimensions.
        device:           Target CUDA/CPU device.
        dit_path:         Path to .safetensors weights.
        attn_mode:        Attention backend (sdpa is used internally).
        split_attn:       Whether to use split attention (reserved for future use).
        loading_device:   Device to load weights onto (CPU for block-swap).
        dit_weight_dtype: Weight dtype. None only when fp8_scaled=True.
        fp8_scaled:       Apply FP8 quantization.

    Returns:
        Loaded Flux1Transformer.
    """
    from accelerate import init_empty_weights

    if loading_device is None:
        loading_device = device

    assert (not fp8_scaled and dit_weight_dtype is not None) or (
        fp8_scaled and dit_weight_dtype is None
    ), "dit_weight_dtype must be None iff fp8_scaled=True"

    with init_empty_weights():
        model = Flux1Transformer(config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info("Loading Flux 1 DiT from %s on %s", dit_path, loading_device)

    # Re-use Wan's load_safetensors_with_lora_and_fp8 - it is arch-agnostic
    from trainer.arch.wan.components.utils import (
        load_safetensors_with_lora_and_fp8,
        apply_fp8_monkey_patch,
    )

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=False,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            logger.info("Moving FP8 weights to %s", loading_device)
            for key in sd:
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info("Flux1 loaded: %s", info)
    return model
