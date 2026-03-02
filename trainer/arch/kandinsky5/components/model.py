"""Kandinsky 5 DiT model — TransformerEncoderBlock, TransformerDecoderBlock, DiffusionTransformer3D.

Ported from Musubi_Tuner's kandinsky5/models/dit.py.
Improvements over source:
- Removed logging.basicConfig() — logging is configured centrally.
- Block-swap logging uses logger.info instead of print.
- _maybe_compile / apply_* helpers imported from nn.py (no duplication).
- Removed dead / commented-out code.
- enable_gradient_checkpointing signature aligned with WanModel pattern.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE1D,
    RoPE3D,
    Modulation,
    MultiheadSelfAttentionEnc,
    MultiheadSelfAttentionDec,
    MultiheadCrossAttention,
    FeedForward,
    OutLayer,
    apply_scale_shift_norm,
    apply_gate_sum,
    _maybe_compile,
)
from .utils import fractal_flatten, fractal_unflatten

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Encoder block for text stream (self-attention + feed-forward, AdaLN conditioned)."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        attention_engine: str = "auto",
    ) -> None:
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionEnc(model_dim, head_dim, attention_engine)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        rope: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift)
        out = self.self_attention(out, rope, attention_mask)
        x = apply_gate_sum(x, out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate)
        return x


class TransformerDecoderBlock(nn.Module):
    """Decoder block for visual stream (self-attn + cross-attn + ff, AdaLN conditioned)."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        attention_engine: str = "auto",
    ) -> None:
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionDec(model_dim, head_dim, attention_engine)
        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, attention_engine)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(
        self,
        visual_embed: torch.Tensor,
        text_embed: torch.Tensor,
        time_embed: torch.Tensor,
        rope: torch.Tensor,
        sparse_params: dict | None,
        attention_mask=None,
    ) -> torch.Tensor:
        self_attn_p, cross_attn_p, ff_p = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_p, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift)
        out = self.self_attention(out, rope, sparse_params)
        visual_embed = apply_gate_sum(visual_embed, out, gate)

        shift, scale, gate = torch.chunk(cross_attn_p, 3, dim=-1)
        out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift)
        out = self.cross_attention(out, text_embed, attention_mask)
        visual_embed = apply_gate_sum(visual_embed, out, gate)

        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift)
        out = self.feed_forward(out)
        visual_embed = apply_gate_sum(visual_embed, out, gate)
        return visual_embed


# ---------------------------------------------------------------------------
# DiffusionTransformer3D — the top-level DiT model
# ---------------------------------------------------------------------------

class DiffusionTransformer3D(nn.Module):
    """Kandinsky 5 Diffusion Transformer for 3D (video/image) generation.

    Architecture: interleaved text encoder blocks and visual decoder blocks,
    connected via cross-attention. Operates on channels-last tensors.

    Args:
        in_visual_dim: Latent channel count (input).
        out_visual_dim: Latent channel count (output).
        in_text_dim: Qwen embedding dimension.
        in_text_dim2: CLIP pooled embedding dimension.
        time_dim: Time-embedding projection dimension.
        patch_size: Spatial-temporal patch size (pT, pH, pW).
        model_dim: Internal transformer width.
        ff_dim: Feed-forward hidden size.
        num_text_blocks: Number of encoder blocks.
        num_visual_blocks: Number of decoder blocks.
        axes_dims: Head dimensions for RoPE3D (T, H, W axes).
        visual_cond: Whether model expects visual conditioning channels.
        attention_engine: Attention back-end for all blocks.
        instruct_type: Optional conditioning type ("channel" for image-to-image).
    """

    def __init__(
        self,
        in_visual_dim: int = 4,
        out_visual_dim: int = 4,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 512,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        model_dim: int = 2048,
        ff_dim: int = 5120,
        num_text_blocks: int = 2,
        num_visual_blocks: int = 32,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        visual_cond: bool = False,
        attention_engine: str = "auto",
        instruct_type: str | None = None,
    ) -> None:
        super().__init__()
        self.instruct_type = instruct_type
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond

        # When visual conditioning is active, the input has extra channels:
        # [latent | visual_cond | mask] = 2*C + 1 channels.
        visual_embed_dim = (
            2 * in_visual_dim + 1
            if (visual_cond or instruct_type == "channel")
            else in_visual_dim
        )

        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine)
            for _ in range(num_text_blocks)
        ])

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine)
            for _ in range(num_visual_blocks)
        ])

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

        # Block-swap state (set by enable_block_swap)
        self.blocks_to_swap: int | None = None
        self.offloader_visual = None
        self.offloader_text = None
        self.num_text_blocks = len(self.text_transformer_blocks)
        self.num_visual_blocks = len(self.visual_transformer_blocks)

        # Gradient checkpointing state
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    # ------------------------------------------------------------------
    # Sub-pass helpers (can be torch.compiled separately)
    # ------------------------------------------------------------------

    @_maybe_compile()
    def _before_text_blocks(
        self,
        text_embed: torch.Tensor,
        time: torch.Tensor,
        pooled_text_embed: torch.Tensor,
        x: torch.Tensor,
        text_rope_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(time)
        pooled_time_embed = self.pooled_text_embeddings(pooled_text_embed)

        # Reduce per-frame pooled embeddings to a single modulation vector.
        if pooled_time_embed.dim() > 1 and pooled_time_embed.shape[0] > 1:
            pooled_time_embed = pooled_time_embed.mean(dim=0, keepdim=True)
        time_embed = time_embed + pooled_time_embed

        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        return text_embed, time_embed, text_rope, visual_embed

    @_maybe_compile()
    def _before_visual_blocks(
        self,
        visual_embed: torch.Tensor,
        visual_rope_pos: list[torch.Tensor],
        scale_factor: tuple[float, float, float],
        sparse_params: dict | None,
    ) -> tuple[torch.Tensor, tuple, bool, torch.Tensor]:
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = (sparse_params["to_fractal"] if sparse_params is not None else False)
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape, block_mask=to_fractal)
        return visual_embed, visual_shape, to_fractal, visual_rope

    def _after_blocks(
        self,
        visual_embed: torch.Tensor,
        visual_shape: tuple,
        to_fractal: bool,
        text_embed: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> torch.Tensor:
        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        return self.out_layer(visual_embed, text_embed, time_embed)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        pooled_text_embed: torch.Tensor,
        time: torch.Tensor,
        visual_rope_pos: list[torch.Tensor],
        text_rope_pos: torch.Tensor,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict | None = None,
        attention_mask=None,
    ) -> torch.Tensor:
        text_embed, time_embed, text_rope, visual_embed = self._before_text_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos,
        )

        for block_idx, text_block in enumerate(self.text_transformer_blocks):
            if self.blocks_to_swap and self.offloader_text:
                self.offloader_text.wait_for_block(block_idx)
            if self.training and self.gradient_checkpointing:
                text_embed = checkpoint(
                    text_block, text_embed, time_embed, text_rope, attention_mask,
                    use_reentrant=False,
                )
            else:
                text_embed = text_block(text_embed, time_embed, text_rope, attention_mask)
            if self.blocks_to_swap and self.offloader_text:
                self.offloader_text.submit_move_blocks_forward(self.text_transformer_blocks, block_idx)

        visual_embed, visual_shape, to_fractal, visual_rope = self._before_visual_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params,
        )

        for block_idx, visual_block in enumerate(self.visual_transformer_blocks):
            if self.blocks_to_swap and self.offloader_visual:
                self.offloader_visual.wait_for_block(block_idx)
            if self.training and self.gradient_checkpointing:
                visual_embed = checkpoint(
                    visual_block,
                    visual_embed, text_embed, time_embed, visual_rope, sparse_params, attention_mask,
                    use_reentrant=False,
                )
            else:
                visual_embed = visual_block(
                    visual_embed, text_embed, time_embed, visual_rope, sparse_params, attention_mask,
                )
            if self.blocks_to_swap and self.offloader_visual:
                self.offloader_visual.submit_move_blocks_forward(self.visual_transformer_blocks, block_idx)

        return self._after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        logger.info("Kandinsky5 DiT: gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    # ------------------------------------------------------------------
    # Block swap (CPU ↔ GPU offloading)
    # ------------------------------------------------------------------

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ) -> None:
        """Enable CPU↔GPU block swapping for training on limited VRAM."""
        self.blocks_to_swap = num_blocks
        if num_blocks <= 0:
            return

        try:
            from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
        except ImportError:
            logger.warning(
                "custom_offloading_utils not available; block swap disabled. "
                "Install Musubi_Tuner or run without block_swap_count."
            )
            self.blocks_to_swap = None
            return

        text_to_swap = max(0, min(self.num_text_blocks // 2, num_blocks // 4))
        visual_to_swap = max(1, min(num_blocks - text_to_swap, self.num_visual_blocks - 2))

        if text_to_swap > 0:
            self.offloader_text = ModelOffloader(
                "text",
                self.text_transformer_blocks,
                self.num_text_blocks,
                text_to_swap,
                supports_backward,
                device,
                use_pinned_memory,
            )

        self.offloader_visual = ModelOffloader(
            "visual",
            self.visual_transformer_blocks,
            self.num_visual_blocks,
            visual_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        logger.info(
            f"Kandinsky5 DiT: block swap enabled — text={text_to_swap}, visual={visual_to_swap}."
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        """Move all params except swapped blocks to device (for block-swap setup)."""
        if self.blocks_to_swap:
            save_text = self.text_transformer_blocks
            save_visual = self.visual_transformer_blocks
            self.text_transformer_blocks = None  # type: ignore[assignment]
            self.visual_transformer_blocks = None  # type: ignore[assignment]

        self.to(device)

        if self.blocks_to_swap:
            self.text_transformer_blocks = save_text  # type: ignore[assignment]
            self.visual_transformer_blocks = save_visual  # type: ignore[assignment]

    def prepare_block_swap_before_forward(self) -> None:
        """Must be called before each forward when block swap is active."""
        if not self.blocks_to_swap:
            return
        if self.offloader_text:
            self.offloader_text.prepare_block_devices_before_forward(self.text_transformer_blocks)
        if self.offloader_visual:
            self.offloader_visual.prepare_block_devices_before_forward(self.visual_transformer_blocks)

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap:
            if self.offloader_text:
                self.offloader_text.set_forward_only(True)
            if self.offloader_visual:
                self.offloader_visual.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap:
            if self.offloader_text:
                self.offloader_text.set_forward_only(False)
            if self.offloader_visual:
                self.offloader_visual.set_forward_only(False)
            self.prepare_block_swap_before_forward()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dit(conf: dict) -> DiffusionTransformer3D:
    """Instantiate DiffusionTransformer3D from a config dict."""
    return DiffusionTransformer3D(**conf)
