"""QwenImage transformer model (QwenImageTransformer2DModel) and loader.

Ported from Musubi_Tuner qwen_image_model.py.

Architecture:
  - 60 dual-stream transformer blocks (joint image + text attention)
  - Qwen2.5-VL text encoder (joint_attention_dim = 3584)
  - 2×2 patch embedding: 16 latent channels → 64 input channels
  - RoPE (scale_rope=True) for all modes; Layer3DRope for layered mode
  - Optional block swapping for VRAM-limited training

Porting improvements:
  - Removed logging.basicConfig()
  - Removed commented-out dead code blocks
  - Replaced print() with logger calls
  - Added type hints on public methods
  - Used consistent torch APIs
"""
from __future__ import annotations

import logging
import math
import numbers
from math import prod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    RMSNorm,
    AdaLayerNormContinuous,
    FeedForward,
    QwenEmbedRope,
    QwenEmbedLayer3DRope,
    QwenTimestepProjEmbeddings,
    apply_rotary_emb_qwen,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joint dual-stream attention
# ---------------------------------------------------------------------------

class QwenDoubleStreamAttention(nn.Module):
    """Joint attention for the dual-stream (image + text) QwenImage architecture.

    Computes QKV for both image and text streams independently, applies QK
    normalization and RoPE, concatenates for joint attention, then splits
    outputs back to the two streams.
    """

    def __init__(
        self,
        query_dim: int,
        added_kv_proj_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: Optional[int] = None,
        eps: float = 1e-5,
        attn_mode: str = "torch",
        split_attn: bool = False,
    ):
        super().__init__()
        out_dim = out_dim if out_dim is not None else query_dim
        self.inner_dim = out_dim
        self.heads = out_dim // dim_head
        self.scale = dim_head ** -0.5
        self.out_dim = out_dim
        self.out_context_dim = query_dim
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # Image stream projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Text stream projections
        self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
        self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
        self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

        # Output projections
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, out_dim, bias=True),
            nn.Identity(),
        ])
        self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        txt_seq_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image stream QKV
        img_q = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        img_k = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        img_v = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        del hidden_states

        # Text stream QKV
        txt_q = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_k = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_v = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        del encoder_hidden_states

        # QK normalization
        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        seq_img = img_q.shape[1]

        # Concatenate for joint attention [img, txt]
        joint_q = torch.cat([img_q, txt_q], dim=1)
        joint_k = torch.cat([img_k, txt_k], dim=1)
        joint_v = torch.cat([img_v, txt_v], dim=1)
        del img_q, txt_q, img_k, txt_k, img_v, txt_v

        org_dtype = joint_q.dtype

        # Build attention mask if not using split attention
        attention_mask = None
        if not self.split_attn and encoder_hidden_states_mask is not None:
            b = encoder_hidden_states_mask.shape[0]
            img_ones = torch.ones(b, seq_img, device=encoder_hidden_states_mask.device, dtype=torch.bool)
            attention_mask = torch.cat([img_ones, encoder_hidden_states_mask.bool()], dim=1)
            attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]

        # Compute attention using SDPA
        # joint_q/k/v: [B, S, H, D] - need [B, H, S, D] for SDPA
        joint_q = joint_q.transpose(1, 2)
        joint_k = joint_k.transpose(1, 2)
        joint_v = joint_v.transpose(1, 2)

        joint_out = F.scaled_dot_product_attention(
            joint_q, joint_k, joint_v,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        del joint_q, joint_k, joint_v
        # [B, H, S, D] -> [B, S, H*D]
        joint_out = joint_out.transpose(1, 2).flatten(2, 3).to(org_dtype)

        img_out = joint_out[:, :seq_img, :]
        txt_out = joint_out[:, seq_img:, :]
        del joint_out

        img_out = self.to_out[0](img_out)
        img_out = self.to_out[1](img_out)
        txt_out = self.to_add_out(txt_out)

        return img_out, txt_out


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class QwenImageTransformerBlock(nn.Module):
    """Dual-stream DiT block for QwenImage.

    Each block runs joint attention over the concatenated image+text sequence,
    then applies separate MLPs to each stream. Adaptive normalization (AdaLN)
    uses time embedding conditioning.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        attn_mode: str = "torch",
        split_attn: bool = False,
        zero_cond_t: bool = False,
    ):
        super().__init__()
        self.zero_cond_t = zero_cond_t

        # Image stream
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            eps=eps,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text stream
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        timestep_zero_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply shift/scale/gate modulation to x.

        Returns modulated x and gate tensor.
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if timestep_zero_index is not None:
            actual_batch = shift.size(0) // 2
            sh_b, sh_e = shift[:actual_batch].unsqueeze(1), shift[actual_batch:].unsqueeze(1)
            sc_b, sc_e = scale[:actual_batch].unsqueeze(1), scale[actual_batch:].unsqueeze(1)
            g_b, g_e = gate[:actual_batch].unsqueeze(1), gate[actual_batch:].unsqueeze(1)
            n = timestep_zero_index
            x_mod = torch.cat([
                x[:, :n] * (1 + sc_b) + sh_b,
                x[:, n:] * (1 + sc_e) + sh_e,
            ], dim=1)
            gate_out = torch.cat([
                g_b.expand(-1, n, -1),
                g_e.expand(-1, x.size(1) - n, -1),
            ], dim=1)
        else:
            x_mod = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            gate_out = gate.unsqueeze(1)

        return x_mod, gate_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        txt_seq_lens: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        timestep_zero_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        # For zero_cond_t: temb is doubled along batch; only use first half for text
        txt_temb = torch.chunk(temb, 2, dim=0)[0] if self.zero_cond_t else temb
        txt_mod_params = self.txt_mod(txt_temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        del img_mod_params, txt_mod_params

        # Pre-attention modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, timestep_zero_index)
        del img_normed, img_mod1

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        del txt_normed, txt_mod1

        # Joint attention
        img_attn_out, txt_attn_out = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            txt_seq_lens=txt_seq_lens,
        )
        del img_modulated, txt_modulated

        # Gated residual
        hidden_states = torch.addcmul(hidden_states, img_gate1, img_attn_out)
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate1, txt_attn_out)
        del img_gate1, img_attn_out, txt_gate1, txt_attn_out

        # Post-attention MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, timestep_zero_index)
        del img_normed2, img_mod2
        hidden_states = torch.addcmul(hidden_states, img_gate2, self.img_mlp(img_modulated2))
        del img_gate2, img_modulated2

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        del txt_normed2, txt_mod2
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, self.txt_mlp(txt_modulated2))
        del txt_gate2, txt_modulated2

        # Clip to avoid fp16 overflow
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clamp(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Full transformer model
# ---------------------------------------------------------------------------

class QwenImageTransformer2DModel(nn.Module):
    """Full QwenImage denoising transformer.

    Architecture constants (fixed across all modes):
        patch_size      = 2
        in_channels     = 64  (16 latent ch × 2×2 patch)
        out_channels    = 16
        num_layers      = 60
        attention_head_dim = 128
        num_attention_heads = 24
        joint_attention_dim = 3584  (Qwen2.5-VL hidden size)
        axes_dims_rope  = (16, 56, 56)

    Mode-specific flags:
        zero_cond_t        - edit-2511 variant
        use_additional_t_cond - layered mode (is_rgb embedding)
        use_layer3d_rope   - layered mode (condition image gets neg frame index)

    Forward signature:
        hidden_states: [B, L, in_channels]    packed latent sequence
        encoder_hidden_states: [B, S, 3584]   text embeddings
        encoder_hidden_states_mask: [B, S] bool or None
        timestep: [B]                          in [0, 1]
        img_shapes: list of list of (F,H,W)   per-image shapes
        txt_seq_lens: list of int
        additional_t_cond: [B] long or None   is_rgb for layered mode
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        attn_mode: str = "torch",
        split_attn: bool = False,
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.zero_cond_t = zero_cond_t

        RopeClass = QwenEmbedLayer3DRope if use_layer3d_rope else QwenEmbedRope
        self.pos_embed = RopeClass(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            use_additional_t_cond=use_additional_t_cond,
        )
        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                attn_mode=attn_mode,
                split_attn=split_attn,
                zero_cond_t=zero_cond_t,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        # Block swap (CPU↔GPU offloading)
        self.blocks_to_swap: Optional[int] = None
        self.offloader = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Block swap helpers
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True
        logger.info("QwenImageTransformer2DModel: gradient checkpointing enabled")

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def enable_block_swap(
        self,
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ) -> None:
        from trainer.arch.wan.components.utils import ModelOffloader  # reuse Wan offloader

        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.transformer_blocks)
        assert blocks_to_swap <= self.num_blocks - 1, (
            f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {blocks_to_swap}."
        )
        self.offloader = ModelOffloader(
            "qwen-image-block",
            self.transformer_blocks,
            self.num_blocks,
            blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        logger.info(
            f"Block swap enabled: {blocks_to_swap}/{self.num_blocks} blocks. "
            f"supports_backward={supports_backward}"
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if self.blocks_to_swap:
            save_blocks = self.transformer_blocks
            self.transformer_blocks = None
        self.to(device)
        if self.blocks_to_swap:
            self.transformer_blocks = save_blocks

    def prepare_block_swap_before_forward(self) -> None:
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(self.transformer_blocks)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_shapes: Optional[List] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_t_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.dtype != torch.bool:
            encoder_hidden_states_mask = encoder_hidden_states_mask.bool()

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        # zero_cond_t: double timestep along batch - second half gets zero timestep
        if self.zero_cond_t:
            if img_shapes is None:
                raise ValueError("`img_shapes` must be provided when zero_cond_t=True.")
            timestep = torch.cat([timestep, timestep * 0], dim=0)

            sample = img_shapes[0]
            if isinstance(sample, (tuple, list)) and len(sample) == 3 and all(isinstance(v, numbers.Integral) for v in sample):
                base_len = int(prod(sample))
            elif isinstance(sample, (tuple, list)) and len(sample) >= 1:
                base = sample[0]
                if not (isinstance(base, (tuple, list)) and len(base) == 3):
                    raise ValueError("Invalid img_shapes entry for zero_cond_t.")
                base_len = int(prod(base))
            else:
                raise ValueError("Invalid img_shapes format for zero_cond_t.")
            timestep_zero_index = base_len
        else:
            timestep_zero_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        # Convert txt_seq_lens to tensor for attention blocks
        txt_seq_lens_t = (
            torch.tensor(txt_seq_lens, device=hidden_states.device)
            if txt_seq_lens is not None
            else None
        )

        input_device = hidden_states.device

        for idx, block in enumerate(self.transformer_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(idx)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    txt_seq_lens_t,
                    attention_kwargs,
                    timestep_zero_index,
                    use_reentrant=False,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    txt_seq_lens=txt_seq_lens_t,
                    joint_attention_kwargs=attention_kwargs,
                    timestep_zero_index=timestep_zero_index,
                )

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.transformer_blocks, idx)

        if input_device != hidden_states.device:
            hidden_states = hidden_states.to(input_device)

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# FP8 optimization keys
# ---------------------------------------------------------------------------

FP8_OPTIMIZATION_TARGET_KEYS = ["transformer_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "time_text_embed"]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_qwen_image_model(
    attn_mode: str,
    split_attn: bool,
    zero_cond_t: bool,
    use_additional_t_cond: bool,
    use_layer3d_rope: bool,
    dtype: Optional[torch.dtype],
    num_layers: int = 60,
) -> QwenImageTransformer2DModel:
    """Instantiate QwenImageTransformer2DModel on empty (meta) device."""
    from accelerate import init_empty_weights

    logger.info(
        f"Creating QwenImageTransformer2DModel: attn={attn_mode}, split_attn={split_attn}, "
        f"zero_cond_t={zero_cond_t}, use_additional_t_cond={use_additional_t_cond}, "
        f"use_layer3d_rope={use_layer3d_rope}, num_layers={num_layers}"
    )
    with init_empty_weights():
        model = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=64,
            out_channels=16,
            num_layers=num_layers,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            guidance_embeds=False,
            axes_dims_rope=(16, 56, 56),
            attn_mode=attn_mode,
            split_attn=split_attn,
            zero_cond_t=zero_cond_t,
            use_additional_t_cond=use_additional_t_cond,
            use_layer3d_rope=use_layer3d_rope,
        )
        if dtype is not None:
            model.to(dtype)
    return model


def load_qwen_image_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    zero_cond_t: bool,
    use_additional_t_cond: bool,
    use_layer3d_rope: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    num_layers: int = 60,
) -> QwenImageTransformer2DModel:
    """Load a QwenImage transformer from a safetensors checkpoint.

    Args:
        device: Target device for optimization/merging.
        dit_path: Path to the safetensors checkpoint.
        attn_mode: Attention backend ("torch", "flash", "xformers", etc.)
        split_attn: Whether to use split attention for memory efficiency.
        zero_cond_t: True for edit-2511 variant.
        use_additional_t_cond: True for layered variant.
        use_layer3d_rope: True for layered variant.
        loading_device: Device to load weights onto initially.
        dit_weight_dtype: Cast weights to this dtype. None iff fp8_scaled=True.
        fp8_scaled: Apply fp8 optimization.
        num_layers: Number of transformer layers (default 60).
    """
    import safetensors.torch as st

    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None), (
        "dit_weight_dtype must be None iff fp8_scaled=True"
    )

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    model = create_qwen_image_model(
        attn_mode, split_attn, zero_cond_t, use_additional_t_cond,
        use_layer3d_rope, dit_weight_dtype, num_layers=num_layers,
    )

    logger.info(f"Loading QwenImage DiT from {dit_path}, loading_device={loading_device}")
    sd = st.load_file(dit_path, device=str(loading_device))

    # Strip "model.diffusion_model." prefix if present (ComfyUI format)
    for key in list(sd.keys()):
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)

    # ComfyUI edit-2511 flag
    if "__index_timestep_zero__" in sd:
        assert zero_cond_t, (
            "Found __index_timestep_zero__ in state_dict but zero_cond_t=False. "
            "Use model_version='edit-2511'."
        )
        sd.pop("__index_timestep_zero__")

    if fp8_scaled:
        try:
            from trainer.arch.wan.components.utils import apply_fp8_monkey_patch
            apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        except ImportError:
            logger.warning("apply_fp8_monkey_patch not available; skipping fp8 monkey patch.")
        if loading_device.type != "cpu":
            for key in sd:
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"QwenImage DiT loaded from {dit_path}: {info}")

    return model
