"""SD3 MMDiT transformer blocks.

Two block types:
- JointTransformerBlock: bidirectional text+image attention (SD3.0 and SD3.5)
- SD3SingleTransformerBlock: image-only attention (SD3.5 only)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trainer.arch.sd3.components.layers import (
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    FeedForward,
)


def _apply_qk_norm(
    q: Tensor,
    k: Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    """Apply per-head RMSNorm to Q and K.

    Args:
        q: (B, L, num_heads * head_dim)
        k: (B, L, num_heads * head_dim)

    Returns:
        q, k normalized, reshaped to (B, num_heads, L, head_dim)
    """
    B, L, _ = q.shape
    head_dim = q.shape[-1] // num_heads

    # Reshape to (B, L, heads, head_dim) for per-head norm
    q = q.view(B, L, num_heads, head_dim)
    k = k.view(B, L, num_heads, head_dim)

    q = q_norm(q)
    k = k_norm(k)

    # Transpose to (B, heads, L, head_dim) for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    return q, k


class JointTransformerBlock(nn.Module):
    """Bidirectional image+text attention block for SD3 MMDiT.

    Both image and text streams share a single attention operation:
    1. Each stream is independently normalized and projected to Q, K, V.
    2. Image and text sequences are concatenated along the sequence dim.
    3. A single scaled_dot_product_attention runs over the joint sequence.
    4. Outputs are split back into image and text parts.
    5. Each stream applies its own gated residual and MLP.

    This is the core building block of the MMDiT architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        context_pre_only: bool = False,
    ) -> None:
        """
        Args:
            hidden_size:           Model hidden dimension.
            num_attention_heads:   Number of attention heads.
            context_pre_only:      If True, text stream skips the post-attn MLP
                                   (used for the final joint block in some variants).
        """
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.context_pre_only = context_pre_only

        # --- Image stream ---
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn_img_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_img_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ff_img = FeedForward(hidden_size)

        # --- Text stream ---
        self.norm1_context = AdaLayerNormZero(hidden_size)
        self.attn_ctx_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_ctx_out = nn.Linear(hidden_size, hidden_size, bias=True)
        if not context_pre_only:
            self.ff_context = FeedForward(hidden_size)

        # QK norms (per-head RMSNorm)
        self.norm_q_img = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k_img = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_q_ctx = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k_ctx = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states:         (B, L_img, D) - image token sequence
            encoder_hidden_states: (B, L_txt, D) - text token sequence
            temb:                  (B, D)         - timestep+pooled conditioning

        Returns:
            (hidden_states, encoder_hidden_states) - updated image and text sequences
        """
        # --- Image norm + modulation ---
        img_normed, gate_msa_img, shift_mlp_img, scale_mlp_img, gate_mlp_img = (
            self.norm1(hidden_states, temb)
        )

        # --- Text norm + modulation ---
        ctx_normed, gate_msa_ctx, shift_mlp_ctx, scale_mlp_ctx, gate_mlp_ctx = (
            self.norm1_context(encoder_hidden_states, temb)
        )

        # --- QKV projections ---
        img_q, img_k, img_v = self.attn_img_qkv(img_normed).chunk(3, dim=-1)
        ctx_q, ctx_k, ctx_v = self.attn_ctx_qkv(ctx_normed).chunk(3, dim=-1)

        # --- QK norm + reshape to (B, heads, L, head_dim) ---
        img_q, img_k = _apply_qk_norm(img_q, img_k, self.norm_q_img, self.norm_k_img, self.num_heads)
        ctx_q, ctx_k = _apply_qk_norm(ctx_q, ctx_k, self.norm_q_ctx, self.norm_k_ctx, self.num_heads)

        # Reshape V: (B, L, D) -> (B, heads, L, head_dim)
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        img_v = img_v.view(B, L_img, self.num_heads, self.head_dim).transpose(1, 2)
        ctx_v = ctx_v.view(B, encoder_hidden_states.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # --- Concatenate image + text for joint attention ---
        # (B, heads, L_img + L_txt, head_dim)
        joint_q = torch.cat([img_q, ctx_q], dim=2)
        joint_k = torch.cat([img_k, ctx_k], dim=2)
        joint_v = torch.cat([img_v, ctx_v], dim=2)

        # --- Scaled dot-product attention ---
        attn_out = F.scaled_dot_product_attention(joint_q, joint_k, joint_v)
        # (B, heads, L_total, head_dim) -> (B, L_total, D)
        L_total = attn_out.shape[2]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_total, -1)

        # --- Split back into image and text parts ---
        img_attn_out = attn_out[:, :L_img, :]
        ctx_attn_out = attn_out[:, L_img:, :]

        # --- Image stream: project, gated residual, MLP ---
        img_attn_out = self.attn_img_out(img_attn_out)
        hidden_states = hidden_states + gate_msa_img * img_attn_out

        # Image MLP with norm2 (reuse AdaLayerNorm shift/scale from norm1)
        img_mlp_in = hidden_states
        img_mlp_in = torch.layer_norm(img_mlp_in, [img_mlp_in.shape[-1]], eps=1e-6)
        img_mlp_in = img_mlp_in * (1 + scale_mlp_img) + shift_mlp_img
        hidden_states = hidden_states + gate_mlp_img * self.ff_img(img_mlp_in)

        # --- Text stream: project, gated residual, optional MLP ---
        ctx_attn_out = self.attn_ctx_out(ctx_attn_out)
        encoder_hidden_states = encoder_hidden_states + gate_msa_ctx * ctx_attn_out

        if not self.context_pre_only:
            ctx_mlp_in = encoder_hidden_states
            ctx_mlp_in = torch.layer_norm(ctx_mlp_in, [ctx_mlp_in.shape[-1]], eps=1e-6)
            ctx_mlp_in = ctx_mlp_in * (1 + scale_mlp_ctx) + shift_mlp_ctx
            encoder_hidden_states = encoder_hidden_states + gate_mlp_ctx * self.ff_context(ctx_mlp_in)

        return hidden_states, encoder_hidden_states


class SD3SingleTransformerBlock(nn.Module):
    """Image-only self-attention block for SD3.5 variants.

    Used after the joint blocks to process image tokens without text involvement.
    Simpler than JointTransformerBlock: single stream, single norm, single attention.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.norm = AdaLayerNormZeroSingle(hidden_size)
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ff = FeedForward(hidden_size)

        # QK norms
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (B, L, D) - image token sequence
            temb:          (B, D)    - timestep+pooled conditioning

        Returns:
            (B, L, D) - updated image token sequence
        """
        # Norm + modulation (6 params, same as joint block)
        x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm(hidden_states, temb)

        # QKV
        B, L, D = hidden_states.shape
        q, k, v = self.attn_qkv(x_normed).chunk(3, dim=-1)

        # QK norm + reshape
        q, k = _apply_qk_norm(q, k, self.norm_q, self.norm_k, self.num_heads)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        # Gated attention residual
        hidden_states = hidden_states + gate_msa * self.attn_out(attn_out)

        # MLP with norm + modulation + gating (matches joint block pattern)
        mlp_in = torch.layer_norm(hidden_states, [hidden_states.shape[-1]], eps=1e-6)
        mlp_in = mlp_in * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(mlp_in)

        return hidden_states
