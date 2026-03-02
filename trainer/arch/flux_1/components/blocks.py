"""Flux 1 transformer blocks.

Key differences from Flux 2:
- Per-block modulation: each block has its own _Modulation layers
  (Flux 2 uses global shared modulation computed once per forward pass)
- GEGLU activation in MLP: x*gelu(gate) instead of SiLU-gated
- Block class names MUST be Flux1DoubleStreamBlock and Flux1SingleStreamBlock
  to avoid LoRA targeting collisions with Flux 2's DoubleStreamBlock/SingleStreamBlock
- RoPE apply via rotate_half + cos/sin (compatible with Flux1RoPE output format)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# RoPE application helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: Tensor) -> Tensor:
    """Rotate tensor for RoPE: [-x2, x1] interleave pattern.

    Splits the last dimension in half, negates the second half and swaps:
    [x1, x2] -> [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_flux1(q: Tensor, k: Tensor, rope: Tensor) -> tuple[Tensor, Tensor]:
    """Apply 3D RoPE to query and key tensors.

    Args:
        q: (B, H, N, D) query tensor
        k: (B, H, N, D) key tensor
        rope: (B, N, D) cos/sin concatenated frequencies from Flux1RoPE
              First D//2 is cos, last D//2 is sin (per axis concatenated)

    Returns:
        Tuple of rotated (q, k) tensors.
    """
    # rope: (B, N, D) — need to expand heads dimension
    # Split into cos and sin halves
    cos = rope[..., : rope.shape[-1] // 2]  # (B, N, D//2)
    sin = rope[..., rope.shape[-1] // 2 :]  # (B, N, D//2)

    # Expand for broadcasting with heads: (B, 1, N, D//2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Concatenate to match full head_dim
    cos = torch.cat([cos, cos], dim=-1)  # (B, 1, N, D)
    sin = torch.cat([sin, sin], dim=-1)  # (B, 1, N, D)

    # Apply RoPE rotation: q * cos + rotate_half(q) * sin
    q_rot = q.float() * cos + _rotate_half(q.float()) * sin
    k_rot = k.float() * cos + _rotate_half(k.float()) * sin

    return q_rot.type_as(q), k_rot.type_as(k)


# ---------------------------------------------------------------------------
# Sub-modules shared by both block types
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """RMS normalization with learned scale."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(x_dtype) * self.scale


class _QKNorm(nn.Module):
    """Per-head RMS normalization of Q and K (improves training stability)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = _RMSNorm(head_dim)
        self.key_norm = _RMSNorm(head_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class _GEGLUActivation(nn.Module):
    """GEGLU activation: splits input in half, applies x * gelu(gate).

    Flux 1 uses GEGLU (not SiLU-gated as in Flux 2).
    Input: (..., 2*D), Output: (..., D)
    """

    def forward(self, x: Tensor) -> Tensor:
        x_val, gate = x.chunk(2, dim=-1)
        return x_val * F.gelu(gate)


class _Modulation(nn.Module):
    """Per-block AdaLayerNorm modulation.

    Produces (shift, scale, gate) triplets. For double blocks, produces
    6 values (2 sets for pre-attn and pre-mlp). For single blocks, 3 values.
    """

    def __init__(self, hidden_size: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(hidden_size, self.multiplier * hidden_size, bias=True)

    def forward(self, vec: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None]:
        """Compute modulation from conditioning vector.

        Args:
            vec: (B, hidden_size) conditioning signal (time + guidance + pooled text)

        Returns:
            (mod1, mod2) where each is (shift, scale, gate) tuple.
            mod2 is None for single blocks.
        """
        org_dtype = vec.dtype
        out = self.lin(F.silu(vec.float())).to(org_dtype)
        if out.ndim == 2:
            out = out[:, None, :]  # (B, 1, multiplier*hidden) for broadcasting
        chunks = out.chunk(self.multiplier, dim=-1)
        return chunks[:3], (chunks[3:] if self.is_double else None)


# ---------------------------------------------------------------------------
# Flux1DoubleStreamBlock — joint image+text attention with per-block modulation
# ---------------------------------------------------------------------------

class Flux1DoubleStreamBlock(nn.Module):
    """Parallel image+text double-stream attention block for Flux 1.

    Each block has its own img_mod and txt_mod modulation networks
    (per-block AdaLayerNorm, unlike Flux 2 which shares modulation globally).
    Uses GEGLU activation in the MLP instead of SiLU-gated.

    Architecture per stream:
        1. AdaLN modulation -> norm -> QKV proj
        2. Joint attention (img+txt concatenated) with 3D RoPE
        3. Gated residual add
        4. AdaLN modulation -> norm -> GEGLU MLP
        5. Gated residual add
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Per-block modulation (6 params each: 2 sets of shift/scale/gate)
        self.img_mod = _Modulation(hidden_size, double=True)
        self.txt_mod = _Modulation(hidden_size, double=True)

        # Image stream
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.img_attn_norm = _QKNorm(head_dim)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # GEGLU: Linear(d, 2*4d) -> split -> x * gelu(gate) -> Linear(4d, d)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=True),
            _GEGLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Text stream
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.txt_attn_norm = _QKNorm(head_dim)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=True),
            _GEGLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    def _forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        img_rope: Tensor,
        txt_rope: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Core forward logic (separated for gradient checkpointing)."""
        # Per-block modulation
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2  # type: ignore[misc]
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2  # type: ignore[misc]

        # --- Image QKV ---
        img_normed = (1 + img_mod1_scale) * self.img_norm1(img) + img_mod1_shift
        img_qkv = self.img_attn_qkv(img_normed)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn_norm(img_q, img_k, img_v)

        # --- Text QKV ---
        txt_normed = (1 + txt_mod1_scale) * self.txt_norm1(txt) + txt_mod1_shift
        txt_qkv = self.txt_attn_qkv(txt_normed)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn_norm(txt_q, txt_k, txt_v)

        # --- Apply RoPE ---
        img_q, img_k = _apply_rope_flux1(img_q, img_k, img_rope)
        txt_q, txt_k = _apply_rope_flux1(txt_q, txt_k, txt_rope)

        txt_len = txt_q.shape[2]

        # --- Joint attention (txt first, then img — matches reference) ---
        q = torch.cat([txt_q, img_q], dim=2)  # (B, H, L_txt+L_img, D)
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # SDPA: inputs are (B, H, L, D) — already in correct format
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, H, L, D)
        attn = attn.transpose(1, 2)  # (B, L, H, D)

        txt_attn = attn[:, :txt_len]   # (B, L_txt, H, D)
        img_attn = attn[:, txt_len:]   # (B, L_img, H, D)

        txt_attn_flat = rearrange(txt_attn, "B L H D -> B L (H D)")
        img_attn_flat = rearrange(img_attn, "B L H D -> B L (H D)")

        # --- Image residuals ---
        img = img + img_mod1_gate * self.img_attn_proj(img_attn_flat)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift
        )

        # --- Text residuals ---
        txt = txt + txt_mod1_gate * self.txt_attn_proj(txt_attn_flat)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift
        )

        return img, txt

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        img_rope: Tensor,
        txt_rope: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.training and self._gradient_checkpointing:
            return checkpoint(
                self._forward, img, txt, vec, img_rope, txt_rope, use_reentrant=False
            )
        return self._forward(img, txt, vec, img_rope, txt_rope)


# ---------------------------------------------------------------------------
# Flux1SingleStreamBlock — merged sequence with per-block modulation
# ---------------------------------------------------------------------------

class Flux1SingleStreamBlock(nn.Module):
    """Merged image+text single-stream attention block for Flux 1.

    Operates on the concatenated img+txt sequence. Each block has its own
    modulation network (3 params: shift, scale, gate). Uses GEGLU MLP.

    Architecture:
        1. AdaLN modulation -> norm
        2. Fused linear: QKV (3*hidden) + MLP-input (2*mlp_hidden)
        3. Self-attention with 3D RoPE on combined sequence
        4. Concat attn output + GEGLU(mlp_input) -> output projection
        5. Gated residual add
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Per-block modulation (3 params: shift, scale, gate)
        self.modulation = _Modulation(hidden_size, double=False)

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Fused: 3*hidden (qkv) + 2*mlp_hidden (GEGLU input)
        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim * 2,
            bias=True,
        )
        self.attn_norm = _QKNorm(head_dim)
        self.mlp_act = _GEGLUActivation()

        # Output: concat(attn_out, mlp_out) -> hidden
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)

        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    def _forward(self, x: Tensor, vec: Tensor, rope: Tensor) -> Tensor:
        """Core forward logic."""
        mod, _ = self.modulation(vec)
        mod_shift, mod_scale, mod_gate = mod

        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        # Fused projection
        qkv, mlp_in = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * 2],
            dim=-1,
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.attn_norm(q, k, v)

        # Apply RoPE
        q, k = _apply_rope_flux1(q, k, rope)

        # Self-attention
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, H, L, D)
        attn = attn.transpose(1, 2)  # (B, L, H, D)
        attn_flat = rearrange(attn, "B L H D -> B L (H D)")

        # Fuse attention output with GEGLU MLP output
        output = self.linear2(torch.cat([attn_flat, self.mlp_act(mlp_in)], dim=-1))
        return x + mod_gate * output

    def forward(self, x: Tensor, vec: Tensor, rope: Tensor) -> Tensor:
        if self.training and self._gradient_checkpointing:
            return checkpoint(self._forward, x, vec, rope, use_reentrant=False)
        return self._forward(x, vec, rope)
