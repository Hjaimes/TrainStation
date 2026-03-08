"""Flux 2 transformer: DoubleStreamBlock + SingleStreamBlock architecture.

Ported from Musubi_Tuner flux_2/flux2_models.py with the following improvements:
- print() calls replaced with logger.info() / logger.warning()
- logging.basicConfig() removed
- Unused commented-out code removed
- RoPE apply_rope kept as-is (mathematically necessary, no simplification possible)
- Block swap lifecycle follows the same pattern as WanModel in trainer/arch/wan/

This module exposes Flux2Model (the nn.Module) and load_flux2_model().
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .configs import Flux2VariantConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP8 optimization keys (used by load_flux2_model)
# ---------------------------------------------------------------------------

FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "pe_embedder", "time_in", "_modulation"]


# ---------------------------------------------------------------------------
# RoPE helpers (unchanged from Musubi_Tuner - math is correct)
# ---------------------------------------------------------------------------

def _rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """Compute RoPE frequency tensor.

    Args:
        pos: ``(..., N)`` integer position indices.
        dim: Feature dimension (must be even).
        theta: RoPE base frequency.

    Returns:
        Float tensor ``(..., N, dim//2, 2, 2)`` rotation matrices.
    """
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


def _apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

def _timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Sinusoidal timestep embedding.

    Args:
        t: ``(B,)`` float timesteps in [0, 1].
        dim: Embedding dimension.
        max_period: Controls minimum frequency.

    Returns:
        ``(B, dim)`` float embeddings.
    """
    t_scaled = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t_scaled[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(x_dtype) * self.scale


class _QKNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = _RMSNorm(head_dim)
        self.key_norm = _RMSNorm(head_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class _SiLUActivation(nn.Module):
    """Gated SiLU: ``SiLU(x1) * x2`` where ``x`` is split in half."""

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return nn.functional.silu(x1) * x2


class _MLPEmbedder(nn.Module):
    """Two-layer MLP with SiLU for time/guidance embedding."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    def _forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self._gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class _EmbedND(nn.Module):
    """N-dimensional RoPE positional embedding."""

    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, ...]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)


class _Modulation(nn.Module):
    """AdaLN modulation: produces (shift, scale, gate) or doubled variant."""

    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=False)

    def forward(self, vec: Tensor) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None]:
        org_dtype = vec.dtype
        # Cast to fp32 for numerical stability, then restore.
        out = self.lin(nn.functional.silu(vec.float())).to(org_dtype)
        if out.ndim == 2:
            out = out[:, None, :]
        chunks = out.chunk(self.multiplier, dim=-1)
        return chunks[:3], (chunks[3:] if self.is_double else None)


class _LastLayer(nn.Module):
    """Final AdaLN-Zero projection."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
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


class _SelfAttention(nn.Module):
    """QKV self-attention sub-block (no output projection here - done in block)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = _QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


def _attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """Unified attention with RoPE application and SDPA backend."""
    q, k = _apply_rope(q, k, pe)
    # Reshape: B H L D → B L H D for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    x = nn.functional.scaled_dot_product_attention(q, k, v)
    return x  # (B, L, H, D)


# ---------------------------------------------------------------------------
# DoubleStreamBlock
# ---------------------------------------------------------------------------

class DoubleStreamBlock(nn.Module):
    """Parallel image+text double-stream attention block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = _SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 2× expansion for gated SiLU (output dim is mlp_hidden_dim)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            _SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = _SelfAttention(dim=hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            _SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self._gradient_checkpointing = False
        self._activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self._gradient_checkpointing = True
        self._activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False
        self._activation_cpu_offloading = False

    def _forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe_img: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
        mod_txt: tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # --- image QKV ---
        img_modulated = (1 + img_mod1_scale) * self.img_norm1(img) + img_mod1_shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # --- text QKV ---
        txt_modulated = (1 + txt_mod1_scale) * self.txt_norm1(txt) + txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        txt_len = txt_q.shape[2]

        # --- joint attention ---
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        pe = torch.cat((pe_ctx, pe_img), dim=2)

        attn = _attention(q, k, v, pe)  # (B, L_txt+L_img, H, D)

        txt_attn = attn[:, :txt_len]    # (B, L_txt, H, D)
        img_attn = attn[:, txt_len:]    # (B, L_img, H, D)

        # Reshape H×D → hidden_size for projection
        txt_attn_flat = rearrange(txt_attn, "B L H D -> B L (H D)")
        img_attn_flat = rearrange(img_attn, "B L H D -> B L (H D)")

        # --- image residual ---
        img = img + img_mod1_gate * self.img_attn.proj(img_attn_flat)
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift)

        # --- text residual ---
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn_flat)
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift)

        return img, txt

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe_img: Tensor,
        pe_ctx: Tensor,
        mod_img: Any,
        mod_txt: Any,
    ) -> tuple[Tensor, Tensor]:
        if self.training and self._gradient_checkpointing:
            return checkpoint(self._forward, img, txt, pe_img, pe_ctx, mod_img, mod_txt, use_reentrant=False)
        return self._forward(img, txt, pe_img, pe_ctx, mod_img, mod_txt)


# ---------------------------------------------------------------------------
# SingleStreamBlock
# ---------------------------------------------------------------------------

class SingleStreamBlock(nn.Module):
    """Merged image+text single-stream attention block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Fused linear: 3*hidden (qkv) + 2*mlp_hidden (gated MLP input)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = _QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = _SiLUActivation()

        self._gradient_checkpointing = False
        self._activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self._gradient_checkpointing = True
        self._activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False
        self._activation_cpu_offloading = False

    def _forward(
        self,
        x: Tensor,
        pe: Tensor,
        mod: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod

        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * 2],
            dim=-1,
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = _attention(q, k, v, pe)                     # (B, L, H, D)
        attn_flat = rearrange(attn, "B L H D -> B L (H D)")

        output = self.linear2(torch.cat((attn_flat, self.mlp_act(mlp)), dim=-1))
        return x + mod_gate * output

    def forward(self, x: Tensor, pe: Tensor, mod: Any) -> Tensor:
        if self.training and self._gradient_checkpointing:
            return checkpoint(self._forward, x, pe, mod, use_reentrant=False)
        return self._forward(x, pe, mod)


# ---------------------------------------------------------------------------
# Flux2Model - main transformer
# ---------------------------------------------------------------------------

class Flux2Model(nn.Module):
    """Flux 2 dual-stream transformer.

    Args:
        config: ``Flux2VariantConfig`` specifying architecture dimensions.
    """

    def __init__(self, config: Flux2VariantConfig) -> None:
        super().__init__()

        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"axes_dim sum {sum(config.axes_dim)} != pe_dim {pe_dim}"
            )

        self.pe_embedder = _EmbedND(dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = _MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = config.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = _MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio)
                for _ in range(config.depth_single_blocks)
            ]
        )

        # Shared modulation networks (one call per forward, shared across blocks)
        self.double_stream_modulation_img = _Modulation(self.hidden_size, double=True)
        self.double_stream_modulation_txt = _Modulation(self.hidden_size, double=True)
        self.single_stream_modulation = _Modulation(self.hidden_size, double=False)

        self.final_layer = _LastLayer(self.hidden_size, self.out_channels)

        # Block-swap state (None = disabled)
        self._blocks_to_swap: int | None = None
        self._offloader_double: Any = None
        self._offloader_single: Any = None
        self._num_double_blocks = len(self.double_blocks)
        self._num_single_blocks = len(self.single_blocks)

    # --- Properties ---

    def get_model_type(self) -> str:
        return "flux_2"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # --- Gradient checkpointing ---

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.time_in.enable_gradient_checkpointing()
        if self.use_guidance_embed:
            self.guidance_in.enable_gradient_checkpointing()
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        logger.info(
            "Flux2: gradient checkpointing enabled (cpu_offload=%s)", activation_cpu_offloading
        )

    def disable_gradient_checkpointing(self) -> None:
        self.time_in.disable_gradient_checkpointing()
        if self.use_guidance_embed:
            self.guidance_in.disable_gradient_checkpointing()
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.disable_gradient_checkpointing()
        logger.info("Flux2: gradient checkpointing disabled")

    # --- Block swap ---

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ) -> None:
        """Enable CPU↔GPU block swapping for memory-constrained training."""
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

            # Prevent swapping too many blocks (keep ≥2 on GPU)
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
                f"available double≤{n_dbl - 2}, single≤{n_sgl - 2}"
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
            "Flux2: block swap enabled - total=%d, double=%d, single=%d",
            num_blocks, double_to_swap, single_to_swap,
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        """Move all model parameters to ``device`` except the blocks being swapped."""
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
        """Pre-fetch blocks to device before the forward pass (block-swap protocol)."""
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
        guidance: Tensor | None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x:         ``(B, HW, C)`` - packed image tokens (128 channels).
            x_ids:     ``(B, HW, 4)`` - image position IDs.
            timesteps: ``(B,)`` - float timesteps in [0, 1].
            ctx:       ``(B, L, D)`` - text embeddings.
            ctx_ids:   ``(B, L, 4)`` - text position IDs.
            guidance:  ``(B,)`` - guidance scale vector, or ``None`` if unused.

        Returns:
            ``(B, HW, C)`` - predicted velocity field (packed).
        """
        num_txt_tokens = ctx.shape[1]

        # Timestep + guidance conditioning vector
        vec = self.time_in(_timestep_embedding(timesteps, 256))
        if self.use_guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(_timestep_embedding(guidance, 256))

        # Compute shared modulation signals once (shared across all blocks of same type)
        double_mod_img, _ = self.double_stream_modulation_img(vec)
        double_mod_txt, _ = self.double_stream_modulation_txt(vec)
        single_mod, _ = self.single_stream_modulation(vec)

        # Project inputs to hidden dim
        img = self.img_in(x)
        txt = self.txt_in(ctx)

        # Positional embeddings
        pe_img = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        # --- Double-stream blocks ---
        for block_idx, block in enumerate(self.double_blocks):
            if self._blocks_to_swap:
                self._offloader_double.wait_for_block(block_idx)

            img, txt = block(img, txt, pe_img, pe_ctx, double_mod_img, double_mod_txt)

            if self._blocks_to_swap:
                self._offloader_double.submit_move_blocks_forward(self.double_blocks, block_idx)

        # Merge text + image for single-stream
        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_img), dim=2)

        # --- Single-stream blocks ---
        for block_idx, block in enumerate(self.single_blocks):
            if self._blocks_to_swap:
                self._offloader_single.wait_for_block(block_idx)

            img = block(img, pe, single_mod)

            if self._blocks_to_swap:
                self._offloader_single.submit_move_blocks_forward(self.single_blocks, block_idx)

        # Move to vec's device (handles CPU offloading edge case)
        img = img.to(vec.device)

        # Discard text tokens, keep image tokens
        img = img[:, num_txt_tokens:]

        return self.final_layer(img, vec)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def detect_flux2_weight_dtype(dit_path: str) -> torch.dtype | None:
    """Inspect the first tensor in the checkpoint to detect its dtype.

    Returns ``None`` if the path doesn't exist or can't be read.
    """
    import os
    if not os.path.exists(dit_path):
        return None
    try:
        import safetensors.torch
        with safetensors.torch.safe_open(dit_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            first_key = next(iter(keys), None)
            if first_key is None:
                return None
            return f.get_tensor(first_key).dtype
    except Exception as exc:
        logger.warning("Could not detect dtype from %s: %s", dit_path, exc)
        return None


def load_flux2_model(
    config: Flux2VariantConfig,
    device: torch.device,
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: torch.device,
    dit_weight_dtype: torch.dtype | None,
    fp8_scaled: bool = False,
) -> Flux2Model:
    """Build and load a Flux2Model from a safetensors checkpoint.

    Args:
        config:           Variant configuration (dimensions).
        device:           Target CUDA/CPU device.
        dit_path:         Path to ``.safetensors`` weights.
        attn_mode:        Attention backend (currently only SDPA is used internally).
        split_attn:       Whether to use split attention (passed through for future use).
        loading_device:   Device to load weights onto (CPU for block-swap).
        dit_weight_dtype: Weight dtype. ``None`` only when ``fp8_scaled=True``.
        fp8_scaled:       Apply FP8 quantisation.

    Returns:
        Loaded ``Flux2Model``.
    """
    from accelerate import init_empty_weights

    assert (not fp8_scaled and dit_weight_dtype is not None) or (
        fp8_scaled and dit_weight_dtype is None
    ), "dit_weight_dtype must be None iff fp8_scaled=True"

    with init_empty_weights():
        model = Flux2Model(config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info("Loading Flux 2 DiT from %s on %s", dit_path, loading_device)

    # Re-use Wan's load_safetensors_with_lora_and_fp8 since it's already ported.
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
    logger.info("Flux2 loaded: %s", info)
    return model
