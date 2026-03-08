"""Flux Kontext transformer: DoubleStreamBlock + SingleStreamBlock architecture.

Self-contained port from Musubi_Tuner's flux/flux_models.py (Flux class) with:
- print() replaced with logger calls
- logging.basicConfig() removed
- Dead/commented-out code removed
- torch.concat → torch.cat
- Block swap uses trainer.arch.wan.components.utils.ModelOffloader (same as Flux 2)

The key Kontext-specific feature is the ``control_lengths`` parameter in the
forward pass. Control latents are concatenated with noisy image latents along
the sequence dimension; ``control_lengths`` tells each attention block where the
control portion ends so it can apply per-batch masking (variable-length control).

Differences from Flux2Model (do NOT import from flux_2):
- in_channels = 64 (16-ch latents packed 2×2)
- 3-axis RoPE (axes_dim sums to 128 via [16, 56, 56])
- qkv_bias = True on SelfAttention layers
- vector_in: CLIP-L pooled embedding (768-dim) added to the timestep vec
- guidance_in: guidance scale embedding (distilled dev model)
- Blocks receive control_lengths and perform per-batch variable-length masking
  via SDPA with causal-style masks built per sample.
- DoubleStreamBlock uses per-instance Modulation (not shared across blocks)
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .configs import FluxKontextVariantConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP8 optimization target keys
# ---------------------------------------------------------------------------

FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "mod", "modulation"]


# ---------------------------------------------------------------------------
# RoPE helpers - math unchanged from reference, just cleaned up
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
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
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

    The scale factor 1000.0 is applied internally so callers pass timesteps
    in [0, 1] (the Flux Kontext convention).
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
        return ((x * rrms) * self.scale.float()).to(x_dtype)


class _QKNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = _RMSNorm(head_dim)
        self.key_norm = _RMSNorm(head_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class _MLPEmbedder(nn.Module):
    """Two-layer MLP with SiLU for timestep/guidance/vector conditioning."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
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
    """N-dimensional RoPE positional embedding (3-axis for Flux Kontext)."""

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
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[Any, Any]:
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        if self.is_double:
            return out[:3], out[3:]
        return out[:3], None


class _LastLayer(nn.Module):
    """Final AdaLN-Zero projection with patch_size=1 (Flux uses 1×1 patches)."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # patch_size=1 → output dim = 1*1*out_channels = out_channels
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


class _SelfAttention(nn.Module):
    """QKV self-attention sub-block with optional QKV bias (Kontext uses bias=True)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = _QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=True)


def _attention_with_control(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pe: Tensor,
    control_lengths: list[int] | None = None,
) -> Tensor:
    """Unified attention with RoPE and optional per-batch control masking.

    When ``control_lengths`` is provided and lengths differ across the batch,
    we build per-sample attention masks so that noise tokens do not attend to
    padding control tokens. For a uniform batch (all same control length),
    no masking is needed and we use a single SDPA call.

    Args:
        q, k, v: ``(B, H, L, D)`` query/key/value tensors.
        pe:      ``(B, 1, L, D/2, 2, 2)`` RoPE frequencies.
        control_lengths: List of per-sample control sequence lengths, or None.

    Returns:
        ``(B, L, H*D)`` attention output (heads merged).
    """
    q, k = _apply_rope(q, k, pe)

    # Determine whether we need per-sample masking
    if control_lengths is not None:
        max_cl = max(control_lengths)
        min_cl = min(control_lengths)
        variable_length = max_cl != min_cl
    else:
        variable_length = False

    if variable_length:
        # Build per-sample masks: pad samples with shorter control to max_cl.
        # Noise tokens (indices >= n_target + ctrl) should be masked out for
        # samples that have shorter control sequences (padding tokens exist in
        # the batch because we concatenate noisy + ctrl and pad to max_cl).
        # Simplest correct approach: process each sample individually.
        bsz, num_heads, seq_len, head_dim = q.shape
        # Transpose for SDPA: B H L D
        outputs = []
        for b in range(bsz):
            cl = control_lengths[b]
            total = seq_len - (max_cl - cl)  # effective sequence length for this sample
            q_b = q[b : b + 1, :, :total]         # (1, H, total, D)
            k_b = k[b : b + 1, :, :total]
            v_b = v[b : b + 1, :, :total]
            # Transpose: (1, H, L, D) → (1, L, H, D) for SDPA
            q_b = q_b.transpose(1, 2)
            k_b = k_b.transpose(1, 2)
            v_b = v_b.transpose(1, 2)
            out_b = F.scaled_dot_product_attention(q_b, k_b, v_b)  # (1, total, H, D)
            # Pad back to seq_len with zeros
            if total < seq_len:
                pad = torch.zeros(
                    1, seq_len - total, num_heads, head_dim,
                    device=out_b.device, dtype=out_b.dtype,
                )
                out_b = torch.cat([out_b, pad], dim=1)
            outputs.append(out_b)
        # Stack: (B, L, H, D) → (B, L, H*D)
        x = torch.cat(outputs, dim=0)
        return rearrange(x, "B L H D -> B L (H D)")
    else:
        # Uniform batch: single SDPA call (fast path)
        q = q.transpose(1, 2)  # B H L D → B L H D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "B L H D -> B L (H D)")


# ---------------------------------------------------------------------------
# DoubleStreamBlock
# ---------------------------------------------------------------------------

class DoubleStreamBlock(nn.Module):
    """Parallel image+text double-stream attention block for Flux Kontext.

    Differences vs Flux2's DoubleStreamBlock:
    - Per-instance Modulation (not shared across blocks as in Flux 2)
    - QKV bias = True
    - control_lengths forwarded to attention function
    - Standard GELU MLP (not gated SiLU)
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = True):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = _Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = _SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = _Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = _SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    def _forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        control_lengths: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # Image QKV
        img_modulated = (1 + img_mod1_scale) * self.img_norm1(img) + img_mod1_shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Text QKV
        txt_modulated = (1 + txt_mod1_scale) * self.txt_norm1(txt) + txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        txt_len = txt_q.shape[2]

        # Joint attention (txt || img)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = _attention_with_control(q, k, v, pe, control_lengths=control_lengths)  # (B, L, H*D)

        txt_attn = attn[:, :txt_len]   # (B, L_txt, H*D)
        img_attn = attn[:, txt_len:]   # (B, L_img, H*D)

        # Image residual
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift
        )

        # Text residual
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift
        )

        return img, txt

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        control_lengths: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.training and self._gradient_checkpointing:
            return checkpoint(self._forward, img, txt, vec, pe, control_lengths, use_reentrant=False)
        return self._forward(img, txt, vec, pe, control_lengths)


# ---------------------------------------------------------------------------
# SingleStreamBlock
# ---------------------------------------------------------------------------

class SingleStreamBlock(nn.Module):
    """Merged image+text single-stream attention block for Flux Kontext.

    Uses per-instance Modulation and passes control_lengths through attention.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Fused linear: 3*hidden (qkv) + mlp_hidden (MLP input, unbiased gated)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)

        self.norm = _QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = _Modulation(hidden_size, double=False)

        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    def _forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        control_lengths: list[int] | None = None,
    ) -> Tensor:
        mod, _ = self.modulation(vec)
        mod_shift, mod_scale, mod_gate = mod

        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim],
            dim=-1,
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = _attention_with_control(q, k, v, pe, control_lengths=control_lengths)  # (B, L, H*D)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
        return x + mod_gate * output

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        control_lengths: list[int] | None = None,
    ) -> Tensor:
        if self.training and self._gradient_checkpointing:
            return checkpoint(self._forward, x, vec, pe, control_lengths, use_reentrant=False)
        return self._forward(x, vec, pe, control_lengths)


# ---------------------------------------------------------------------------
# FluxKontextModel - main transformer
# ---------------------------------------------------------------------------

class FluxKontextModel(nn.Module):
    """Flux Kontext dual-stream transformer.

    Accepts concatenated (noisy_latents || control_latents) as the image token
    sequence. The ``control_lengths`` parameter tells the attention mechanism how
    many tokens at the end of each sample's image sequence are control tokens,
    enabling correct attention masking for variable-length batches.

    Args:
        config: ``FluxKontextVariantConfig`` specifying architecture dimensions.
    """

    def __init__(self, config: FluxKontextVariantConfig) -> None:
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
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = _MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = _MLPEmbedder(in_dim=config.vec_in_dim, hidden_dim=self.hidden_size)
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size, bias=True)

        self.use_guidance_embed = config.guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = _MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size, self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                )
                for _ in range(config.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio)
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.final_layer = _LastLayer(self.hidden_size, self.out_channels)

        # Block-swap state (None = disabled)
        self._blocks_to_swap: int | None = None
        self._offloader_double: Any = None
        self._offloader_single: Any = None
        self._num_double_blocks = len(self.double_blocks)
        self._num_single_blocks = len(self.single_blocks)

    # --- Properties ---

    def get_model_type(self) -> str:
        return "flux_kontext"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # --- Gradient checkpointing ---

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.time_in.enable_gradient_checkpointing()
        self.vector_in.enable_gradient_checkpointing()
        if self.use_guidance_embed:
            self.guidance_in.enable_gradient_checkpointing()
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        logger.info(
            "FluxKontext: gradient checkpointing enabled (cpu_offload=%s)", activation_cpu_offloading
        )

    def disable_gradient_checkpointing(self) -> None:
        self.time_in.disable_gradient_checkpointing()
        self.vector_in.disable_gradient_checkpointing()
        if self.use_guidance_embed:
            self.guidance_in.disable_gradient_checkpointing()
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.disable_gradient_checkpointing()
        logger.info("FluxKontext: gradient checkpointing disabled")

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
            # Proportional split (same logic as Flux2Model)
            swap_ratio = n_sgl / n_dbl
            double_to_swap = int(round(num_blocks / (1.0 + swap_ratio / 2.0)))
            single_to_swap = int(round(double_to_swap * swap_ratio))

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
            "FluxKontext: block swap enabled - total=%d, double=%d, single=%d",
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
        """Pre-fetch blocks to device before the forward pass."""
        if not self._blocks_to_swap:
            return
        self._offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self._offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    # --- Forward ---

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        control_lengths: list[int] | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            img:             ``(B, L_img, C)`` - packed image tokens (64 channels).
                             For Kontext, L_img = L_noisy + L_ctrl (concatenated).
            img_ids:         ``(B, L_img, 3)`` - 3-axis image position IDs.
            txt:             ``(B, L_txt, 4096)`` - T5-XXL text embeddings.
            txt_ids:         ``(B, L_txt, 3)`` - text position IDs (all zeros).
            timesteps:       ``(B,)`` - float timesteps in [0, 1].
            y:               ``(B, 768)`` - CLIP-L pooled text embedding.
            guidance:        ``(B,)`` - guidance scale vector (use 1.0 for training).
            control_lengths: List of per-sample control sequence lengths, or None.

        Returns:
            ``(B, L_img, C)`` - predicted velocity field (full sequence, incl. control).
            The strategy must slice ``[:, :L_noisy]`` before computing loss.
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("img and txt must both be 3-D tensors (B, L, D).")

        # --- Input projections ---
        img = self.img_in(img)

        # Timestep + guidance + CLIP conditioning vector
        vec = self.time_in(_timestep_embedding(timesteps, 256))
        if self.use_guidance_embed:
            if guidance is None:
                raise ValueError("guidance is required for guidance-distilled models.")
            vec = vec + self.guidance_in(_timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)

        # --- Position embeddings ---
        # Flux Kontext concatenates txt_ids + img_ids for the joint PE
        ids = torch.cat((txt_ids, img_ids), dim=1)  # (B, L_txt + L_img, 3)
        pe = self.pe_embedder(ids)                    # (B, 1, L_txt+L_img, D/2, 2, 2)

        # --- Double-stream blocks ---
        for block_idx, block in enumerate(self.double_blocks):
            if self._blocks_to_swap:
                self._offloader_double.wait_for_block(block_idx)

            img, txt = block(img, txt, vec, pe, control_lengths=control_lengths)

            if self._blocks_to_swap:
                self._offloader_double.submit_move_blocks_forward(self.double_blocks, block_idx)

        # Merge txt + img for single-stream
        img = torch.cat((txt, img), dim=1)

        # --- Single-stream blocks ---
        for block_idx, block in enumerate(self.single_blocks):
            if self._blocks_to_swap:
                self._offloader_single.wait_for_block(block_idx)

            img = block(img, vec, pe, control_lengths=control_lengths)

            if self._blocks_to_swap:
                self._offloader_single.submit_move_blocks_forward(self.single_blocks, block_idx)

        # Restore to input device if block swap moved it
        img = img.to(vec.device)

        # Discard text tokens, keep image tokens
        img = img[:, txt.shape[1]:]

        return self.final_layer(img, vec)  # (B, L_img, out_channels)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def detect_flux_kontext_weight_dtype(dit_path: str) -> torch.dtype | None:
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


def load_flux_kontext_model(
    config: FluxKontextVariantConfig,
    device: torch.device,
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: torch.device,
    dit_weight_dtype: torch.dtype | None,
    fp8_scaled: bool = False,
) -> FluxKontextModel:
    """Build and load a FluxKontextModel from a safetensors checkpoint.

    Args:
        config:           Variant configuration (dimensions).
        device:           Target CUDA/CPU device.
        dit_path:         Path to ``.safetensors`` weights.
        attn_mode:        Attention backend (currently ignored - uses SDPA internally).
        split_attn:       Whether to use split attention (currently ignored).
        loading_device:   Device to load weights onto (CPU for block-swap).
        dit_weight_dtype: Weight dtype. ``None`` only when ``fp8_scaled=True``.
        fp8_scaled:       Apply FP8 quantisation.

    Returns:
        Loaded ``FluxKontextModel``.
    """
    from accelerate import init_empty_weights

    assert (not fp8_scaled and dit_weight_dtype is not None) or (
        fp8_scaled and dit_weight_dtype is None
    ), "dit_weight_dtype must be None iff fp8_scaled=True"

    with init_empty_weights():
        model = FluxKontextModel(config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info("Loading Flux Kontext DiT from %s on %s", dit_path, loading_device)

    # Re-use Wan's safetensors loader (already ported, handles fp8 + prefix stripping)
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
    logger.info("FluxKontext loaded: %s", info)
    return model
