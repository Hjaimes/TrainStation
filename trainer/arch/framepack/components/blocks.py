"""HunyuanVideo transformer blocks for FramePack.

Self-contained — no imports from any other architecture module.
Ported from Musubi_Tuner/src/musubi_tuner/frame_pack/hunyuan_video_packed.py.

Original code: https://github.com/lllyasviel/FramePack
Original license: Apache-2.0
Portions from HuggingFace diffusers: Apache-2.0

Porting policy applied:
- print() -> logger.info()/logger.warning()
- logging.basicConfig() removed
- Dead/commented-out code removed
- torch.concat -> torch.cat (already correct in source)
- in-place ops and memory-free patterns preserved
"""
from __future__ import annotations

import logging
import math
import numbers
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional attention backends (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from xformers.ops import memory_efficient_attention as xformers_attn_func
    logger.info("xformers attention available for FramePack.")
except Exception:
    xformers_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    logger.info("Flash attention available for FramePack.")
except Exception:
    flash_attn_varlen_func = None
    flash_attn_func = None

try:
    from sageattention import sageattn_varlen, sageattn
    logger.info("SageAttention available for FramePack.")
except Exception:
    sageattn_varlen = None
    sageattn = None


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def _get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in _ACT2CLS:
        return _ACT2CLS[act_fn]()
    raise ValueError(f"Activation '{act_fn}' not in {list(_ACT2CLS)}")


# ---------------------------------------------------------------------------
# Timestep sinusoidal embedding
# ---------------------------------------------------------------------------

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps must be 1D"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


# ---------------------------------------------------------------------------
# Building-block modules
# ---------------------------------------------------------------------------

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", out_dim: int | None = None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = _get_activation(act_fn)
        out_dim_actual = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, out_dim_actual)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class FP32SiLU(nn.Module):
    """SiLU with fp32 cast for numerical stability."""
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states), approximate=self.approximate)


class LinearActivation(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, activation: str = "silu"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = _get_activation(activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(hidden_states))


class PixArtAlphaTextProjection(nn.Module):
    """Projects pooled text embeddings into conditioning space."""
    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        out_features = out_features if out_features is not None else hidden_size
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation: {act_fn}")
        self.linear_2 = nn.Linear(hidden_size, out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)


class FeedForward(nn.Module):
    """Feed-forward layer with configurable activation."""
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        bias: bool = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")
        else:
            raise ValueError(f"Unknown activation_fn: {activation_fn}")

        self.net = nn.ModuleList([act_fn, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out, bias=bias)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Layer norm variants
# ---------------------------------------------------------------------------

class LayerNormFramePack(nn.LayerNorm):
    """LayerNorm that casts output to input dtype."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps).to(x.dtype)


class FP32LayerNormFramePack(nn.LayerNorm):
    """LayerNorm computed in float32 for numerical stability."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNormFramePack(nn.Module):
    """RMS Norm (Zhang et al. 2019). Casts back to input dtype."""
    def __init__(self, dim: int | tuple, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.dim = torch.Size(dim)
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.bias_param = nn.Parameter(torch.zeros(dim)) if (elementwise_affine and bias) else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is None:
            return hidden_states.to(input_dtype)
        return hidden_states.to(input_dtype) * self.weight.to(input_dtype)


# ---------------------------------------------------------------------------
# Adaptive norm layers
# ---------------------------------------------------------------------------

class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with 6-way modulation for double-stream blocks."""
    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNormFramePack(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Adaptive LayerNorm with 3-way modulation for single-stream blocks."""
    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNormFramePack(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    """Continuous AdaLN for output projection norm."""
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNormFramePack(embedding_dim, eps, elementwise_affine, bias)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.unsqueeze(-2)
        emb = self.linear(self.silu(emb))
        scale, shift = emb.chunk(2, dim=-1)
        del emb
        return self.norm(x) * (1 + scale) + shift


class HunyuanVideoAdaNorm(nn.Module):
    """AdaNorm used in token refiner blocks (2-way: gate_msa, gate_mlp)."""
    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=-1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


# ---------------------------------------------------------------------------
# Timestep+guidance conditioning
# ---------------------------------------------------------------------------

class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Combines timestep, guidance scale, and pooled text for conditioning."""
    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        time_guidance_emb = timesteps_emb + guidance_emb
        pooled_projections = self.text_embedder(pooled_projection)
        return time_guidance_emb + pooled_projections


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combines timestep and pooled text (for token refiner)."""
    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        return timesteps_emb + self.text_embedder(pooled_projection)


# ---------------------------------------------------------------------------
# Attention processors
# ---------------------------------------------------------------------------

def apply_rotary_emb_transposed(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query/key tensors. freqs_cis: [B, L, dim]."""
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    del freqs_cis
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    del x_real, x_imag
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    seq_len,
    attn_mode: str | None = None,
    split_attn: bool = False,
) -> torch.Tensor:
    """Dispatch attention based on available backends.

    Shapes: q, k, v: [B, L, H, D] (NHD layout).
    """
    if cu_seqlens_q is None:
        # No variable length — plain batched attention
        if attn_mode == "sageattn" or (attn_mode is None and sageattn is not None):
            return sageattn(q, k, v, tensor_layout="NHD")
        if attn_mode == "flash" or (attn_mode is None and flash_attn_func is not None):
            return flash_attn_func(q, k, v)
        if attn_mode == "xformers" or (attn_mode is None and xformers_attn_func is not None):
            return xformers_attn_func(q, k, v)
        # SDPA fallback: expects [B, H, L, D]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    if split_attn:
        # Per-sample attention with variable sequence lengths
        results = []
        for i in range(q.size(0)):
            sl = seq_len[i]
            if attn_mode == "sageattn" or (attn_mode is None and sageattn is not None):
                x_i = sageattn(q[i:i+1, :sl], k[i:i+1, :sl], v[i:i+1, :sl], tensor_layout="NHD")
            elif attn_mode == "flash" or (attn_mode is None and flash_attn_func is not None):
                x_i = flash_attn_func(q[i:i+1, :sl], k[i:i+1, :sl], v[i:i+1, :sl])
            elif attn_mode == "xformers" or (attn_mode is None and xformers_attn_func is not None):
                x_i = xformers_attn_func(q[i:i+1, :sl], k[i:i+1, :sl], v[i:i+1, :sl])
            else:
                x_i = F.scaled_dot_product_attention(
                    q[i:i+1, :sl].transpose(1, 2),
                    k[i:i+1, :sl].transpose(1, 2),
                    v[i:i+1, :sl].transpose(1, 2),
                ).transpose(1, 2)
            if sl < max_seqlen_q:
                x_i = F.pad(x_i, (0, 0, 0, 0, 0, max_seqlen_q - sl))
            results.append(x_i)
        return torch.cat(results, dim=0)

    # Varlen attention (requires flash or sageattn)
    batch_size = q.shape[0]
    q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])
    k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])
    v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])
    if attn_mode == "sageattn" or (attn_mode is None and sageattn_varlen is not None):
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    elif attn_mode == "flash" or (attn_mode is None and flash_attn_varlen_func is not None):
        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    else:
        raise NotImplementedError(
            "No varlen attention backend available. "
            "Install flash-attn or sageattention, or use --split_attn."
        )
    del q, k, v
    return x.view(batch_size, max_seqlen_q, *x.shape[1:])


class HunyuanAttnProcessorDouble:
    """Attention processor for double-stream (joint image+text) blocks."""

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask,
        image_rotary_emb: torch.Tensor,
        attn_mode: str | None = None,
        split_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len = attention_mask

        # Project image latents
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        del hidden_states

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = apply_rotary_emb_transposed(query, image_rotary_emb)
        key = apply_rotary_emb_transposed(key, image_rotary_emb)
        del image_rotary_emb

        # Project context (text)
        enc_q = attn.add_q_proj(encoder_hidden_states)
        enc_k = attn.add_k_proj(encoder_hidden_states)
        enc_v = attn.add_v_proj(encoder_hidden_states)
        txt_length = encoder_hidden_states.shape[1]
        del encoder_hidden_states

        enc_q = enc_q.unflatten(2, (attn.heads, -1))
        enc_k = enc_k.unflatten(2, (attn.heads, -1))
        enc_v = enc_v.unflatten(2, (attn.heads, -1))

        enc_q = attn.norm_added_q(enc_q)
        enc_k = attn.norm_added_k(enc_k)

        query = torch.cat([query, enc_q], dim=1)
        key = torch.cat([key, enc_k], dim=1)
        value = torch.cat([value, enc_v], dim=1)
        del enc_q, enc_k, enc_v

        out = _attn_varlen_func(
            query, key, value,
            cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len,
            attn_mode=attn_mode, split_attn=split_attn,
        )
        del query, key, value
        out = out.flatten(-2)

        hidden_states, encoder_hidden_states = out[:, :-txt_length], out[:, -txt_length:]
        del out

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class HunyuanAttnProcessorSingle:
    """Attention processor for single-stream blocks (concatenated input)."""

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask,
        image_rotary_emb: torch.Tensor,
        attn_mode: str | None = None,
        split_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len = attention_mask
        txt_length = encoder_hidden_states.shape[1]

        hidden_states_cat = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        del hidden_states, encoder_hidden_states

        query = attn.to_q(hidden_states_cat)
        key = attn.to_k(hidden_states_cat)
        value = attn.to_v(hidden_states_cat)
        del hidden_states_cat

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = torch.cat([
            apply_rotary_emb_transposed(query[:, :-txt_length], image_rotary_emb),
            query[:, -txt_length:],
        ], dim=1)
        key = torch.cat([
            apply_rotary_emb_transposed(key[:, :-txt_length], image_rotary_emb),
            key[:, -txt_length:],
        ], dim=1)
        del image_rotary_emb

        out = _attn_varlen_func(
            query, key, value,
            cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len,
            attn_mode=attn_mode, split_attn=split_attn,
        )
        del query, key, value
        out = out.flatten(-2)

        hidden_states, encoder_hidden_states = out[:, :-txt_length], out[:, -txt_length:]
        return hidden_states, encoder_hidden_states


class AttnProcessorSDP:
    """Fallback SDPA processor (used inside Attention for standard cross-attn)."""
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessorSDP requires PyTorch >= 2.0")

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        del query, key, value
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class Attention(nn.Module):
    """Minimal Attention module compatible with FramePack processors."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        qk_norm: str | None = None,
        added_kv_proj_dim: int | None = None,
        eps: float = 1e-5,
        processor=None,
        out_dim: int | None = None,
        context_pre_only=None,
        pre_only: bool = False,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        if qk_norm is None:
            self.norm_q = self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNormFramePack(dim_head, eps=eps)
            self.norm_k = RMSNormFramePack(dim_head, eps=eps)
        else:
            raise ValueError(f"Unknown qk_norm: {qk_norm}")

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            if context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            else:
                self.add_q_proj = None
        else:
            self.add_q_proj = self.add_k_proj = self.add_v_proj = None

        if not pre_only:
            self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.out_dim, bias=True), nn.Identity()])
        else:
            self.to_out = None

        if context_pre_only is not None and not context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=True)
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            self.norm_added_q = RMSNormFramePack(dim_head, eps=eps)
            self.norm_added_k = RMSNormFramePack(dim_head, eps=eps)
        else:
            self.norm_added_q = self.norm_added_k = None

        self.processor = processor if processor is not None else AttnProcessorSDP()

    def set_processor(self, processor) -> None:
        self.processor = processor

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states=None, attention_mask=None, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)


# ---------------------------------------------------------------------------
# Token refiner (text encoder post-processing)
# ---------------------------------------------------------------------------

class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.norm1 = LayerNormFramePack(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )
        self.norm2 = LayerNormFramePack(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate)
        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None, attention_mask=attention_mask)
        del norm_hidden_states
        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = torch.addcmul(hidden_states, attn_output, gate_msa)
        del attn_output, gate_msa
        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = torch.addcmul(hidden_states, ff_output, gate_mlp)
        del ff_output, gate_mlp
        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList([
            HunyuanVideoIndividualTokenRefinerBlock(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_width_ratio=mlp_width_ratio,
                mlp_drop_rate=mlp_drop_rate,
                attention_bias=attention_bias,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).expand(-1, 1, seq_len, -1)
            mask_2 = mask_1.transpose(2, 3)
            self_attn_mask = (mask_1 & mask_2).bool()
            self_attn_mask[:, :, :, 0] = True
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)
        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    """Text token refiner: embeds + self-attention refines text tokens."""
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        del pooled_projections
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)
        del temb, attention_mask
        return hidden_states


# ---------------------------------------------------------------------------
# Main transformer blocks
# ---------------------------------------------------------------------------

class HunyuanVideoTransformerBlock(nn.Module):
    """Double-stream (joint image+text) transformer block."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
        attn_mode: str | None = None,
        split_attn: bool = False,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanAttnProcessorDouble(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = LayerNormFramePack(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")
        self.norm2_context = LayerNormFramePack(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask=None,
        freqs_cis=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_enc,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            attn_mode=self.attn_mode,
            split_attn=self.split_attn,
        )
        del norm_hidden_states, norm_enc, freqs_cis

        hidden_states = torch.addcmul(hidden_states, attn_output, gate_msa)
        del attn_output, gate_msa
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, context_attn_output, c_gate_msa)
        del context_attn_output, c_gate_msa

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        del shift_mlp, scale_mlp
        norm_enc = torch.addcmul(c_shift_mlp, self.norm2_context(encoder_hidden_states), (1 + c_scale_mlp))
        del c_shift_mlp, c_scale_mlp

        ff_output = self.ff(norm_hidden_states)
        del norm_hidden_states
        context_ff_output = self.ff_context(norm_enc)
        del norm_enc

        hidden_states = torch.addcmul(hidden_states, gate_mlp, ff_output)
        del ff_output, gate_mlp
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, c_gate_mlp, context_ff_output)
        del context_ff_output, c_gate_mlp

        return hidden_states, encoder_hidden_states


class HunyuanVideoSingleTransformerBlock(nn.Module):
    """Single-stream transformer block (concatenated image+text tokens)."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
        attn_mode: str | None = None,
        split_attn: bool = False,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanAttnProcessorSingle(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )
        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        del encoder_hidden_states

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_img, norm_txt = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_img,
            encoder_hidden_states=norm_txt,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            attn_mode=self.attn_mode,
            split_attn=self.split_attn,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        del norm_img, norm_txt, context_attn_output, image_rotary_emb

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        del attn_output, mlp_hidden_states
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states
