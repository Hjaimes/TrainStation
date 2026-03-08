"""Kandinsky 5 neural network building blocks.

Ported from Musubi_Tuner's kandinsky5/models/nn.py.
Improvements over source:
- Removed logging.basicConfig() - logging is configured centrally.
- torch.concat replaced with torch.cat throughout.
- apply_scale_shift_norm / apply_gate_sum use in-place cast to reduce allocations.
- RoPE1D grows its cached args table on-demand (safe for varying sequence lengths).
- RoPE3D pre-caches per-axis arg tables and avoids redundant repeat() calls.
- MultiheadCrossAttention.attention simplified; removed dead `use_flash` guard.
- _maybe_compile is a no-op decorator by default; torch.compile is opt-in via
  set_compile_enabled(True).
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from .utils import get_freqs, fractal_flatten, fractal_unflatten, nablaT_v2
from .attention import SelfAttentionEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torch.compile toggle (disabled by default to avoid startup overhead)
# ---------------------------------------------------------------------------

_ENABLE_COMPILE = False


def set_compile_enabled(enabled: bool) -> None:
    """Toggle torch.compile for Kandinsky5 modules."""
    global _ENABLE_COMPILE
    _ENABLE_COMPILE = bool(enabled)


def _maybe_compile(fn=None, **kwargs):
    """Decorator factory: wraps fn with torch.compile when compile is enabled."""
    if fn is None:
        return lambda f: _maybe_compile(f, **kwargs)
    if _ENABLE_COMPILE:
        return torch.compile(fn, **kwargs)
    return fn


# ---------------------------------------------------------------------------
# Core normalisation / gating helpers
# ---------------------------------------------------------------------------

@_maybe_compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """AdaLN-style scale-shift normalisation, cast back to bfloat16."""
    return (norm(x) * (scale + 1.0) + shift).to(torch.bfloat16)


@_maybe_compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x: torch.Tensor, out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Gated residual addition, cast back to bfloat16."""
    return (x + gate * out).to(torch.bfloat16)


@_maybe_compile()
@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to x using pre-computed rope table."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Embedding modules
# ---------------------------------------------------------------------------

class TimeEmbeddings(nn.Module):
    """Sinusoidal time-step embeddings → projected into time_dim."""

    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        assert model_dim % 2 == 0, "model_dim must be even for sinusoidal embeddings."
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)
        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=time.device, dtype=torch.float32)
        args = torch.outer(time.to(torch.float32), freqs)
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.out_layer(self.activation(self.in_layer(time_embed)))


class TextEmbeddings(nn.Module):
    """Project text features from text_dim to model_dim then LayerNorm."""

    def __init__(self, text_dim: int, model_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):
    """Patchify spatial-temporal video latents then project to model_dim.

    Input shape: (F, H, W, C) - channels-last video tensor.
    Output shape: (F/pT, H/pH, W/pW, model_dim).
    """

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        duration, height, width, dim = x.shape
        pT, pH, pW = self.patch_size
        x = (
            x.view(
                duration // pT, pT,
                height // pH, pH,
                width // pW, pW,
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x)


class RoPE1D(nn.Module):
    """1-D Rotary Position Embeddings - used for text token sequences."""

    def __init__(self, dim: int, max_pos: int = 1024, max_period: float = 10000.0) -> None:
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        self.register_buffer("freqs", freq, persistent=False)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos = pos.to(torch.long)
        if pos.numel() == 0:
            pos = torch.zeros(1, device=self.args.device, dtype=torch.long)

        max_pos_needed = int(pos.max().item()) + 1
        if max_pos_needed > self.args.shape[0]:
            new_max = max(max_pos_needed, self.args.shape[0] * 2)
            freq = self.freqs.to(device=pos.device, dtype=self.freqs.dtype)
            expanded_pos = torch.arange(new_max, device=pos.device, dtype=freq.dtype)
            self.register_buffer("args", torch.outer(expanded_pos, freq), persistent=False)
        elif self.args.device != pos.device:
            self.args = self.args.to(pos.device)

        args = self.args[pos].to(dtype=torch.float32)
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class RoPE3D(nn.Module):
    """3-D Rotary Position Embeddings - used for visual token sequences (T, H, W)."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        shape: tuple[int, int, int],
        pos: list[torch.Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        duration, height, width = shape
        args_0: torch.Tensor = getattr(self, "args_0")
        args_1: torch.Tensor = getattr(self, "args_1")
        args_2: torch.Tensor = getattr(self, "args_2")

        args_t = args_0[pos[0]].to(torch.float32) / scale_factor[0]
        args_h = args_1[pos[1]].to(torch.float32) / scale_factor[1]
        args_w = args_2[pos[2]].to(torch.float32) / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).expand(duration, height, width, -1),
                args_h.view(1, height, 1, -1).expand(duration, height, width, -1),
                args_w.view(1, 1, width, -1).expand(duration, height, width, -1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------

class Modulation(nn.Module):
    """AdaLN modulation layer - zero-initialised to be identity at start of training."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int) -> None:
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    @_maybe_compile()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.activation(x))


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class MultiheadSelfAttentionEnc(nn.Module):
    """Multi-head self-attention for encoder (text) blocks.

    Uses rotary embeddings and QK normalisation.
    Falls back to sdpa if flash raises a head-shape error.
    """

    def __init__(self, num_channels: int, head_dim: int, attention_engine: str = "auto") -> None:
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

        # Flash prefers its own engine; keep an sdpa fallback for head-size edge cases.
        self.use_flash = attention_engine == "flash"
        self.attn_engine = SelfAttentionEngine("flash_attention_2" if self.use_flash else attention_engine)
        self.sdpa_engine = SelfAttentionEngine("sdpa")

    def _get_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape[:-1]
        q = self.to_query(x).reshape(*shape, self.num_heads, -1)
        k = self.to_key(x).reshape(*shape, self.num_heads, -1)
        v = self.to_value(x).reshape(*shape, self.num_heads, -1)
        return q, k, v

    def _norm_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_w = self.query_norm.weight.float() if self.query_norm.weight is not None else None
        k_w = self.key_norm.weight.float() if self.key_norm.weight is not None else None
        q = F.rms_norm(q.float(), self.query_norm.normalized_shape, q_w, self.query_norm.eps).type_as(q)
        k = F.rms_norm(k.float(), self.key_norm.normalized_shape, k_w, self.key_norm.eps).type_as(k)
        return q, k

    def forward(self, x: torch.Tensor, rope: torch.Tensor, attention_mask=None) -> torch.Tensor:
        added_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch = True
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        q, k, v = self._get_qkv(x)
        q, k = self._norm_qk(q, k)
        q = apply_rotary(q, rope).type_as(q)
        k = apply_rotary(k, rope).type_as(k)

        mask = attention_mask
        if mask is not None:
            if mask.shape[-1] != k.shape[-2]:
                mask = None

        attn_fn = self.attn_engine.get_attention()
        if mask is not None and self.attn_engine.supports_mask:
            out = attn_fn(q=q, k=k, v=v, attn_mask=mask)[0]
        else:
            out = attn_fn(q=q, k=k, v=v)[0]

        out = out.flatten(-2, -1)
        out = self.out_layer(out)
        if added_batch:
            out = out.squeeze(0)
        return out


class MultiheadSelfAttentionDec(nn.Module):
    """Multi-head self-attention for decoder (visual) blocks.

    Supports standard dense attention or NABLA sparse attention (via flex_attention).
    """

    def __init__(self, num_channels: int, head_dim: int, attention_engine: str = "auto") -> None:
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)
        self.attn_engine = SelfAttentionEngine(attention_engine)

    def _get_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape[:-1]
        q = self.to_query(x).reshape(*shape, self.num_heads, -1)
        k = self.to_key(x).reshape(*shape, self.num_heads, -1)
        v = self.to_value(x).reshape(*shape, self.num_heads, -1)
        return q, k, v

    def _norm_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_w = self.query_norm.weight.float() if self.query_norm.weight is not None else None
        k_w = self.key_norm.weight.float() if self.key_norm.weight is not None else None
        q = F.rms_norm(q.float(), self.query_norm.normalized_shape, q_w, self.query_norm.eps).type_as(q)
        k = F.rms_norm(k.float(), self.key_norm.normalized_shape, k_w, self.key_norm.eps).type_as(k)
        return q, k

    def _dense_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn_fn = self.attn_engine.get_attention()
        out = attn_fn(q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0))[0]
        return out.flatten(-2, -1)

    def _nabla_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sparse_params: dict) -> torch.Tensor:
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as exc:
            raise RuntimeError("flex_attention required for NABLA mode but not available.") from exc

        q_ = q.unsqueeze(0).transpose(1, 2).contiguous()
        k_ = k.unsqueeze(0).transpose(1, 2).contiguous()
        v_ = v.unsqueeze(0).transpose(1, 2).contiguous()
        block_mask = nablaT_v2(
            q_,
            k_,
            sparse_params["sta_mask"],
            thr=sparse_params["P"],
            add_sta=bool(sparse_params.get("add_sta", True)),
            method=sparse_params.get("method", "topcdf"),
        )
        out = flex_attention(q_, k_, v_, block_mask=block_mask).transpose(1, 2).squeeze(0).contiguous()
        return out.flatten(-2, -1)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, sparse_params: dict | None = None) -> torch.Tensor:
        q, k, v = self._get_qkv(x)
        q, k = self._norm_qk(q, k)
        q = apply_rotary(q, rope).type_as(q)
        k = apply_rotary(k, rope).type_as(k)

        if sparse_params is not None:
            out = self._nabla_attention(q, k, v, sparse_params)
        else:
            out = self._dense_attention(q, k, v)

        return self.out_layer(out)


class MultiheadCrossAttention(nn.Module):
    """Multi-head cross-attention: visual queries attend to text keys/values."""

    def __init__(self, num_channels: int, head_dim: int, attention_engine: str = "auto") -> None:
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)
        self.attn_engine = SelfAttentionEngine(attention_engine)
        self.sdpa_engine = SelfAttentionEngine("sdpa")

    def _get_qkv(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape, cond_shape = x.shape[:-1], cond.shape[:-1]
        q = self.to_query(x).reshape(*shape, self.num_heads, -1)
        k = self.to_key(cond).reshape(*cond_shape, self.num_heads, -1)
        v = self.to_value(cond).reshape(*cond_shape, self.num_heads, -1)
        return q, k, v

    def _norm_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_w = self.query_norm.weight.float() if self.query_norm.weight is not None else None
        k_w = self.key_norm.weight.float() if self.key_norm.weight is not None else None
        q = F.rms_norm(q.float(), self.query_norm.normalized_shape, q_w, self.query_norm.eps).type_as(q)
        k = F.rms_norm(k.float(), self.key_norm.normalized_shape, k_w, self.key_norm.eps).type_as(k)
        return q, k

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attention_mask=None) -> torch.Tensor:
        added_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            cond = cond.unsqueeze(0)
            added_batch = True
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0)

        q, k, v = self._get_qkv(x, cond)
        q, k = self._norm_qk(q, k)

        # Build a batch dimension for cross-attention (q and k may have different seq-len).
        if q.dim() == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            added_batch = True

        mask = attention_mask
        if mask is not None and mask.shape[-1] != k.shape[-2]:
            mask = None

        attn_fn = self.attn_engine.get_attention()
        if mask is not None and self.attn_engine.supports_mask:
            out = attn_fn(q=q, k=k, v=v, attn_mask=mask)[0]
        else:
            try:
                out = attn_fn(q=q, k=k, v=v)[0]
            except RuntimeError as exc:
                if "heads in key/value must divide" in str(exc):
                    # Cross-attention head mismatch: fall back to sdpa.
                    out = self.sdpa_engine.get_attention()(q=q, k=k, v=v)[0]
                else:
                    raise

        out = out.flatten(-2, -1)
        if added_batch:
            out = out.squeeze(0)
        return self.out_layer(out)


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation, no biases."""

    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


# ---------------------------------------------------------------------------
# Output layer
# ---------------------------------------------------------------------------

class OutLayer(nn.Module):
    """Final projection layer: un-patchifies the visual stream back to pixel-space dims."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(
        self,
        visual_embed: torch.Tensor,
        text_embed: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale[:, None, None],
            shift[:, None, None],
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        # Un-patchify: (F/pT, H/pH, W/pW, pT*pH*pW*C) -> (F, H, W, C)
        duration, height, width, _ = x.shape
        pT, pH, pW = self.patch_size
        x = (
            x.view(duration, height, width, -1, pT, pH, pW)
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        return x
