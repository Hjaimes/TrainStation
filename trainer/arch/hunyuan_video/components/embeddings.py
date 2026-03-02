"""Embedding layers for HunyuanVideo: RoPE, timestep, patch, text, token refiner.

Ported from Musubi_Tuner's hunyuan_model/{posemb_layers,embed_layers,token_refiner}.py.
Improvements:
  - Removed logging.basicConfig() calls
  - Removed print() statements
  - Removed dead/commented-out code
  - Pre-computed freqs_cos/freqs_sin with repeat_interleave (use_real=True path)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import attention
from .layers import (
    MLP,
    RMSNorm,
    get_activation_layer,
    get_norm_layer,
    modulate,
)


# ---------------------------------------------------------------------------
# Re-export TimestepEmbedder here (defined in layers.py, imported by model.py)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal encoding + MLP."""

    def __init__(
        self,
        hidden_size: int,
        act_layer,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        out_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        out_size = out_size or hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self._sinusoidal(t).to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


# ---------------------------------------------------------------------------
# PatchEmbed (3D convolution-based)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """3D patch embedding: (B, C, T, H, W) -> (B, S, hidden_size)."""

    def __init__(
        self,
        patch_size,
        in_chans: int = 16,
        embed_dim: int = 3072,
        norm_layer=None,
        flatten: bool = True,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, C, T, H, W) -> (B, THW, C)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# TextProjection
# ---------------------------------------------------------------------------

class TextProjection(nn.Module):
    """Two-layer MLP for projecting text embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        act_layer,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden = self.linear_1(caption)
        hidden = self.act_1(hidden)
        return self.linear_2(hidden)


# ---------------------------------------------------------------------------
# RoPE position embeddings
# ---------------------------------------------------------------------------

def _to_tuple(x, dim: int = 2):
    if isinstance(x, int):
        return (x,) * dim
    if len(x) == dim:
        return tuple(x)
    raise ValueError(f"Expected length {dim} or int, but got {x!r}")


def get_meshgrid_nd(start, *args, dim: int = 2) -> torch.Tensor:
    """Create n-D meshgrid with linspace semantics.

    Returns:
        grid: [dim, *sizes] float32 tensor.
    """
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start_t = (0,) * dim
        stop_t = num
    elif len(args) == 1:
        start_t = _to_tuple(start, dim=dim)
        stop_t = _to_tuple(args[0], dim=dim)
        num = tuple(stop_t[i] - start_t[i] for i in range(dim))
    elif len(args) == 2:
        start_t = _to_tuple(start, dim=dim)
        stop_t = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"Expected 0-2 extra args, got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start_t[i], stop_t[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)

    grid = torch.meshgrid(*axis_grid, indexing="ij")
    return torch.stack(grid, dim=0)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.Tensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute 1-D rotary position embeddings.

    Args:
        dim: Embedding dimension (must be even).
        pos: Positions, either an int (0..pos-1) or a float tensor [S].
        theta: Base frequency.
        use_real: If True, return (cos, sin) real tensors; else return complex.

    Returns:
        If use_real: (freqs_cos [S, D], freqs_sin [S, D])
        Else:        freqs_cis [S, D//2] complex64
    """
    if isinstance(pos, int):
        pos = torch.arange(pos, dtype=torch.float32)

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    freqs = torch.outer(pos.float() * interpolation_factor, freqs)  # [S, D/2]

    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        return torch.polar(torch.ones_like(freqs), freqs)  # [S, D/2] complex64


def get_nd_rotary_pos_embed(
    rope_dim_list: List[int],
    start,
    *args,
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """N-dimensional RoPE position embeddings.

    Args:
        rope_dim_list: Per-axis embedding dims, must sum to head_dim.
        start: Grid start (int or tuple) — see get_meshgrid_nd.
        *args: Additional grid args.
        theta: Base frequency.
        use_real: Return (cos, sin) instead of complex tensor.

    Returns:
        If use_real: (cos [S, D], sin [S, D])
        Else:        freqs_cis [S, D//2] complex64
    """
    n = len(rope_dim_list)
    grid = get_meshgrid_nd(start, *args, dim=n)  # [n, ...]

    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [float(theta_rescale_factor)] * n
    elif len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * n

    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [float(interpolation_factor)] * n
    elif len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * n

    embs = []
    for i in range(n):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([e[0] for e in embs], dim=1)
        sin = torch.cat([e[1] for e in embs], dim=1)
        return cos, sin
    else:
        return torch.cat(embs, dim=1)


def get_rotary_pos_embed_by_shape(
    patch_size: List[int],
    hidden_size: int,
    heads_num: int,
    rope_dim_list: List[int],
    rope_theta: float,
    latents_size: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE (cos, sin) tensors given latent spatial dimensions.

    Args:
        patch_size: [pt, ph, pw] patch sizes.
        hidden_size: transformer hidden dim.
        heads_num: number of attention heads.
        rope_dim_list: per-axis RoPE dims [t_dim, h_dim, w_dim].
        rope_theta: base frequency (256 for HunyuanVideo).
        latents_size: [T, H, W] latent dimensions.

    Returns:
        freqs_cos, freqs_sin: [S, head_dim] tensors.
    """
    rope_sizes = [
        latents_size[i] // patch_size[i]
        for i in range(len(latents_size))
    ]
    # Ensure 3D
    if len(rope_sizes) < 3:
        rope_sizes = [1] * (3 - len(rope_sizes)) + rope_sizes

    head_dim = hidden_size // heads_num
    assert sum(rope_dim_list) == head_dim, (
        f"rope_dim_list sum {sum(rope_dim_list)} != head_dim {head_dim}"
    )

    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1.0,
    )
    return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# RoPE application
# ---------------------------------------------------------------------------

def _reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    head_first: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Reshape freqs for broadcasting against query/key tensors."""
    ndim = x.ndim
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
        if head_first:
            shape = [d if i in (ndim - 2, ndim - 1) else 1 for i, d in enumerate(x.shape)]
        else:
            shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
        return cos.view(*shape), sin.view(*shape)
    else:
        if head_first:
            shape = [d if i in (ndim - 2, ndim - 1) else 1 for i, d in enumerate(x.shape)]
        else:
            shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq: [B, S, H, D] query tensor.
        xk: [B, S, H, D] key tensor.
        freqs_cis: Precomputed cos/sin tuple or complex tensor.
        head_first: If True, head dim is before seq dim.

    Returns:
        (xq_out, xk_out) with rotary embeddings applied.
    """
    if isinstance(freqs_cis, tuple):
        cos, sin = _reshape_for_broadcast(freqs_cis, xq, head_first)
        cos = cos.to(xq.device)
        sin = sin.to(xq.device)
        xq_out = (xq.float() * cos + _rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + _rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        freqs = _reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)
        xq_out = torch.view_as_real(xq_ * freqs).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xk_out = torch.view_as_real(xk_ * freqs).flatten(3).type_as(xk)

    return xq_out, xk_out


# ---------------------------------------------------------------------------
# SingleTokenRefiner (for LLM text embedding refinement)
# ---------------------------------------------------------------------------

class IndividualTokenRefinerBlock(nn.Module):
    """Single block of the token refiner: self-attention + MLP with adaLN."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        act_factory = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_factory,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )
        self.adaLN_modulation = nn.Sequential(
            act_factory(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from einops import rearrange
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        attn_out = attention(q, k, v, mode="torch", attn_mask=attn_mask)
        x = torch.addcmul(x, self.self_attn_proj(attn_out), gate_msa.unsqueeze(1))
        x = torch.addcmul(x, self.mlp(self.norm2(x)), gate_mlp.unsqueeze(1))
        return x

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)


class IndividualTokenRefiner(nn.Module):
    """Stack of IndividualTokenRefinerBlocks."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.blocks = nn.ModuleList([
            IndividualTokenRefinerBlock(
                hidden_size=hidden_size,
                heads_num=heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_drop_rate=mlp_drop_rate,
                act_type=act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                **factory_kwargs,
            )
            for _ in range(depth)
        ])

    def enable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            block.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            block.disable_gradient_checkpointing()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if mask is not None:
            b, s = mask.shape[0], mask.shape[1]
            mask = mask.to(x.device)
            m1 = mask.view(b, 1, 1, s).expand(b, 1, s, s)
            m2 = m1.transpose(2, 3)
            self_attn_mask = (m1 & m2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    """Token refiner for LLM text embeddings (LI-DiT style).

    Refines token-level text embeddings conditioned on the timestep
    before passing them to the main transformer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        act_factory = get_activation_layer(act_type)
        self.t_embedder = TimestepEmbedder(hidden_size, act_factory, **factory_kwargs)
        self.c_embedder = TextProjection(in_channels, hidden_size, act_factory, **factory_kwargs)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs,
        )

    def enable_gradient_checkpointing(self) -> None:
        self.individual_token_refiner.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.individual_token_refiner.disable_gradient_checkpointing()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_repr = self.t_embedder(t)

        if mask is None:
            ctx_repr = x.mean(dim=1)
        else:
            mask_f = mask.float().unsqueeze(-1)
            ctx_repr = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1)
        ctx_repr = self.c_embedder(ctx_repr)

        c = t_repr + ctx_repr
        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, mask)
        return x
