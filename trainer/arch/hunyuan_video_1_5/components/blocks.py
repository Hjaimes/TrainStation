"""HunyuanVideo 1.5 transformer blocks.

Only MMDoubleStreamBlock is included — HV 1.5 has NO single-stream blocks.
Self-contained: all imports from this package, not from hunyuan_video/.

Porting improvements over Musubi_Tuner source:
- print() → logging
- Removed dead/commented code (LinearWarpforSingle, etc.)
- Consistent torch API
- cpu_offloading_wrapper inlined locally
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn
from einops import rearrange

from .attention import AttentionParams, attention
from .embeddings import apply_rotary_emb
from .layers import MLP, ModulateDiT, RMSNorm, modulate, apply_gate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU offloading wrapper (inlined from musubi_tuner.utils.model_utils)
# ---------------------------------------------------------------------------

def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(e, device) for e in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """Move inputs to CPU, run func, move outputs back to device."""
    def wrapper(*args, **kwargs):
        cpu_args = tuple(_to_device(a, torch.device("cpu")) for a in args)
        cpu_kwargs = {k: _to_device(v, torch.device("cpu")) for k, v in kwargs.items()}
        result = func(*cpu_args, **cpu_kwargs)
        return _to_device(result, device)
    return wrapper


# ---------------------------------------------------------------------------
# Token refiner (needed by SingleTokenRefiner inside model.py)
# ---------------------------------------------------------------------------

class IndividualTokenRefinerBlock(nn.Module):
    """Single block of the text token refiner."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.heads_num = heads_num
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        # Identity norms (qk_norm=False for refiner)
        self.self_attn_q_norm = nn.Identity()
        self.self_attn_k_norm = nn.Identity()
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=nn.SiLU, drop=mlp_drop_rate)

        # AdaLN modulation: shift + gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_params: AttentionParams) -> torch.Tensor:
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        del norm_x
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num).unbind(0)
        del qkv

        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        attn_out = attention([q, k, v], attn_params=attn_params)
        x = x + apply_gate(self.self_attn_proj(attn_out), gate_msa)
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
        return x


class IndividualTokenRefiner(nn.Module):
    """Stack of token refiner blocks."""

    def __init__(self, hidden_size: int, heads_num: int, depth: int, mlp_width_ratio: float = 4.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            IndividualTokenRefinerBlock(hidden_size, heads_num, mlp_width_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_params: AttentionParams) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, c, attn_params)
        return x


class SingleTokenRefiner(nn.Module):
    """Refine text embeddings with timestep + context conditioning.

    Used to pre-process Qwen2.5-VL word-level embeddings before the main
    double-stream blocks.
    """

    def __init__(self, in_channels: int, hidden_size: int, heads_num: int, depth: int) -> None:
        super().__init__()
        from .embeddings import TimestepEmbedder, TextProjection
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, nn.SiLU)
        self.c_embedder = TextProjection(in_channels, hidden_size, nn.SiLU)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, attn_params: AttentionParams) -> torch.Tensor:
        t_emb = self.t_embedder(t)  # [B, hidden_size]

        # Average valid tokens for context-aware conditioning
        txt_lens = attn_params.seqlens  # [B] int32 lengths
        ctx = torch.stack(
            [x[i, : int(txt_lens[i].item())].mean(dim=0) for i in range(x.shape[0])], dim=0
        )
        c = t_emb + self.c_embedder(ctx)
        del t_emb, ctx

        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, attn_params)
        return x


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Output projection with AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: list[int],
        out_channels: int,
        act_layer: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        out_size = patch_size[0] * patch_size[1] * patch_size[2] * out_channels
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        del shift, scale, c
        return self.linear(x)


# ---------------------------------------------------------------------------
# MMDoubleStreamBlock
# ---------------------------------------------------------------------------

class MMDoubleStreamBlock(nn.Module):
    """Multimodal double-stream transformer block for HunyuanVideo 1.5.

    Processes image and text token streams separately but applies joint
    cross-modal attention via concatenation. HV 1.5 has 54 of these blocks
    and no single-stream blocks.

    Args:
        hidden_size: Model dimension (2048 for HV 1.5).
        heads_num: Attention heads (16 for HV 1.5).
        mlp_width_ratio: MLP expansion factor (4.0).
        mlp_act_type: Activation — must be "gelu_tanh".
        qk_norm: Must be True (RMS QK-norm is always on).
        qk_norm_type: Must be "rms".
        qkv_bias: Whether to use bias in QKV projections.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        assert mlp_act_type == "gelu_tanh", f"Expected 'gelu_tanh' activation, got '{mlp_act_type}'"
        assert qk_norm_type == "rms", f"Expected 'rms' QK-norm, got '{qk_norm_type}'"
        assert qk_norm, "QK normalization must be enabled for MMDoubleStreamBlock"

        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        gelu_tanh = lambda: nn.GELU(approximate="tanh")

        # --- Image stream ---
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=nn.SiLU)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.img_attn_q_norm = RMSNorm(head_dim, eps=1e-6)
        self.img_attn_k_norm = RMSNorm(head_dim, eps=1e-6)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=gelu_tanh, bias=True)

        # --- Text stream ---
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=nn.SiLU)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.txt_attn_q_norm = RMSNorm(head_dim, eps=1e-6)
        self.txt_attn_k_norm = RMSNorm(head_dim, eps=1e-6)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=gelu_tanh, bias=True)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False) -> None:
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_params: AttentionParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract 6 modulation params per stream
        img_shifts_scales_gates = self.img_mod(vec).chunk(6, dim=-1)
        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_shifts_scales_gates[:3]
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_shifts_scales_gates[3:]

        txt_shifts_scales_gates = self.txt_mod(vec).chunk(6, dim=-1)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_shifts_scales_gates[:3]
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_shifts_scales_gates[3:]

        # --- Image stream: attention ---
        img_mod = modulate(self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale)
        del img_mod1_shift, img_mod1_scale

        img_q = rearrange(self.img_attn_q(img_mod), "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(self.img_attn_k(img_mod), "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(self.img_attn_v(img_mod), "B L (H D) -> B L H D", H=self.heads_num)
        del img_mod

        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE to image tokens
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)
            del freqs_cis

        # --- Text stream: attention ---
        txt_mod = modulate(self.txt_norm1(txt), shift=txt_mod1_shift, scale=txt_mod1_scale)

        txt_q = rearrange(self.txt_attn_q(txt_mod), "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(self.txt_attn_k(txt_mod), "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(self.txt_attn_v(txt_mod), "B L (H D) -> B L H D", H=self.heads_num)
        del txt_mod

        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # --- Joint cross-modal attention ---
        img_seq_len = img.shape[1]
        q = torch.cat([img_q, txt_q], dim=1)
        del img_q, txt_q
        k = torch.cat([img_k, txt_k], dim=1)
        del img_k, txt_k
        v = torch.cat([img_v, txt_v], dim=1)
        del img_v, txt_v

        attn_out = attention([q, k, v], attn_params=attn_params)

        img_attn = attn_out[:, :img_seq_len].contiguous()
        txt_attn = attn_out[:, img_seq_len:].contiguous()
        del attn_out

        # --- Residual updates ---
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        del img_attn, img_mod1_gate

        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )
        del img_mod2_shift, img_mod2_scale, img_mod2_gate

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        del txt_attn, txt_mod1_gate

        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        del txt_mod2_shift, txt_mod2_scale, txt_mod2_gate

        return img, txt

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_params: AttentionParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_checkpointing and self.training:
            forward_fn = self._forward
            if self.cpu_offload_checkpointing:
                forward_fn = _create_cpu_offloading_wrapper(
                    forward_fn, self.img_attn_q.weight.device
                )
            return torch.utils.checkpoint.checkpoint(
                forward_fn, img, txt, vec, freqs_cis, attn_params, use_reentrant=False
            )
        return self._forward(img, txt, vec, freqs_cis, attn_params)
