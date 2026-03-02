"""MMDoubleStreamBlock and MMSingleStreamBlock for HunyuanVideo.

Ported from Musubi_Tuner's hunyuan_model/models.py.
Improvements:
  - Removed logging.basicConfig(), print() calls
  - Removed dead/commented-out code
  - Used torch.addcmul instead of apply_gate helper
  - Cleaner variable lifetimes for memory efficiency
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .attention import attention, get_cu_seqlens
from .embeddings import apply_rotary_emb
from .layers import MLP, ModulateDiT, get_activation_layer, get_norm_layer, modulate

logger = logging.getLogger(__name__)


class MMDoubleStreamBlock(nn.Module):
    """Multimodal double-stream DiT block.

    Processes image and text tokens with separate modulation streams,
    then combines them for joint attention (SD3 / Flux-style).
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
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attn_mode: str = "flash",
        split_attn: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.heads_num = heads_num

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        silu = get_activation_layer("silu")

        # Image stream
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=silu, **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size, mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        # Text stream
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=silu, **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size, mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        total_len: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift, img_mod1_scale, img_mod1_gate,
            img_mod2_shift, img_mod2_scale, img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift, txt_mod1_scale, txt_mod1_gate,
            txt_mod2_shift, txt_mod2_scale, txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image QKV
        img_modulated = modulate(self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_modulated = None
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        img_qkv = None
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE to image tokens only
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)

        # Prepare text QKV
        txt_modulated = modulate(self.txt_norm1(txt), shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_modulated = None
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        txt_qkv = None
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Concatenate for joint attention
        batch_size = img_q.shape[0]
        q = torch.cat((img_q, txt_q), dim=1); img_q = txt_q = None
        k = torch.cat((img_k, txt_k), dim=1); img_k = txt_k = None
        v = torch.cat((img_v, txt_v), dim=1); img_v = txt_v = None

        attn = attention(
            [q, k, v],
            mode=self.attn_mode,
            attn_mask=attn_mask,
            total_len=total_len,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=batch_size,
        )

        img_attn = attn[:, : img.shape[1]]
        txt_attn = attn[:, img.shape[1]:]
        attn = None

        # Update image stream
        img = torch.addcmul(img, self.img_attn_proj(img_attn), img_mod1_gate.unsqueeze(1))
        img_attn = None
        img = torch.addcmul(
            img,
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            img_mod2_gate.unsqueeze(1),
        )

        # Update text stream
        txt = torch.addcmul(txt, self.txt_attn_proj(txt_attn), txt_mod1_gate.unsqueeze(1))
        txt_attn = None
        txt = torch.addcmul(
            txt,
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            txt_mod2_gate.unsqueeze(1),
        )

        return img, txt

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)


class MMSingleStreamBlock(nn.Module):
    """Single-stream DiT block: parallel attention + MLP.

    Adapted from SD3 / Flux single-stream design.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attn_mode: str = "flash",
        split_attn: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        qk_norm_layer = get_norm_layer(qk_norm_type)
        silu = get_activation_layer("silu")

        # Fused QKV + MLP-in projection
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs)
        # Fused attn-out + MLP-out projection
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs)

        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm else nn.Identity()
        )
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=silu, **factory_kwargs)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        attn_mask: Optional[torch.Tensor] = None,
        total_len: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Optional[Tuple] = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        x_mod = None

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        qkv = None
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE only to image portion (not text tokens at the end)
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
            img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
            q = k = None
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            del img_q, txt_q, img_k, txt_k

        attn = attention(
            [q, k, v],
            mode=self.attn_mode,
            attn_mask=attn_mask,
            total_len=total_len,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )

        mlp = self.mlp_act(mlp)
        attn_mlp = torch.cat((attn, mlp), dim=2)
        attn = mlp = None

        output = self.linear2(attn_mlp)
        attn_mlp = None
        return torch.addcmul(x, output, mod_gate.unsqueeze(1))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)
