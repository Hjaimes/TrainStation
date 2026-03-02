"""QwenImage attention, MLP, and normalization modules.

Ported from Musubi_Tuner qwen_image_modules.py and the relevant module classes
in qwen_image_model.py.

Porting improvements:
  - Removed logging.basicConfig()
  - Replaced commented-out dead code with clean implementations
  - Used consistent PyTorch APIs
  - Added type hints on public methods
"""
from __future__ import annotations

import math
import numbers
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

_ACT_MAP = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Return an instantiated activation module by name."""
    key = act_fn.lower()
    if key not in _ACT_MAP:
        raise ValueError(f"Unknown activation '{act_fn}'. Available: {list(_ACT_MAP)}")
    return _ACT_MAP[key]()


# ---------------------------------------------------------------------------
# Timestep utilities
# ---------------------------------------------------------------------------

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (matches DDPM paper implementation)."""
    assert len(timesteps.shape) == 1, "timesteps must be 1-D"

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


class Timesteps(nn.Module):
    """Projects scalar timesteps to a frequency embedding."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
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
    """Two-layer MLP that projects frequency embeddings to model dimension."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim: Optional[int] = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.cond_proj = (
            nn.Linear(cond_proj_dim, in_channels, bias=False)
            if cond_proj_dim is not None
            else None
        )
        self.act = get_activation(act_fn)
        out_channels = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, out_channels, sample_proj_bias)
        self.post_act = get_activation(post_act_fn) if post_act_fn is not None else None

    def forward(self, sample: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS Norm (Zhang et al., 2019). Supports fp8 weight dtype."""

    def __init__(
        self,
        dim: Union[int, tuple],
        eps: float,
        elementwise_affine: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.dim = torch.Size(dim)
        self.weight: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            elif self.weight.dtype == torch.float8_e4m3fn:
                hidden_states = hidden_states * self.weight.to(hidden_states.dtype)
                if self.bias is not None:
                    hidden_states = hidden_states + self.bias.to(hidden_states.dtype)
                return hidden_states.to(input_dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    """Adaptive layer norm with scale+shift derived from a conditioning embedding."""

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
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"Unknown norm_type '{norm_type}'")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class GELUApprox(nn.Module):
    """GELU with tanh approximation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate="tanh")


class FeedForward(nn.Module):
    """Transformer feed-forward block with GELU-approximate activation."""

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
    ):
        super().__init__()
        if activation_fn != "gelu-approximate":
            raise ValueError(f"QwenImage FeedForward only supports 'gelu-approximate', got '{activation_fn}'")
        inner_dim = int(dim * mult)
        out_channels = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([
            GELUApprox(dim, inner_dim, bias=bias),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, out_channels, bias=bias),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------

@torch.compiler.disable
def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    use_real: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings using complex multiplication.

    Args:
        x: Query or key tensor [B, S, H, D].
        freqs_cis: Complex frequency tensor [S, D//2].
        use_real: If True use real/imag split (not used for QwenImage).
    """
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)  # [S, 1, D//2]
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    """3D RoPE for QwenImage (t2i and edit modes).

    Frequency tensors are pre-computed and cached. The rope_cache dict avoids
    redundant computation for repeated (height, width) shapes.
    """

    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self._rope_params(pos_index, axes_dim[0], theta),
            self._rope_params(pos_index, axes_dim[1], theta),
            self._rope_params(pos_index, axes_dim[2], theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self._rope_params(neg_index, axes_dim[0], theta),
            self._rope_params(neg_index, axes_dim[1], theta),
            self._rope_params(neg_index, axes_dim[2], theta),
        ], dim=1)
        # Cache rope computations per shape
        self.rope_cache: dict = {}

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: list,
        txt_seq_lens: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list) and video_fhw and isinstance(video_fhw[0], list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if rope_key not in self.rope_cache:
                self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
            video_freq = self.rope_cache[rope_key].to(device)
            vid_freqs.append(video_freq)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len]
        vid_freqs_cat = torch.cat(vid_freqs, dim=0)
        return vid_freqs_cat, txt_freqs

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        half_dims = [x // 2 for x in self.axes_dim]
        freqs_pos = self.pos_freqs.split(half_dims, dim=1)
        freqs_neg = self.neg_freqs.split(half_dims, dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenEmbedLayer3DRope(QwenEmbedRope):
    """3D RoPE for the layered mode.

    Condition image gets a special negative frame-index to distinguish it from
    the target layers.
    """

    def forward(
        self,
        video_fhw: list,
        txt_seq_lens: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list) and video_fhw and isinstance(video_fhw[0], list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            is_cond = idx == layer_num
            rope_key = f"{is_cond}_{idx}_{height}_{width}"
            if rope_key not in self.rope_cache:
                if not is_cond:
                    freq = self._compute_video_freqs(frame, height, width, idx)
                else:
                    freq = self._compute_condition_freqs(frame, height, width)
                self.rope_cache[rope_key] = freq
            vid_freqs.append(self.rope_cache[rope_key].to(device))
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    def _compute_condition_freqs(self, frame: int, height: int, width: int) -> torch.Tensor:
        seq_lens = frame * height * width
        half_dims = [x // 2 for x in self.axes_dim]
        freqs_pos = self.pos_freqs.split(half_dims, dim=1)
        freqs_neg = self.neg_freqs.split(half_dims, dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


# ---------------------------------------------------------------------------
# Timestep projection
# ---------------------------------------------------------------------------

class QwenTimestepProjEmbeddings(nn.Module):
    """Project scalar timesteps → model dimension conditioning vector.

    Optionally supports an additional_t_cond embedding (for layered mode,
    which uses an 'is_rgb' index to distinguish layer types).
    """

    def __init__(self, embedding_dim: int, use_additional_t_cond: bool = False):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        addition_t_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        conditioning = timesteps_emb

        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("addition_t_cond must be provided when use_additional_t_cond=True")
            conditioning = conditioning + self.addition_t_embedding(addition_t_cond).to(hidden_states.dtype)

        return conditioning
