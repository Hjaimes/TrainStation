"""QwenImage VAE with per-channel normalization.

Ported from Musubi_Tuner qwen_image_autoencoder_kl.py.

Porting improvements:
  - Removed logging.basicConfig()
  - Removed commented-out dead code
  - Cached conv3d counts at init time (avoids module scan every call)
  - Added type hints on public methods
  - Used consistent PyTorch APIs
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import get_activation

logger = logging.getLogger(__name__)

# Number of frames to cache for causal convolutions during chunked encode/decode
_CACHE_T = 2


# ---------------------------------------------------------------------------
# Gaussian distribution
# ---------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    """Diagonal Gaussian as returned by the VAE encoder."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None and generator.device.type != self.parameters.device.type:
            rand_device = generator.device
        else:
            rand_device = self.parameters.device
        noise = torch.randn(
            self.mean.shape, generator=generator, device=rand_device, dtype=self.parameters.dtype
        ).to(self.parameters.device)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0])
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class QwenImageCausalConv3d(nn.Conv3d):
    """3D causal convolution - zero-pads the time dimension on the left only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Set up causal padding (time axis: 2*p left, 0 right; h/w: symmetric)
        self._causal_padding = (
            self.padding[2], self.padding[2],
            self.padding[1], self.padding[1],
            2 * self.padding[0], 0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        padding = list(self._causal_padding)
        if cache_x is not None and self._causal_padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))


class QwenImageRMSNorm(nn.Module):
    """Channel-wise RMS norm for feature maps."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias_param = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dim = 1 if self.channel_first else -1
        out = F.normalize(x, dim=norm_dim) * self.scale * self.gamma
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


class _Upsample(nn.Upsample):
    """Upsample that preserves input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    """2D/3D resampling module (upsample or downsample)."""

    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                _Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                _Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()

        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -_CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                x = self.time_conv(x, None if feat_cache[idx] == "Rep" else feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), dim=3).reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class QwenImageResidualBlock(nn.Module):
    """Residual block for encoder/decoder."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu"):
        super().__init__()
        self.nonlinearity = get_activation(non_linearity)
        self.norm1 = QwenImageRMSNorm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        h = self.conv_shortcut(x)
        x = self.nonlinearity(self.norm1(x))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.dropout(self.nonlinearity(self.norm2(x)))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class QwenImageAttentionBlock(nn.Module):
    """Single-head 2D self-attention within each temporal frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x).view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class QwenImageMidBlock(nn.Module):
    """Middle block (residual → attention → residual)."""

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class QwenImageEncoder3d(nn.Module):
    """3D encoder with temporal + spatial downsampling."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = None,
        num_res_blocks: int = 2,
        attn_scales: List[float] = None,
        temporal_downsample: List[bool] = None,
        dropout: float = 0.0,
        input_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [True, True, False]

        self.nonlinearity = get_activation(non_linearity)
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv_in = QwenImageCausalConv3d(input_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_d, out_d, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_d))
                in_d = out_d
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_d, mode=mode))
                scale /= 2.0
        self.mid_block = QwenImageMidBlock(out_d, dropout, non_linearity, num_layers=1)
        self.norm_out = QwenImageRMSNorm(out_d, images=False)
        self.conv_out = QwenImageCausalConv3d(out_d, z_dim, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)

        x = self.mid_block(x, feat_cache, feat_idx)
        x = self.nonlinearity(self.norm_out(x))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """Upsampling block for the decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(cur, out_dim, dropout, non_linearity))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])
            if upsample_mode is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx) if feat_cache is not None else resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache, feat_idx) if feat_cache is not None else self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    """3D decoder with temporal + spatial upsampling."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = None,
        num_res_blocks: int = 2,
        attn_scales: List[float] = None,
        temporal_upsample: List[bool] = None,
        dropout: float = 0.0,
        output_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_upsample is None:
            temporal_upsample = [False, True, True]

        self.nonlinearity = get_activation(non_linearity)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)
        self.up_blocks = nn.ModuleList()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_d = in_d // 2
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
            self.up_blocks.append(
                QwenImageUpBlock(in_d, out_d, num_res_blocks, dropout, upsample_mode, non_linearity)
            )
        self.norm_out = QwenImageRMSNorm(out_d, images=False)
        self.conv_out = QwenImageCausalConv3d(out_d, output_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache, feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        x = self.nonlinearity(self.norm_out(x))

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


# ---------------------------------------------------------------------------
# AutoencoderKLQwenImage
# ---------------------------------------------------------------------------

# Per-channel normalization constants derived from the Qwen-Image VAE checkpoint
_DEFAULT_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
]
_DEFAULT_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


class AutoencoderKLQwenImage(nn.Module):
    """Variational autoencoder for the QwenImage architecture.

    Features:
      - 8× spatial compression (3 downsample stages)
      - Per-channel mean/std normalization of latents
      - Causal 3D convolutions for temporal consistency
      - Optional tiling for large images
      - encode_pixels_to_latents / decode_to_pixels convenience methods
    """

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: List[float] = None,
        temporal_downsample: List[bool] = None,
        dropout: float = 0.0,
        latents_mean: List[float] = None,
        latents_std: List[float] = None,
        input_channels: int = 3,
    ):
        super().__init__()
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]
        if latents_mean is None:
            latents_mean = _DEFAULT_LATENTS_MEAN
        if latents_std is None:
            latents_std = _DEFAULT_LATENTS_STD

        self.z_dim = z_dim
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        temporal_upsample = temporal_downsample[::-1]

        dim_mult_list = list(dim_mult)
        self.encoder = QwenImageEncoder3d(
            base_dim, z_dim * 2, dim_mult_list, num_res_blocks, attn_scales,
            temporal_downsample, dropout, input_channels,
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder3d(
            base_dim, z_dim, dim_mult_list, num_res_blocks, attn_scales,
            temporal_upsample, dropout, input_channels,
        )

        self.spatial_compression_ratio = 2 ** len(temporal_downsample)
        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Pre-cache conv counts to avoid module scans at runtime
        self._dec_conv_count = sum(
            isinstance(m, QwenImageCausalConv3d) for m in self.decoder.modules()
        )
        self._enc_conv_count = sum(
            isinstance(m, QwenImageCausalConv3d) for m in self.encoder.modules()
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        """Reset decode and encode feature caches."""
        self._conv_idx = [0]
        self._feat_map: list = [None] * self._dec_conv_count
        self._enc_conv_idx = [0]
        self._enc_feat_map: list = [None] * self._enc_conv_count

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frame, height, width = x.shape
        if (
            self.use_tiling
            and (width > self.tile_sample_min_width or height > self.tile_sample_min_height)
        ):
            return self._tiled_encode(x)

        self._init_cache()
        iter_ = 1 + (num_frame - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            enc = self.encoder(chunk, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            out = enc if out is None else torch.cat([out, enc], dim=2)
        out = self.quant_conv(out)
        self._init_cache()
        return out

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """Encode pixel values to a Gaussian distribution over latents."""
        if self.use_slicing and x.shape[0] > 1:
            h = torch.cat([self._encode(s) for s in x.split(1)])
        else:
            h = self._encode(x)
        return DiagonalGaussianDistribution(h)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        _, _, num_frame, height, width = z.shape
        tile_lat_h = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_lat_w = self.tile_sample_min_width // self.spatial_compression_ratio
        if self.use_tiling and (width > tile_lat_w or height > tile_lat_h):
            return self._tiled_decode(z)

        self._init_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(num_frame):
            self._conv_idx = [0]
            dec = self.decoder(x[:, :, i : i + 1], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = dec if out is None else torch.cat([out, dec], dim=2)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self._init_cache()
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel values in [-1, 1]."""
        if self.use_slicing and z.shape[0] > 1:
            return torch.cat([self._decode(s) for s in z.split(1)])
        return self._decode(z)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels (0 - 1 range) to normalized latents.

        Args:
            pixels: [B, C, H, W] or [B, C, T, H, W] in [0, 1].

        Returns:
            Normalized latent tensor.
        """
        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(2)
        pixels = pixels.to(self.dtype)
        posterior = self.encode(pixels)
        latents = posterior.mode()
        mean = torch.tensor(self.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.z_dim, 1, 1, 1
        )
        inv_std = 1.0 / torch.tensor(self.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.z_dim, 1, 1, 1
        )
        return (latents - mean) * inv_std

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents to pixel values in [0, 1].

        Args:
            latents: Normalized latent tensor (output of encode_pixels_to_latents).

        Returns:
            Pixel tensor in [0, 1].
        """
        latents = latents.to(self.dtype)
        mean = torch.tensor(self.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.z_dim, 1, 1, 1
        )
        inv_std = 1.0 / torch.tensor(self.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.z_dim, 1, 1, 1
        )
        latents = latents / inv_std + mean
        image = self.decode(latents)[:, :, 0]  # take first (only) temporal frame
        return (image * 0.5 + 0.5).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Tiling helpers
    # ------------------------------------------------------------------

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        self.use_tiling = True
        if tile_sample_min_height is not None:
            self.tile_sample_min_height = tile_sample_min_height
        if tile_sample_min_width is not None:
            self.tile_sample_min_width = tile_sample_min_width
        if tile_sample_stride_height is not None:
            self.tile_sample_stride_height = tile_sample_stride_height
        if tile_sample_stride_width is not None:
            self.tile_sample_stride_width = tile_sample_stride_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def _blend_v(self, a: torch.Tensor, b: torch.Tensor, extent: int) -> torch.Tensor:
        extent = min(a.shape[-2], b.shape[-2], extent)
        for y in range(extent):
            b[:, :, :, y, :] = (
                a[:, :, :, -extent + y, :] * (1 - y / extent)
                + b[:, :, :, y, :] * (y / extent)
            )
        return b

    def _blend_h(self, a: torch.Tensor, b: torch.Tensor, extent: int) -> torch.Tensor:
        extent = min(a.shape[-1], b.shape[-1], extent)
        for x in range(extent):
            b[:, :, :, :, x] = (
                a[:, :, :, :, -extent + x] * (1 - x / extent)
                + b[:, :, :, :, x] * (x / extent)
            )
        return b

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape
        lat_h = height // self.spatial_compression_ratio
        lat_w = width // self.spatial_compression_ratio
        tile_lat_stride_h = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_lat_stride_w = self.tile_sample_stride_width // self.spatial_compression_ratio
        blend_h = (self.tile_sample_min_height - self.tile_sample_stride_height) // self.spatial_compression_ratio
        blend_w = (self.tile_sample_min_width - self.tile_sample_stride_width) // self.spatial_compression_ratio

        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self._init_cache()
                time_chunks = []
                n_iter = 1 + (num_frames - 1) // 4
                for k in range(n_iter):
                    self._enc_conv_idx = [0]
                    chunk = (
                        x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                        if k == 0
                        else x[:, :, 1 + 4 * (k - 1) : 1 + 4 * k, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    )
                    tile = self.encoder(chunk, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    tile = self.quant_conv(tile)
                    time_chunks.append(tile)
                row.append(torch.cat(time_chunks, dim=2))
            rows.append(row)
        self._init_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_w)
                result_row.append(tile[:, :, :, :tile_lat_stride_h, :tile_lat_stride_w])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=3)[:, :, :, :lat_h, :lat_w]

    def _tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = z.shape
        sample_h = height * self.spatial_compression_ratio
        sample_w = width * self.spatial_compression_ratio
        tile_lat_stride_h = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_lat_stride_w = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_lat_min_h = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_lat_min_w = self.tile_sample_min_width // self.spatial_compression_ratio
        blend_h = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_w = self.tile_sample_min_width - self.tile_sample_stride_width

        rows = []
        for i in range(0, height, tile_lat_stride_h):
            row = []
            for j in range(0, width, tile_lat_stride_w):
                self._init_cache()
                time_chunks = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_lat_min_h, j : j + tile_lat_min_w]
                    tile = self.post_quant_conv(tile)
                    dec = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                    time_chunks.append(dec)
                row.append(torch.cat(time_chunks, dim=2))
            rows.append(row)
        self._init_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_w)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_h, :sample_w]
        return dec


def load_qwen_image_vae(
    vae_path: str,
    input_channels: int = 3,
    device: Union[str, torch.device] = "cpu",
) -> AutoencoderKLQwenImage:
    """Load a QwenImage VAE from a safetensors checkpoint.

    Args:
        vae_path: Path to the VAE .safetensors file.
        input_channels: 3 for t2i/edit, 4 for layered.
        device: Target device.

    Returns:
        Loaded and eval-mode AutoencoderKLQwenImage.
    """
    import safetensors.torch as st

    logger.info(f"Loading QwenImage VAE from {vae_path}")
    vae = AutoencoderKLQwenImage(input_channels=input_channels)

    sd = st.load_file(vae_path, device=str(device))
    info = vae.load_state_dict(sd, strict=True)
    logger.info(f"Loaded VAE: {info}")
    vae.eval()
    return vae
