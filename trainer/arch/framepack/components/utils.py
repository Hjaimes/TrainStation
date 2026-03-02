"""FramePack temporal packing utilities and RoPE helpers.

Ported from Musubi_Tuner frame_pack/hunyuan_video_packed.py and frame_pack/utils.py.
Self-contained — no imports from any other architecture module.

Porting improvements applied:
- print() -> logger.info()/logger.warning()
- logging.basicConfig() removed
- Dead code and debug prints removed
- torch.concat -> torch.cat (already correct in source)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3D convolution padding helpers
# ---------------------------------------------------------------------------

def pad_for_3d_conv(x: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """Pad a 5D tensor [B, C, T, H, W] so all dims are divisible by kernel_size."""
    _, _, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """3D average-pool downsampling (equivalent to center-crop subsampling)."""
    return F.avg_pool3d(x, kernel_size, stride=kernel_size)


# ---------------------------------------------------------------------------
# Sequence length helpers for variable-length attention
# ---------------------------------------------------------------------------

def get_cu_seqlens(text_mask: torch.Tensor, img_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cumulative sequence lengths for variable-length attention.

    Args:
        text_mask: [B, L_text] boolean/int mask (1 = valid token)
        img_len: number of image tokens (same for all items in batch)

    Returns:
        cu_seqlens: [2*B + 1] cumulative sequence lengths (int32)
        seq_len: [B] per-sample sequence lengths
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros(
        [2 * batch_size + 1], dtype=torch.int32, device=text_mask.device
    )

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    seq_len = text_len + img_len
    return cu_seqlens, seq_len


# ---------------------------------------------------------------------------
# RoPE positional embedding
# ---------------------------------------------------------------------------

class HunyuanVideoRotaryPosEmbed(nn.Module):
    """Rotary position embeddings for FramePack's 3D latent space.

    Computes per-frame RoPE frequencies for temporal (T), height (H), width (W)
    axes. Supports multi-frame indexing for packed temporal context.
    """

    def __init__(self, rope_dim: tuple[int, int, int], theta: float):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        self.h_w_scaling_factor: float = 1.0

    @torch.no_grad()
    def _get_frequency(self, dim: int, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine frequencies for one axis.

        Args:
            dim: frequency dimension for this axis
            pos: [T, H, W] position grid

        Returns:
            (cos, sin) each of shape [dim, T, H, W]
        """
        t, h, w = pos.shape
        freqs = 1.0 / (
            self.theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[: (dim // 2)] / dim
            )
        )
        # outer product then reshape to (dim, T, H, W); repeat_interleave converts (d/2) -> d
        freqs = torch.outer(freqs, pos.reshape(-1)).unflatten(-1, (t, h, w)).repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def _forward_inner(
        self,
        frame_indices: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute RoPE freqs for a single sample's frame indices.

        Args:
            frame_indices: [T] 1D tensor of frame positions
            height, width: spatial patch dimensions (post-patch)
            device: target device

        Returns:
            Tensor of shape [total_dim, T, H, W] where total_dim = 2*(DT+DY+DX)
        """
        gt, gy, gx = torch.meshgrid(
            frame_indices.to(device=device, dtype=torch.float32),
            torch.arange(0, height, device=device, dtype=torch.float32) * self.h_w_scaling_factor,
            torch.arange(0, width, device=device, dtype=torch.float32) * self.h_w_scaling_factor,
            indexing="ij",
        )

        fct, fst = self._get_frequency(self.DT, gt)
        del gt
        fcy, fsy = self._get_frequency(self.DY, gy)
        del gy
        fcx, fsx = self._get_frequency(self.DX, gx)
        del gx

        result = torch.cat([fct, fcy, fcx, fst, fsy, fsx], dim=0)
        del fct, fcy, fcx, fst, fsy, fsx
        return result

    @torch.no_grad()
    def forward(
        self,
        frame_indices: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute batched RoPE frequencies.

        Args:
            frame_indices: [B, T] — each row is one sample's frame positions
            height, width: post-patch spatial dimensions
            device: target device

        Returns:
            [B, total_dim, T, H, W]
        """
        samples = frame_indices.unbind(0)
        results = [self._forward_inner(f, height, width, device) for f in samples]
        return torch.stack(results, dim=0)


# ---------------------------------------------------------------------------
# Crop/pad helper for text tokens
# ---------------------------------------------------------------------------

def crop_or_pad_yield_mask(x: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop or zero-pad the frame dimension of x to exactly 'length' tokens.

    Args:
        x: [B, F, C] text embedding tensor
        length: target sequence length

    Returns:
        (padded_x, mask) — mask is [B, length] bool, True where content exists
    """
    b, f, c = x.shape
    device = x.device
    dtype = x.dtype

    if f < length:
        y = torch.zeros((b, length, c), dtype=dtype, device=device)
        mask = torch.zeros((b, length), dtype=torch.bool, device=device)
        y[:, :f, :] = x
        mask[:, :f] = True
        return y, mask

    return x[:, :length, :], torch.ones((b, length), dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# Patch embedding helpers
# ---------------------------------------------------------------------------

class HunyuanVideoPatchEmbed(nn.Module):
    """3D patch embedding for noisy latents."""
    def __init__(self, patch_size: tuple[int, int, int], in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HunyuanVideoPatchEmbedForCleanLatents(nn.Module):
    """Multi-scale patch embedding for clean (context) latents.

    Three projections at 1x, 2x, and 4x spatial downsampling to handle the
    packed temporal context format.
    """
    def __init__(self, inner_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer: nn.Conv3d) -> None:
        """Initialize multi-scale weights from the main patch embed weights."""
        import einops
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        sd = {
            "proj.weight": weight.clone(),
            "proj.bias": bias.clone(),
            "proj_2x.weight": einops.repeat(
                weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=2, hk=2, wk=2
            ) / 8.0,
            "proj_2x.bias": bias.clone(),
            "proj_4x.weight": einops.repeat(
                weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=4, hk=4, wk=4
            ) / 64.0,
            "proj_4x.bias": bias.clone(),
        }
        self.load_state_dict({k: v.clone() for k, v in sd.items()})


# ---------------------------------------------------------------------------
# CLIP vision projection
# ---------------------------------------------------------------------------

class ClipVisionProjection(nn.Module):
    """Two-layer MLP to project SigLIP image embeddings into DiT token space."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))
