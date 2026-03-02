"""Embedding layers for SD3: patch embedding, timestep sinusoidal, and conditioning MLP."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """2D patch embedding via Conv2d.

    Projects (B, C, H, W) latents into (B, H/p * W/p, embed_dim) token sequences.
    Using Conv2d with stride=patch_size implements non-overlapping patches efficiently.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1536,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            bias=bias,
        )

    def forward(self, latent: Tensor) -> Tensor:
        """
        Args:
            latent: (B, C, H, W)

        Returns:
            (B, H/p * W/p, embed_dim)
        """
        # (B, embed_dim, H/p, W/p) -> flatten spatial -> (B, H/p*W/p, embed_dim)
        return self.proj(latent).flatten(2).transpose(1, 2)


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings (no learnable parameters).

    Standard positional encoding adapted for scalar timesteps.
    Uses cosine and sine of log-spaced frequencies.
    """

    def __init__(self, num_channels: int = 256) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Args:
            timesteps: (B,) — scalar timesteps, typically in [0, 1000]

        Returns:
            (B, num_channels) — sinusoidal embeddings
        """
        half = self.num_channels // 2
        # Log-spaced frequencies: exp(-log(10000) * i / half) for i in [0, half)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        # Outer product: (B, 1) * (1, half) -> (B, half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding: Linear -> SiLU -> Linear."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Timestep embedding + pooled text projection, summed.

    Combines:
    - Sinusoidal timestep encoding -> MLP -> (B, D)
    - Pooled CLIP embeddings -> Linear -> (B, D)
    These are summed to produce the conditioning vector for AdaLayerNorm.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.text_embedder = nn.Linear(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: Tensor, pooled_projection: Tensor) -> Tensor:
        """
        Args:
            timestep:          (B,) — in [0, 1000]
            pooled_projection: (B, pooled_projection_dim) — CLIP-L + CLIP-G pooled embeds

        Returns:
            (B, embedding_dim) — combined conditioning vector
        """
        t_emb = self.timestep_embedder(self.time_proj(timestep))
        p_emb = self.text_embedder(pooled_projection)
        return t_emb + p_emb
