"""SD3 MMDiT (Multi-Modal Diffusion Transformer) model.

Implements SD3Transformer2DModel with:
- PatchEmbed for 16-channel latents
- JointTransformerBlocks (bidirectional image+text attention)
- SD3SingleTransformerBlocks (image-only, SD3.5 only)
- AdaLayerNormContinuous final norm
- Output projection + unpatchify
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from trainer.arch.sd3.components.blocks import JointTransformerBlock, SD3SingleTransformerBlock
from trainer.arch.sd3.components.embeddings import PatchEmbed, CombinedTimestepTextProjEmbeddings
from trainer.arch.sd3.components.layers import AdaLayerNormContinuous

if TYPE_CHECKING:
    from trainer.arch.sd3.components.configs import SD3Config

logger = logging.getLogger(__name__)


class SD3Transformer2DModel(nn.Module):
    """MMDiT transformer for SD3 image generation.

    Architecture (per SD3 paper / reference implementation):
    1. PatchEmbed: (B, 16, H, W) -> (B, N, D)  where N = (H/p)*(W/p)
    2. Context projector: T5 embeddings (B, L, 4096) -> (B, L, D)
    3. Timestep+pooled conditioning: (B,) + (B, 2048) -> (B, D)
    4. N joint transformer blocks (bidirectional image+text attention)
    5. M single transformer blocks (image only, SD3.5 variants)
    6. Final AdaLayerNormContinuous
    7. Output projection: (B, N, D) -> (B, N, p²×16)
    8. Unpatchify: -> (B, 16, H, W)
    """

    def __init__(
        self,
        num_layers: int,
        num_single_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        patch_size: int,
        latent_channels: int,
        pooled_projection_dim: int,
        caption_projection_dim: int,
        joint_attention_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.gradient_checkpointing = False

        # Latent -> patch tokens
        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=latent_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        # T5 text embeddings -> model hidden size
        self.context_embedder = nn.Linear(joint_attention_dim, hidden_size)

        # Timestep + pooled CLIP conditioning
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size,
            pooled_projection_dim=pooled_projection_dim,
        )

        # Joint (image+text) transformer blocks
        # Last block uses context_pre_only=True to skip text MLP after final joint block
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                context_pre_only=(i == num_layers - 1),
            )
            for i in range(num_layers)
        ])

        # Single (image-only) transformer blocks for SD3.5 variants
        self.single_transformer_blocks = nn.ModuleList([
            SD3SingleTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
            )
            for _ in range(num_single_layers)
        ])

        # Final norm + output projection
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_dim=hidden_size,
        )
        self.proj_out = nn.Linear(
            hidden_size,
            patch_size * patch_size * latent_channels,
            bias=True,
        )

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory savings during training."""
        self.gradient_checkpointing = True

    def _forward_joint_block(
        self,
        block: JointTransformerBlock,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single joint block forward, with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                use_reentrant=False,
            )
        return block(hidden_states, encoder_hidden_states, temb)

    def _forward_single_block(
        self,
        block: SD3SingleTransformerBlock,
        hidden_states: Tensor,
        temb: Tensor,
    ) -> Tensor:
        """Single image-only block forward, with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                block,
                hidden_states,
                temb,
                use_reentrant=False,
            )
        return block(hidden_states, temb)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        pooled_projections: Tensor,
    ) -> Tensor:
        """
        Args:
            hidden_states:         (B, 16, H, W) — noisy latents
            encoder_hidden_states: (B, L, 4096)  — T5-XXL text embeddings
            timestep:              (B,)           — in [0, 1000]
            pooled_projections:    (B, 2048)      — CLIP-L + CLIP-G pooled embeddings

        Returns:
            (B, 16, H, W) — predicted velocity (noise - x0)
        """
        B, C, H, W = hidden_states.shape

        # 1. Patch embed: (B, 16, H, W) -> (B, N, D)
        hidden_states = self.pos_embed(hidden_states)

        # 2. Project T5 context: (B, L, 4096) -> (B, L, D)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Timestep + pooled conditioning: (B, D)
        temb = self.time_text_embed(timestep, pooled_projections)

        # 4. Joint transformer blocks (bidirectional image+text)
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._forward_joint_block(
                block, hidden_states, encoder_hidden_states, temb
            )

        # 5. Single transformer blocks (image only, SD3.5 variants)
        for block in self.single_transformer_blocks:
            hidden_states = self._forward_single_block(block, hidden_states, temb)

        # 6. Final norm with timestep conditioning
        hidden_states = self.norm_out(hidden_states, temb)

        # 7. Output projection: (B, N, D) -> (B, N, p²×16)
        hidden_states = self.proj_out(hidden_states)

        # 8. Unpatchify: (B, N, p²×C) -> (B, C, H, W)
        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        # hidden_states: (B, h_patches*w_patches, p*p*C)
        hidden_states = hidden_states.view(B, h_patches, w_patches, p, p, C)
        # Rearrange: (B, h, w, p, p, C) -> (B, C, h*p, w*p)
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4).contiguous()
        hidden_states = hidden_states.view(B, C, H, W)

        return hidden_states


def load_sd3_model(
    config: "SD3Config",
    device: torch.device,
    path: str,
    dtype: torch.dtype,
) -> SD3Transformer2DModel:
    """Load SD3 transformer from safetensors checkpoint.

    Creates the model from config, loads weights from safetensors file,
    and moves to the target device and dtype.

    Args:
        config: SD3Config specifying architecture dimensions
        device: Target device for the model
        path:   Path to safetensors checkpoint
        dtype:  Training dtype (bfloat16, float16, float32)

    Returns:
        SD3Transformer2DModel with loaded weights
    """
    import safetensors.torch

    model = SD3Transformer2DModel(
        num_layers=config.num_layers,
        num_single_layers=config.num_single_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        patch_size=config.patch_size,
        latent_channels=config.latent_channels,
        pooled_projection_dim=config.pooled_projection_dim,
        caption_projection_dim=config.caption_projection_dim,
        joint_attention_dim=config.joint_attention_dim,
    )

    logger.info(f"Loading SD3 model from {path}")
    state_dict = safetensors.torch.load_file(path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device=device, dtype=dtype)
    logger.info(f"SD3 model loaded: {config.name}, device={device}, dtype={dtype}")
    return model
