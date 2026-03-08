"""HunyuanVideoTransformer3DModelPacked - FramePack's main DiT model.

Ported from Musubi_Tuner/src/musubi_tuner/frame_pack/hunyuan_video_packed.py.
Original code: https://github.com/lllyasviel/FramePack (Apache-2.0)

Key characteristics:
- Always I2V: first frame is conditioning image via CLIP vision projection
- Packed temporal format: 1x/2x/4x clean latents are prepended to noisy tokens
- RoPE for 3D position encoding (temporal + spatial)
- Dual-stream (double blocks) then single-stream (single blocks)
- Block swap supported via ModelOffloader

Porting improvements applied:
- print() -> logger.info()/logger.warning()
- logging.basicConfig() removed
- Dead code removed (teacache left as clean stub, RoPE scaling kept)
- torch.concat -> torch.cat (already correct)
"""
from __future__ import annotations

import glob
import logging
import os
from types import SimpleNamespace
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn

from trainer.arch.framepack.components.blocks import (
    AdaLayerNormContinuous,
    CombinedTimestepGuidanceTextProjEmbeddings,
    HunyuanVideoTransformerBlock,
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTokenRefiner,
)
from trainer.arch.framepack.components.utils import (
    ClipVisionProjection,
    HunyuanVideoPatchEmbed,
    HunyuanVideoPatchEmbedForCleanLatents,
    HunyuanVideoRotaryPosEmbed,
    get_cu_seqlens,
    pad_for_3d_conv,
    center_down_sample_3d,
)

logger = logging.getLogger(__name__)

# Keys targeted for FP8 optimization (match Musubi_Tuner convention)
FP8_OPTIMIZATION_TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm"]


class HunyuanVideoTransformer3DModelPacked(nn.Module):
    """FramePack: HunyuanVideo DiT with packed multi-scale temporal context.

    Architecture:
    - x_embedder: 3D patch embed for noisy latents
    - clean_x_embedder: multi-scale patch embed for 1x/2x/4x context latents
    - image_projection: CLIP vision projection for I2V conditioning
    - context_embedder: text token refiner (LLaMA + CLIP-L -> DiT tokens)
    - time_text_embed: timestep + guidance + pooled text conditioning
    - rope: 3D RoPE for image tokens
    - transformer_blocks: N double-stream blocks (joint attn over img+text)
    - single_transformer_blocks: M single-stream blocks (concatenated attn)
    - norm_out + proj_out: output projection back to latent space
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, int, int] = (16, 56, 56),
        image_proj_dim: int = 1152,
        attn_mode: Optional[str] = None,
        split_attn: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.config_patch_size = patch_size
        self.config_patch_size_t = patch_size_t
        self.inner_dim = inner_dim

        # 1. Latent and conditioning embedders
        self.x_embedder = HunyuanVideoPatchEmbed(
            patch_size=(patch_size_t, patch_size, patch_size),
            in_chans=in_channels,
            embed_dim=inner_dim,
        )
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim,
            num_layers=num_refiner_layers,
        )
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            inner_dim, pooled_projection_dim
        )
        # Multi-scale clean latent embedder (always installed for FramePack)
        self.clean_x_embedder = HunyuanVideoPatchEmbedForCleanLatents(inner_dim)
        # CLIP vision projection for I2V conditioning (always installed)
        self.image_projection = ClipVisionProjection(in_channels=image_proj_dim, out_channels=inner_dim)

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(rope_axes_dim, rope_theta)

        # 3. Transformer blocks
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.transformer_blocks = nn.ModuleList([
            HunyuanVideoTransformerBlock(
                num_attention_heads,
                attention_head_dim,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                attn_mode=attn_mode,
                split_attn=split_attn,
            )
            for _ in range(num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            HunyuanVideoSingleTransformerBlock(
                num_attention_heads,
                attention_head_dim,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                attn_mode=attn_mode,
                split_attn=split_attn,
            )
            for _ in range(num_single_layers)
        ])

        # 4. Output projection
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            inner_dim, patch_size_t * patch_size * patch_size * out_channels
        )

        # Training flags
        self.use_gradient_checkpointing: bool = False
        self.high_quality_fp32_output_for_inference: bool = True

        # Block swap state (None = disabled)
        self.blocks_to_swap: Optional[int] = None
        self.offloader_double = None
        self.offloader_single = None

        # RoPE scaling (disabled by default)
        self.rope_scaling_timestep_threshold: Optional[int] = None
        self.rope_scaling_factor: float = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        self.use_gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled for HunyuanVideoTransformer3DModelPacked.")

    def disable_gradient_checkpointing(self) -> None:
        self.use_gradient_checkpointing = False
        logger.info("Gradient checkpointing disabled for HunyuanVideoTransformer3DModelPacked.")

    def _checkpoint(self, block, *args):
        if self.use_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    # ------------------------------------------------------------------
    # Block swap
    # ------------------------------------------------------------------

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ) -> None:
        from trainer.arch.wan.components.utils import ModelOffloader  # deferred; shared utility

        self.blocks_to_swap = num_blocks
        n_double = len(self.transformer_blocks)
        n_single = len(self.single_transformer_blocks)
        double_swap = num_blocks // 2
        single_swap = (num_blocks - double_swap) * 2 + 1

        assert double_swap <= n_double - 1 and single_swap <= n_single - 1, (
            f"Cannot swap {double_swap} double + {single_swap} single blocks. "
            f"Limits: {n_double - 1} double, {n_single - 1} single."
        )

        self.offloader_double = ModelOffloader(
            "double", self.transformer_blocks, n_double, double_swap,
            supports_backward, device, use_pinned_memory,
        )
        self.offloader_single = ModelOffloader(
            "single", self.single_transformer_blocks, n_single, single_swap,
            supports_backward, device, use_pinned_memory,
        )
        logger.info(
            f"Block swap enabled: {num_blocks} total "
            f"({double_swap} double, {single_swap} single), "
            f"supports_backward={supports_backward}."
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        """Move all model components to device, leaving swapped blocks on CPU."""
        if self.blocks_to_swap:
            saved_double = self.transformer_blocks
            saved_single = self.single_transformer_blocks
            self.transformer_blocks = None
            self.single_transformer_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.transformer_blocks = saved_double
            self.single_transformer_blocks = saved_single

    def prepare_block_swap_before_forward(self) -> None:
        """Prepare block device placement before each forward pass."""
        if not self.blocks_to_swap:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.transformer_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_transformer_blocks)

    # ------------------------------------------------------------------
    # RoPE scaling
    # ------------------------------------------------------------------

    def enable_rope_scaling(
        self,
        timestep_threshold: Optional[int],
        rope_scaling_factor: float = 1.0,
    ) -> None:
        if timestep_threshold is not None and rope_scaling_factor > 0:
            self.rope_scaling_timestep_threshold = timestep_threshold
            self.rope_scaling_factor = rope_scaling_factor
            logger.info(
                f"RoPE scaling enabled: threshold={timestep_threshold}, factor={rope_scaling_factor}."
            )
        else:
            self.rope_scaling_timestep_threshold = None
            self.rope_scaling_factor = 1.0
            self.rope.h_w_scaling_factor = 1.0
            logger.info("RoPE scaling disabled.")

    # ------------------------------------------------------------------
    # Input processing
    # ------------------------------------------------------------------

    def process_input_hidden_states(
        self,
        latents: torch.Tensor,
        latent_indices: Optional[torch.Tensor] = None,
        clean_latents: Optional[torch.Tensor] = None,
        clean_latent_indices: Optional[torch.Tensor] = None,
        clean_latents_2x: Optional[torch.Tensor] = None,
        clean_latent_2x_indices: Optional[torch.Tensor] = None,
        clean_latents_4x: Optional[torch.Tensor] = None,
        clean_latent_4x_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed noisy latents and all clean context latents, build RoPE freqs.

        Returns:
            hidden_states: [B, L_total, inner_dim]
            rope_freqs: [B, L_total, rope_dim]
        """
        hidden_states = self._checkpoint(self.x_embedder.proj, latents)
        b, c, t, h, w = hidden_states.shape

        if latent_indices is None:
            latent_indices = torch.arange(0, t).unsqueeze(0).expand(b, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        rope_freqs = self.rope(frame_indices=latent_indices, height=h, width=w, device=hidden_states.device)
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2)

        # Prepend 1x clean latents
        if clean_latents is not None and clean_latent_indices is not None:
            cl = clean_latents.to(hidden_states)
            cl = self._checkpoint(self.clean_x_embedder.proj, cl)
            cl = cl.flatten(2).transpose(1, 2)
            cl_rope = self.rope(frame_indices=clean_latent_indices, height=h, width=w, device=cl.device)
            cl_rope = cl_rope.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([cl, hidden_states], dim=1)
            rope_freqs = torch.cat([cl_rope, rope_freqs], dim=1)

        # Prepend 2x clean latents
        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            cl2 = clean_latents_2x.to(hidden_states)
            cl2 = pad_for_3d_conv(cl2, (2, 4, 4))
            cl2 = self._checkpoint(self.clean_x_embedder.proj_2x, cl2)
            cl2 = cl2.flatten(2).transpose(1, 2)
            cl2_rope = self.rope(frame_indices=clean_latent_2x_indices, height=h, width=w, device=cl2.device)
            cl2_rope = pad_for_3d_conv(cl2_rope, (2, 2, 2))
            cl2_rope = center_down_sample_3d(cl2_rope, (2, 2, 2))
            cl2_rope = cl2_rope.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([cl2, hidden_states], dim=1)
            rope_freqs = torch.cat([cl2_rope, rope_freqs], dim=1)

        # Prepend 4x clean latents
        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            cl4 = clean_latents_4x.to(hidden_states)
            cl4 = pad_for_3d_conv(cl4, (4, 8, 8))
            cl4 = self._checkpoint(self.clean_x_embedder.proj_4x, cl4)
            cl4 = cl4.flatten(2).transpose(1, 2)
            cl4_rope = self.rope(frame_indices=clean_latent_4x_indices, height=h, width=w, device=cl4.device)
            cl4_rope = pad_for_3d_conv(cl4_rope, (4, 4, 4))
            cl4_rope = center_down_sample_3d(cl4_rope, (4, 4, 4))
            cl4_rope = cl4_rope.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([cl4, hidden_states], dim=1)
            rope_freqs = torch.cat([cl4_rope, rope_freqs], dim=1)

        return hidden_states, rope_freqs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        latent_indices: Optional[torch.Tensor] = None,
        clean_latents: Optional[torch.Tensor] = None,
        clean_latent_indices: Optional[torch.Tensor] = None,
        clean_latents_2x: Optional[torch.Tensor] = None,
        clean_latent_2x_indices: Optional[torch.Tensor] = None,
        clean_latents_4x: Optional[torch.Tensor] = None,
        clean_latent_4x_indices: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """Forward pass through the FramePack transformer.

        Args:
            hidden_states: [B, C, T, H, W] noisy latents
            timestep: [B] timestep in [0, 1000]
            encoder_hidden_states: [B, L_text, text_dim] LLaMA features
            encoder_attention_mask: [B, L_text] attention mask
            pooled_projections: [B, clip_dim] CLIP-L pooler output
            guidance: [B] embedded guidance scale
            latent_indices: [B, T] optional temporal position indices
            clean_latents: [B, 16, T1, H, W] 1x clean context latents
            clean_latent_indices: [B, T1] temporal indices for 1x latents
            clean_latents_2x: [B, 16, T2, H, W] 2x context latents
            clean_latent_2x_indices: [B, T2] temporal indices for 2x latents
            clean_latents_4x: [B, 16, T4, H, W] 4x context latents
            clean_latent_4x_indices: [B, T4] temporal indices for 4x latents
            image_embeddings: [B, L_img, 1152] SigLIP image features
            return_dict: whether to return a dict/namespace or a tuple

        Returns:
            predicted noise (same shape as hidden_states), as namespace or tuple
        """
        # RoPE scaling (optional)
        if self.rope_scaling_timestep_threshold is not None:
            if (timestep >= self.rope_scaling_timestep_threshold).any():
                self.rope.h_w_scaling_factor = self.rope_scaling_factor
            else:
                self.rope.h_w_scaling_factor = 1.0

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config_patch_size, self.config_patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        input_device = hidden_states.device

        # 1. Embed latents + context + compute RoPE
        hidden_states, rope_freqs = self.process_input_hidden_states(
            hidden_states,
            latent_indices, clean_latents, clean_latent_indices,
            clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices,
        )
        del latent_indices, clean_latents, clean_latent_indices
        del clean_latents_2x, clean_latent_2x_indices
        del clean_latents_4x, clean_latent_4x_indices

        # 2. Compute conditioning embedding (timestep + guidance + pooled text)
        temb = self._checkpoint(self.time_text_embed, timestep, guidance, pooled_projections)

        # 3. Refine text tokens
        encoder_hidden_states = self._checkpoint(
            self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask
        )

        # 4. Prepend image embeddings from CLIP vision encoder
        if self.image_projection is not None and image_embeddings is not None:
            extra_enc = self._checkpoint(self.image_projection, image_embeddings)
            extra_mask = torch.ones(
                (batch_size, extra_enc.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
            # image tokens must come BEFORE text tokens (attention mask ordering)
            encoder_hidden_states = torch.cat([extra_enc, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([extra_mask, encoder_attention_mask], dim=1)
            del extra_enc, extra_mask

        # 5. Build attention mask for variable-length sequences
        with torch.no_grad():
            if batch_size == 1 and not self.split_attn:
                # For batch=1, crop is numerically identical to masked attention
                text_len = int(encoder_attention_mask.sum().item())
                encoder_hidden_states = encoder_hidden_states[:, :text_len]
                attention_mask = (None, None, None, None, None)
            else:
                img_seq_len = hidden_states.shape[1]
                txt_seq_len = encoder_hidden_states.shape[1]
                cu_seqlens_q, seq_len = get_cu_seqlens(encoder_attention_mask, img_seq_len)
                cu_seqlens_kv = cu_seqlens_q
                max_seqlen_q = img_seq_len + txt_seq_len
                max_seqlen_kv = max_seqlen_q
                attention_mask = (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len)
                del cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, seq_len
        del encoder_attention_mask

        # 6. Double-stream transformer blocks
        for block_id, block in enumerate(self.transformer_blocks):
            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(block_id)

            hidden_states, encoder_hidden_states = self._checkpoint(
                block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs
            )

            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks_forward(self.transformer_blocks, block_id)

        # 7. Single-stream transformer blocks
        for block_id, block in enumerate(self.single_transformer_blocks):
            if self.blocks_to_swap:
                self.offloader_single.wait_for_block(block_id)

            hidden_states, encoder_hidden_states = self._checkpoint(
                block, hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs
            )

            if self.blocks_to_swap:
                self.offloader_single.submit_move_blocks_forward(self.single_transformer_blocks, block_id)

        del attention_mask, rope_freqs, encoder_hidden_states

        # 8. Output projection
        hidden_states = self._checkpoint(self.norm_out, hidden_states, temb)

        # Slice to only the original (non-context) tokens
        hidden_states = hidden_states[:, -original_context_length:, :]

        if self.high_quality_fp32_output_for_inference and not self.training:
            hidden_states = hidden_states.to(dtype=torch.float32)
            if self.proj_out.weight.dtype != torch.float32:
                self.proj_out.to(dtype=torch.float32)

        hidden_states = self._checkpoint(self.proj_out, hidden_states)

        if hidden_states.device != input_device:
            hidden_states = hidden_states.to(input_device)

        # 9. Reshape from [B, T*H*W, C*pt*ph*pw] -> [B, C, T*pt, H*ph, W*pw]
        hidden_states = einops.rearrange(
            hidden_states,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=post_patch_num_frames,
            h=post_patch_height,
            w=post_patch_width,
            pt=p_t,
            ph=p,
            pw=p,
        )

        if return_dict:
            return SimpleNamespace(sample=hidden_states)
        return (hidden_states,)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_packed_model(
    device: torch.device | str,
    dit_path: str,
    attn_mode: str,
    loading_device: torch.device | str,
    fp8_scaled: bool = False,
    split_attn: bool = False,
    disable_numpy_memmap: bool = False,
) -> HunyuanVideoTransformer3DModelPacked:
    """Load a FramePack packed DiT model from a checkpoint path.

    Args:
        device: Device for computation (usually "cuda").
        dit_path: Path to safetensors file or directory containing one.
        attn_mode: Attention backend ("sdpa", "flash", "xformers", "sageattn").
        loading_device: Device to load weights to (cpu when block swapping).
        fp8_scaled: Apply FP8 weight optimization.
        split_attn: Use per-sample split attention (for batch > 1 without varlen).
        disable_numpy_memmap: Disable numpy memory mapping when loading.

    Returns:
        Loaded HunyuanVideoTransformer3DModelPacked.
    """
    from accelerate import init_empty_weights
    from trainer.arch.wan.components.utils import load_safetensors_with_lora_and_fp8  # shared loading util

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # Resolve directory to first .safetensors file
    if os.path.isdir(dit_path):
        safetensor_files = sorted(glob.glob(os.path.join(dit_path, "*.safetensors")))
        if not safetensor_files:
            raise ValueError(f"No .safetensors files found in {dit_path}")
        dit_path = safetensor_files[0]

    logger.info("Creating HunyuanVideoTransformer3DModelPacked")

    with init_empty_weights():
        model = HunyuanVideoTransformer3DModelPacked(
            attention_head_dim=128,
            guidance_embeds=True,
            image_proj_dim=1152,
            in_channels=16,
            mlp_ratio=4.0,
            num_attention_heads=24,
            num_layers=20,
            num_refiner_layers=2,
            num_single_layers=40,
            out_channels=16,
            patch_size=2,
            patch_size_t=1,
            pooled_projection_dim=768,
            qk_norm="rms_norm",
            rope_axes_dim=(16, 56, 56),
            rope_theta=256.0,
            text_embed_dim=4096,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )

    logger.info(f"Loading DiT weights from {dit_path}, target device={loading_device}")

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
    )

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded DiT model: {info}")

    return model
