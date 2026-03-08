"""HunyuanVideoTransformer3DModel - main transformer backbone.

Ported from Musubi_Tuner's hunyuan_model/models.py (HYVideoDiffusionTransformer).
Improvements:
  - Removed print() calls, replaced with logger
  - Removed logging.basicConfig()
  - Removed dead/commented-out code
  - Cleaner import structure
"""
from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional, Tuple, Union

import accelerate
import torch
import torch.nn as nn

from .attention import attention, get_cu_seqlens
from .blocks import MMDoubleStreamBlock, MMSingleStreamBlock
from .configs import HunyuanVideoConfig
from .embeddings import (
    PatchEmbed,
    SingleTokenRefiner,
    TextProjection,
    TimestepEmbedder,
    get_rotary_pos_embed_by_shape,
)
from .layers import FinalLayer, MLPEmbedder, get_activation_layer
from .offloading import ModelOffloader, _clean_memory_on_device, _synchronize_device

logger = logging.getLogger(__name__)


class HunyuanVideoTransformer3DModel(nn.Module):
    """HunyuanVideo transformer backbone (HYVideo-T/2-cfgdistill variant).

    Architecture:
      - 20 MMDoubleStreamBlocks (joint image+text attention)
      - 40 MMSingleStreamBlocks (merged image+text single stream)
      - Guidance embedding for CFG-distilled inference
      - Dual ModelOffloader for consumer-GPU block swapping
    """

    def __init__(
        self,
        text_states_dim: int,
        text_states_dim_2: int,
        patch_size: List[int] = [1, 2, 2],
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attn_mode: str = "flash",
        split_attn: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"sum(rope_dim_list)={sum(rope_dim_list)} != head_dim={pe_dim}"
            )

        logger.info(
            "HunyuanVideoTransformer3DModel: attn_mode=%s split_attn=%s",
            attn_mode, split_attn,
        )

        silu = get_activation_layer("silu")

        # Patch embedding
        self.img_in = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs)

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim, self.hidden_size, silu, **factory_kwargs
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection!r}")

        # Timestep embedding
        self.time_in = TimestepEmbedder(self.hidden_size, silu, **factory_kwargs)

        # CLIP pooled embedding
        self.vector_in = MLPEmbedder(text_states_dim_2, self.hidden_size, **factory_kwargs)

        # Guidance embedding (CFG-distilled variant)
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, silu, **factory_kwargs)
            if guidance_embed else None
        )

        # Double-stream blocks (20)
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                self.hidden_size, self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                attn_mode=attn_mode,
                split_attn=split_attn,
                **factory_kwargs,
            )
            for _ in range(mm_double_blocks_depth)
        ])

        # Single-stream blocks (40)
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                self.hidden_size, self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                attn_mode=attn_mode,
                split_attn=split_attn,
                **factory_kwargs,
            )
            for _ in range(mm_single_blocks_depth)
        ])

        # Final projection
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels,
            silu, **factory_kwargs,
        )

        # Block swap state
        self.blocks_to_swap: Optional[int] = None
        self.offloader_double: Optional[ModelOffloader] = None
        self.offloader_single: Optional[ModelOffloader] = None
        self._enable_img_in_txt_in_offloading = False
        self.gradient_checkpointing = False

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # -----------------------------------------------------------------------
    # Gradient checkpointing
    # -----------------------------------------------------------------------

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        if hasattr(self.txt_in, "enable_gradient_checkpointing"):
            self.txt_in.enable_gradient_checkpointing()
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        logger.info(
            "HunyuanVideoTransformer3DModel: gradient checkpointing enabled "
            "(activation_cpu_offloading=%s)", activation_cpu_offloading
        )

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        if hasattr(self.txt_in, "disable_gradient_checkpointing"):
            self.txt_in.disable_gradient_checkpointing()
        for block in list(self.double_blocks) + list(self.single_blocks):
            block.disable_gradient_checkpointing()
        logger.info("HunyuanVideoTransformer3DModel: gradient checkpointing disabled")

    # -----------------------------------------------------------------------
    # Block swap
    # -----------------------------------------------------------------------

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ) -> None:
        """Set up dual ModelOffloader for double and single stream blocks."""
        self.blocks_to_swap = num_blocks
        n_double = len(self.double_blocks)
        n_single = len(self.single_blocks)
        double_to_swap = num_blocks // 2
        single_to_swap = (num_blocks - double_to_swap) * 2 + 1

        if double_to_swap >= n_double or single_to_swap >= n_single:
            raise ValueError(
                f"Cannot swap {double_to_swap} double and {single_to_swap} single blocks - "
                f"max is {n_double - 1} and {n_single - 1} respectively."
            )

        self.offloader_double = ModelOffloader(
            "double", list(self.double_blocks), n_double, double_to_swap,
            supports_backward, device, use_pinned_memory,
        )
        self.offloader_single = ModelOffloader(
            "single", list(self.single_blocks), n_single, single_to_swap,
            supports_backward, device, use_pinned_memory,
        )
        logger.info(
            "HunyuanVideoTransformer3DModel: block swap enabled - "
            "%d total (%d double, %d single)", num_blocks, double_to_swap, single_to_swap,
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        """Move everything except the swap blocks to device (reduces peak memory)."""
        if self.blocks_to_swap:
            saved_double = self.double_blocks
            saved_single = self.single_blocks
            self.double_blocks = None  # type: ignore[assignment]
            self.single_blocks = None  # type: ignore[assignment]

        self.to(device)

        if self.blocks_to_swap:
            self.double_blocks = saved_double
            self.single_blocks = saved_single

    def prepare_block_swap_before_forward(self) -> None:
        if not self.blocks_to_swap:
            return
        self.offloader_double.prepare_block_devices_before_forward(list(self.double_blocks))
        self.offloader_single.prepare_block_devices_before_forward(list(self.single_blocks))

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(True)
            self.offloader_single.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            logger.info("HunyuanVideoTransformer3DModel: block swap set to forward-only")

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(False)
            self.offloader_single.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            logger.info("HunyuanVideoTransformer3DModel: block swap set to forward+backward")

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt = ot // self.patch_size[0]
        th = oh // self.patch_size[1]
        tw = ow // self.patch_size[2]

        # Build conditioning vector
        vec = self.time_in(t)
        vec = vec + self.vector_in(text_states_2)

        if self.guidance_embed:
            if guidance is None:
                raise ValueError("guidance is required for guidance-distilled model")
            vec = vec + self.guidance_in(guidance)

        # Optionally offload img_in / txt_in during embedding step
        if self._enable_img_in_txt_in_offloading:
            self.img_in.to(x.device, non_blocking=True)
            self.txt_in.to(x.device, non_blocking=True)
            _synchronize_device(x.device)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection!r}")

        if self._enable_img_in_txt_in_offloading:
            self.img_in.to(torch.device("cpu"), non_blocking=True)
            self.txt_in.to(torch.device("cpu"), non_blocking=True)
            _synchronize_device(x.device)
            _clean_memory_on_device(x.device)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_seqlens for varlen flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        attn_mask = total_len = None
        if self.split_attn or self.attn_mode == "torch":
            text_len = text_mask.sum(dim=1)
            total_len = img_seq_len + text_len

        if self.attn_mode == "torch" and not self.split_attn:
            bs = img.shape[0]
            attn_mask = torch.zeros(
                (bs, 1, max_seqlen_q, max_seqlen_q), dtype=torch.bool, device=text_mask.device
            )
            for i in range(bs):
                attn_mask[i, :, : total_len[i], : total_len[i]] = True
            total_len = None

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Double-stream blocks
        input_device = img.device
        double_blocks = list(self.double_blocks)
        for block_idx, block in enumerate(double_blocks):
            block_args = [
                img, txt, vec,
                attn_mask, total_len,
                cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv,
                freqs_cis,
            ]
            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(block_idx)

            img, txt = block(*block_args)

            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks_forward(double_blocks, block_idx)

        # Merge to single stream
        x = torch.cat((img, txt), dim=1)
        if self.blocks_to_swap:
            del img, txt
            _clean_memory_on_device(x.device)

        # Single-stream blocks
        if self.single_blocks:
            single_blocks = list(self.single_blocks)
            for block_idx, block in enumerate(single_blocks):
                block_args = [
                    x, vec, txt_seq_len,
                    attn_mask, total_len,
                    cu_seqlens_q, cu_seqlens_kv,
                    max_seqlen_q, max_seqlen_kv,
                    freqs_cis,
                ]
                if self.blocks_to_swap:
                    self.offloader_single.wait_for_block(block_idx)

                x = block(*block_args)

                if self.blocks_to_swap:
                    self.offloader_single.submit_move_blocks_forward(single_blocks, block_idx)

        img = x[:, :img_seq_len]
        x = None
        if img.device != input_device:
            img = img.to(input_device)

        # Final projection + unpatchify
        img = self.final_layer(img, vec)
        img = self._unpatchify(img, tt, th, tw)

        if return_dict:
            return {"x": img}
        return img

    def _unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """Reshape (B, T*H*W, patch_vol*C) -> (B, C, T*pt, H*ph, W*pw)."""
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1], (
            f"Seq length mismatch: {t}*{h}*{w}={t*h*w} != {x.shape[1]}"
        )
        x = x.reshape(x.shape[0], t, h, w, c, pt, ph, pw)
        x = torch.einsum("nthwcopq->nctohpwq", x)
        return x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def load_hunyuan_video_model(
    dit_path: str,
    config: HunyuanVideoConfig,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    loading_device: Optional[torch.device] = None,
) -> HunyuanVideoTransformer3DModel:
    """Load HunyuanVideoTransformer3DModel from a safetensors checkpoint.

    Args:
        dit_path: Path to the .safetensors model file.
        config: HunyuanVideoConfig with architecture parameters.
        device: Target device for inference/training.
        dtype: Weight dtype (None = auto-detect from checkpoint).
        attn_mode: Attention backend ("torch", "flash", "sageattn", etc.).
        split_attn: Use split-batch attention (for VRAM-constrained setups).
        loading_device: Device to use during weight loading (default: same as device).

    Returns:
        Loaded HunyuanVideoTransformer3DModel.
    """
    from safetensors.torch import load_file

    if loading_device is None:
        loading_device = device

    factor_kwargs = {
        "device": loading_device,
        "dtype": dtype,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
    }

    with accelerate.init_empty_weights():
        model = HunyuanVideoTransformer3DModel(
            text_states_dim=config.text_states_dim,
            text_states_dim_2=config.text_states_dim_2,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_size=config.hidden_size,
            heads_num=config.heads_num,
            mlp_width_ratio=config.mlp_width_ratio,
            mlp_act_type=config.mlp_act_type,
            mm_double_blocks_depth=config.mm_double_blocks_depth,
            mm_single_blocks_depth=config.mm_single_blocks_depth,
            rope_dim_list=config.rope_dim_list,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            qk_norm_type=config.qk_norm_type,
            guidance_embed=config.guidance_embed,
            text_projection=config.text_projection,
            use_attention_mask=config.use_attention_mask,
            **factor_kwargs,
        )

    logger.info("Loading HunyuanVideo model from: %s", dit_path)
    state_dict = load_file(dit_path, device=str(loading_device))
    model.load_state_dict(state_dict, strict=True, assign=True)
    logger.info("HunyuanVideo model loaded successfully")

    return model
