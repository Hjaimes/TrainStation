"""HunyuanVideo 1.5 Diffusion Transformer model.

Self-contained - no imports from hunyuan_video/ or Musubi_Tuner.

Key architecture:
- 54 MMDoubleStreamBlock (NO single-stream blocks)
- patch_size = [1, 1, 1] - no spatial/temporal patching
- guidance_embed = False - no guidance parameter in forward()
- Text encoders: Qwen2.5-VL + ByT5 (cached; not loaded here)
- Block swap: single ModelOffloader for double blocks

Porting improvements:
- print() → logger.info()/logger.warning()
- Removed logging.basicConfig()
- torch.concat → torch.cat
- Cached constants in __init__
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .attention import AttentionParams
from .blocks import FinalLayer, MMDoubleStreamBlock, SingleTokenRefiner
from .configs import HV15_CONFIG, HV15ModelConfig
from .embeddings import ByT5Mapper, PatchEmbed, TimestepEmbedder, VisionProjection, get_nd_rotary_pos_embed
from .offloading import ModelOffloader

logger = logging.getLogger(__name__)


class HunyuanVideo15Transformer(nn.Module):
    """HunyuanVideo 1.5 Diffusion Transformer.

    Args:
        task_type: "t2v" or "i2v".
        attn_mode: Attention backend - "torch", "flash", "sageattn", "xformers".
        split_attn: Use split-attention (processes each batch element separately).
        config: Architecture hyperparameters (defaults to HV15_CONFIG).
    """

    def __init__(
        self,
        task_type: str = "t2v",
        attn_mode: str = "torch",
        split_attn: bool = False,
        config: HV15ModelConfig = HV15_CONFIG,
    ) -> None:
        super().__init__()

        assert task_type in ("t2v", "i2v"), f"Unknown task_type: {task_type!r}"

        self.task_type = task_type
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.cfg = config

        # Expose for external code
        self.patch_size = list(config.patch_size)
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.guidance_embed = config.guidance_embed   # always False
        self.rope_dim_list = list(config.rope_dim_list)
        self.rope_theta = config.rope_theta
        self.hidden_size = config.hidden_size
        self.heads_num = config.heads_num

        # ByT5 character-level text mapper
        self.byt5_in = ByT5Mapper(
            in_dim=config.byt5_dim,
            out_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            out_dim1=config.hidden_size,
            use_residual=False,
        )

        # Image latent patchification
        self.img_in = PatchEmbed(config.patch_size, config.in_channels, config.hidden_size)

        # Vision feature projection (SigLIP → hidden_size, for I2V)
        self.vision_in = VisionProjection(config.vision_states_dim, config.hidden_size)

        # Qwen2.5-VL word-level text refiner
        self.txt_in = SingleTokenRefiner(
            in_channels=config.text_states_dim,
            hidden_size=config.hidden_size,
            heads_num=config.heads_num,
            depth=config.text_refiner_depth,
        )

        # Timestep embedding
        self.time_in = TimestepEmbedder(config.hidden_size, nn.SiLU)

        # Guidance embedding is DISABLED for HV 1.5
        self.guidance_in = None
        self.time_r_in = None

        # 54 double-stream blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                hidden_size=config.hidden_size,
                heads_num=config.heads_num,
                mlp_width_ratio=config.mlp_width_ratio,
                mlp_act_type=config.mlp_act_type,
                qk_norm=config.qk_norm,
                qk_norm_type=config.qk_norm_type,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.num_double_blocks)
        ])

        self.final_layer = FinalLayer(
            config.hidden_size, list(config.patch_size), config.out_channels, nn.SiLU
        )

        # Conditioning type embedding (VL / ByT5 / vision tokens)
        self.cond_type_embedding = nn.Embedding(3, config.hidden_size)

        # Block-swap state (set by enable_block_swap)
        self.blocks_to_swap: int | None = None
        self.offloader_double: ModelOffloader | None = None
        self.num_double_blocks = len(self.double_blocks)

        # Gradient checkpointing state
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

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

    def enable_gradient_checkpointing(self, cpu_offload: bool = False) -> None:
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload
        for block in self.double_blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload)
        logger.info("HV 1.5: gradient checkpointing enabled (cpu_offload=%s)", cpu_offload)

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        for block in self.double_blocks:
            block.disable_gradient_checkpointing()
        logger.info("HV 1.5: gradient checkpointing disabled")

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
        assert num_blocks < self.num_double_blocks - 2, (
            f"Cannot swap {num_blocks} blocks - max is {self.num_double_blocks - 2}"
        )
        self.blocks_to_swap = num_blocks
        self.offloader_double = ModelOffloader(
            block_type="double",
            blocks=list(self.double_blocks),
            num_blocks=self.num_double_blocks,
            blocks_to_swap=num_blocks,
            supports_backward=supports_backward,
            device=device,
            use_pinned_memory=use_pinned_memory,
        )
        logger.info(
            "HV 1.5: block swap enabled - swapping %d double blocks to %s (backward=%s)",
            num_blocks, device, supports_backward,
        )

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap and self.offloader_double is not None:
            self.offloader_double.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap and self.offloader_double is not None:
            self.offloader_double.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        """Move non-swap-blocks to *device* while keeping swap blocks on CPU."""
        if self.blocks_to_swap:
            saved = self.double_blocks
            self.double_blocks = nn.ModuleList()  # temporarily hide

        self.to(device)

        if self.blocks_to_swap:
            self.double_blocks = saved

    def prepare_block_swap_before_forward(self) -> None:
        if not self.blocks_to_swap or self.offloader_double is None:
            return
        self.offloader_double.prepare_block_devices_before_forward(list(self.double_blocks))

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def get_rotary_pos_embed(
        self,
        rope_sizes: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D RoPE embeddings for given (T, H, W) token counts."""
        return get_nd_rotary_pos_embed(self.rope_dim_list, rope_sizes, theta=self.rope_theta)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: Optional[torch.Tensor] = None,
        byt5_text_states: Optional[torch.Tensor] = None,
        byt5_text_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb_cache: Optional[Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Forward pass of HunyuanVideo 1.5 transformer.

        Args:
            hidden_states: Concatenated [noisy_latent, cond_latent] [B, C, T, H, W].
            timestep: Diffusion timesteps [B].
            text_states: Qwen2.5-VL word embeddings [B, L, text_dim].
            encoder_attention_mask: Text attention mask [B, L].
            vision_states: SigLIP vision embeddings [B, V, vision_dim] (I2V only).
            byt5_text_states: ByT5 character embeddings [B, L_byt5, byt5_dim].
            byt5_text_mask: ByT5 attention mask [B, L_byt5].
            rotary_pos_emb_cache: Optional cache for RoPE embeddings.

        Returns:
            Predicted velocity field [B, out_channels, T, H, W].
        """
        bs = hidden_states.shape[0]
        _, _, ot, oh, ow = hidden_states.shape
        # patch_size=[1,1,1] → token dims equal spatial dims
        tt, th, tw = ot, oh, ow

        # --- RoPE embeddings (cached if provided) ---
        if rotary_pos_emb_cache is not None:
            key = (tt, th, tw)
            if key in rotary_pos_emb_cache:
                freqs_cos, freqs_sin = rotary_pos_emb_cache[key]
                freqs_cos = freqs_cos.to(hidden_states.device)
                freqs_sin = freqs_sin.to(hidden_states.device)
            else:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
                rotary_pos_emb_cache[key] = (freqs_cos.cpu(), freqs_sin.cpu())
        else:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
        freqs_cis = (freqs_cos, freqs_sin)

        # --- Patchify image latents → [B, N, hidden_size] ---
        img = self.img_in(hidden_states)

        # --- Timestep conditioning ---
        vec = self.time_in(timestep)

        # --- Refine Qwen2.5-VL text tokens ---
        txt_attn_params = AttentionParams.create_attention_params_from_mask(
            self.attn_mode, self.split_attn, 0, encoder_attention_mask
        )
        txt = self.txt_in(text_states, timestep, txt_attn_params)

        # Add conditioning-type embedding for word tokens (type 0)
        cond_emb = self.cond_type_embedding(
            torch.zeros_like(txt[:, :, 0], dtype=torch.long)
        )
        txt = txt + cond_emb

        # --- ByT5 character tokens ---
        byt5_txt = self.byt5_in(byt5_text_states)
        cond_emb = self.cond_type_embedding(
            torch.ones_like(byt5_txt[:, :, 0], dtype=torch.long)
        )
        byt5_txt = byt5_txt + cond_emb

        # --- Vision tokens (I2V) ---
        extra_encoder_hidden_states: Optional[torch.Tensor] = None
        if vision_states is not None:
            # In T2V mode, all-zero vision states are a no-op sentinel
            if self.task_type == "t2v" and torch.all(vision_states == 0):
                vision_states = None
            else:
                extra_encoder_hidden_states = self.vision_in(vision_states)
                cond_emb = self.cond_type_embedding(
                    torch.full_like(
                        extra_encoder_hidden_states[:, :, 0], 2, dtype=torch.long
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

        # --- Concatenate text tokens: [vision?] + [byt5] + [word] ---
        txt_lens: list[int] = []
        concatenated_txt_list: list[torch.Tensor] = []
        for i in range(bs):
            txt_len = int(encoder_attention_mask[i].to(torch.bool).sum().item())
            byt5_len = int(byt5_text_mask[i].to(torch.bool).sum().item()) if byt5_text_mask is not None else byt5_txt.shape[1]

            txt_i = txt[i, :txt_len]
            byt5_i = byt5_txt[i, :byt5_len]

            if extra_encoder_hidden_states is not None:
                vis_i = extra_encoder_hidden_states[i]
                cat_i = torch.cat([vis_i, byt5_i, txt_i], dim=0)
                total = txt_len + byt5_len + vis_i.shape[0]
            else:
                cat_i = torch.cat([byt5_i, txt_i], dim=0)
                total = txt_len + byt5_len

            concatenated_txt_list.append(cat_i)
            txt_lens.append(total)

        # Pad to max length
        max_txt_len = max(txt_lens)
        txt = torch.stack([
            torch.cat([
                concatenated_txt_list[i],
                torch.zeros(
                    max_txt_len - txt_lens[i],
                    concatenated_txt_list[i].shape[-1],
                    device=txt.device,
                    dtype=txt.dtype,
                ),
            ])
            for i in range(bs)
        ])

        # Combined text attention mask
        if bs == 1:
            text_mask = None  # single-sample: no mask needed
        else:
            text_mask = torch.stack([
                torch.cat([
                    torch.ones(txt_lens[i], device=hidden_states.device, dtype=hidden_states.dtype),
                    torch.zeros(max_txt_len - txt_lens[i], device=hidden_states.device, dtype=hidden_states.dtype),
                ])
                for i in range(bs)
            ])

        img_seq_len = img.shape[1]
        attn_params = AttentionParams.create_attention_params_from_mask(
            self.attn_mode, self.split_attn, img_seq_len, text_mask
        )

        # --- Double-stream blocks ---
        for idx, block in enumerate(self.double_blocks):
            if self.blocks_to_swap and self.offloader_double is not None:
                self.offloader_double.wait_for_block(idx)

            img, txt = block(img, txt, vec, freqs_cis, attn_params)

            if self.blocks_to_swap and self.offloader_double is not None:
                self.offloader_double.submit_move_blocks_forward(list(self.double_blocks), idx)

        del txt, attn_params, freqs_cis

        # --- Output projection ---
        img = self.final_layer(img, vec)
        del vec

        # --- Unpatchify: [B, N, C] → [B, C, T, H, W] ---
        img = self._unpatchify(img, tt, th, tw)
        return img

    def _unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """Reshape sequence tokens back to video tensor [B, C, T, H, W]."""
        c = self.out_channels
        x = x.reshape(x.shape[0], t, h, w, c)
        return x.permute(0, 4, 1, 2, 3)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def detect_hunyuan_video_1_5_sd_dtype(path: str) -> torch.dtype:
    """Detect model weight dtype from safetensors checkpoint."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        key1 = "double_blocks.0.img_attn_k.weight"    # official format
        key2 = "double_blocks.0.img_attn_qkv.weight"  # ComfyUI repackaged
        if key1 in keys:
            dtype = f.get_tensor(key1).dtype
        elif key2 in keys:
            dtype = f.get_tensor(key2).dtype
        else:
            raise ValueError(
                f"Cannot detect dtype from '{path}': expected key '{key1}' or '{key2}' not found"
            )

    logger.info("Detected HV 1.5 DiT dtype: %s", dtype)
    return dtype


def load_hunyuan_video_1_5_model(
    device: torch.device | str,
    task_type: str,
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: torch.device | str,
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
) -> HunyuanVideo15Transformer:
    """Load HunyuanVideo 1.5 transformer from a safetensors checkpoint.

    This function defers heavy imports (accelerate, safetensors) to call time
    so that unit tests importing only model classes stay fast.

    Args:
        device: Target device for training (cuda / cpu).
        task_type: "t2v" or "i2v".
        dit_path: Path to the .safetensors checkpoint file.
        attn_mode: Attention backend.
        split_attn: Whether to use split-attention.
        loading_device: Device to load weights onto initially.
        dit_weight_dtype: Dtype to cast weights to (None for fp8_scaled).
        fp8_scaled: Use FP8 weight scaling.

    Returns:
        Loaded and weight-initialised HunyuanVideo15Transformer.
    """
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # Build model on meta device to avoid OOM
    with init_empty_weights():
        model = HunyuanVideo15Transformer(
            task_type=task_type,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info("Loading HV 1.5 DiT from %s (device=%s)", dit_path, loading_device)

    sd = load_file(dit_path, device=str(loading_device))

    # Handle ComfyUI repackaged QKV → split Q/K/V
    patched_sd: dict = {}
    for key, val in sd.items():
        if any(suffix in key for suffix in ("img_attn_qkv.weight", "img_attn_qkv.bias",
                                            "txt_attn_qkv.weight", "txt_attn_qkv.bias")):
            for new_key, chunk in zip(
                [key.replace("_qkv", s) for s in ("_q", "_k", "_v")],
                torch.chunk(val, 3, dim=0),
            ):
                patched_sd[new_key] = chunk
        else:
            patched_sd[key] = val

    if dit_weight_dtype is not None:
        patched_sd = {k: v.to(dit_weight_dtype) for k, v in patched_sd.items()}

    info = model.load_state_dict(patched_sd, strict=True, assign=True)
    logger.info("Loaded HV 1.5 DiT: %s", info)
    return model
