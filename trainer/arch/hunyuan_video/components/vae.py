"""HunyuanVideo VAE loader stub.

The full VAE implementation requires diffusers + custom causal 3D conv blocks.
This module provides the VAE scaling factor constant and a loader that delegates
to diffusers' AutoencoderKL interface.

VAE type: "884-16c-hy" (temporal 4x, spatial 8x, 16 latent channels)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Scaling factor for HunyuanVideo 884-16c VAE
VAE_SCALING_FACTOR = 0.476986
VAE_VER = "884-16c-hy"

# VAE config for reference (same as Musubi_Tuner)
_VAE_CONFIG = {
    "_class_name": "AutoencoderKLCausal3D",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlockCausal3D",
        "DownEncoderBlockCausal3D",
        "DownEncoderBlockCausal3D",
        "DownEncoderBlockCausal3D",
    ],
    "in_channels": 3,
    "latent_channels": 16,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 256,
    "sample_tsize": 64,
    "up_block_types": [
        "UpDecoderBlockCausal3D",
        "UpDecoderBlockCausal3D",
        "UpDecoderBlockCausal3D",
        "UpDecoderBlockCausal3D",
    ],
    "scaling_factor": VAE_SCALING_FACTOR,
    "time_compression_ratio": 4,
    "mid_block_add_attention": True,
}


def load_vae(
    vae_path: str,
    vae_dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Load the HunyuanVideo causal 3D VAE.

    Requires the Musubi_Tuner package to be installed (for causal 3D conv blocks).
    Falls back gracefully with a descriptive error if unavailable.

    Args:
        vae_path: Path to the VAE safetensors checkpoint.
        vae_dtype: Weight dtype (None = float32).
        device: Target device.

    Returns:
        Loaded VAE module in eval mode.
    """
    vae_path = Path(vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")

    if isinstance(vae_dtype, str):
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        vae_dtype = dtype_map.get(vae_dtype, torch.float32)

    try:
        from musubi_tuner.hunyuan_model.vae import load_vae as _musubi_load_vae

        vae = _musubi_load_vae(
            vae_type=VAE_VER,
            vae_dtype=vae_dtype,
            vae_path=str(vae_path),
            device=device,
        )
        logger.info("HunyuanVideo VAE loaded from: %s", vae_path)
        return vae

    except ImportError:
        raise ImportError(
            "musubi_tuner is required for the HunyuanVideo VAE. "
            "The VAE uses custom causal 3D conv blocks not re-implemented here. "
            "Install musubi_tuner or provide pre-cached latents."
        )
