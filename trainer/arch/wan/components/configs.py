# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Ported from Musubi_Tuner wan/configs/. EasyDict replaced with SimpleNamespace.
import copy
import os
from types import SimpleNamespace

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


# ---- Shared config ----

_shared = _ns(
    t5_model="umt5_xxl",
    t5_dtype=torch.bfloat16,
    text_len=512,
    param_dtype=torch.bfloat16,
    out_dim=16,
    num_train_timesteps=1000,
    sample_fps=16,
    sample_neg_prompt=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    frame_num=81,
)


def _make_config(name: str, **overrides) -> SimpleNamespace:
    """Create a config starting from shared defaults with overrides."""
    cfg = copy.deepcopy(_shared)
    cfg.__name__ = name
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---- T2V 14B ----

t2v_14B = _make_config(
    "Config: Wan T2V 14B",
    i2v=False, is_fun_control=False, flf2v=False, v2_2=False,
    t5_checkpoint="models_t5_umt5-xxl-enc-bf16.pth",
    t5_tokenizer="google/umt5-xxl",
    vae_checkpoint="Wan2.1_VAE.pth",
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    dim=5120, ffn_dim=13824, freq_dim=256, in_dim=16,
    num_heads=40, num_layers=40, window_size=(-1, -1),
    qk_norm=True, cross_attn_norm=True, eps=1e-6,
    sample_shift=5.0, sample_steps=50, boundary=None,
    sample_guide_scale=(5.0,),
)

# ---- T2V 1.3B ----

t2v_1_3B = _make_config(
    "Config: Wan T2V 1.3B",
    i2v=False, is_fun_control=False, flf2v=False, v2_2=False,
    t5_checkpoint="models_t5_umt5-xxl-enc-bf16.pth",
    t5_tokenizer="google/umt5-xxl",
    vae_checkpoint="Wan2.1_VAE.pth",
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    dim=1536, ffn_dim=8960, freq_dim=256, in_dim=16,
    num_heads=12, num_layers=30, window_size=(-1, -1),
    qk_norm=True, cross_attn_norm=True, eps=1e-6,
    sample_shift=5.0, sample_steps=50, boundary=None,
    sample_guide_scale=(5.0,),
)

# ---- I2V 14B ----

i2v_14B = _make_config(
    "Config: Wan I2V 14B",
    i2v=True, is_fun_control=False, flf2v=False, v2_2=False,
    t5_checkpoint="models_t5_umt5-xxl-enc-bf16.pth",
    t5_tokenizer="google/umt5-xxl",
    clip_model="clip_xlm_roberta_vit_h_14",
    clip_dtype=torch.float16,
    clip_checkpoint="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    clip_tokenizer="xlm-roberta-large",
    vae_checkpoint="Wan2.1_VAE.pth",
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    dim=5120, ffn_dim=13824, freq_dim=256, in_dim=36,
    num_heads=40, num_layers=40, window_size=(-1, -1),
    qk_norm=True, cross_attn_norm=True, eps=1e-6,
    sample_shift=5.0, sample_steps=40, boundary=None,
    sample_guide_scale=(5.0,),
)

# ---- Wan 2.2 I2V A14B ----

i2v_A14B = _make_config(
    "Config: Wan I2V A14B",
    i2v=True, is_fun_control=False, flf2v=False, v2_2=True,
    t5_checkpoint="models_t5_umt5-xxl-enc-bf16.pth",
    t5_tokenizer="google/umt5-xxl",
    vae_checkpoint="Wan2.1_VAE.pth",
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    dim=5120, ffn_dim=13824, freq_dim=256, in_dim=36,
    num_heads=40, num_layers=40, window_size=(-1, -1),
    qk_norm=True, cross_attn_norm=True, eps=1e-6,
    low_noise_checkpoint="low_noise_model",
    high_noise_checkpoint="high_noise_model",
    sample_shift=5.0, sample_steps=40, boundary=0.900,
    sample_guide_scale=(3.5, 3.5),
)

# ---- Wan 2.2 T2V A14B ----

t2v_A14B = _make_config(
    "Config: Wan T2V A14B",
    i2v=False, is_fun_control=False, flf2v=False, v2_2=True,
    t5_checkpoint="models_t5_umt5-xxl-enc-bf16.pth",
    t5_tokenizer="google/umt5-xxl",
    vae_checkpoint="Wan2.1_VAE.pth",
    vae_stride=(4, 8, 8),
    patch_size=(1, 2, 2),
    dim=5120, ffn_dim=13824, in_dim=16, freq_dim=256,
    num_heads=40, num_layers=40, window_size=(-1, -1),
    qk_norm=True, cross_attn_norm=True, eps=1e-6,
    low_noise_checkpoint="low_noise_model",
    high_noise_checkpoint="high_noise_model",
    sample_shift=12.0, sample_steps=40, boundary=0.875,
    sample_guide_scale=(3.0, 4.0),
)

# ---- Derived configs ----

t2i_14B = copy.deepcopy(t2v_14B)
t2i_14B.__name__ = "Config: Wan T2I 14B"

flf2v_14B = copy.deepcopy(i2v_14B)
flf2v_14B.__name__ = "Config: Wan FLF2V 14B"
flf2v_14B.sample_neg_prompt = "镜头切换，" + flf2v_14B.sample_neg_prompt
flf2v_14B.i2v = False
flf2v_14B.flf2v = True

t2v_1_3B_FC = copy.deepcopy(t2v_1_3B)
t2v_1_3B_FC.__name__ = "Config: Wan-Fun-Control T2V 1.3B"
t2v_1_3B_FC.i2v = True
t2v_1_3B_FC.in_dim = 48
t2v_1_3B_FC.is_fun_control = True

t2v_14B_FC = copy.deepcopy(t2v_14B)
t2v_14B_FC.__name__ = "Config: Wan-Fun-Control T2V 14B"
t2v_14B_FC.i2v = True
t2v_14B_FC.in_dim = 48
t2v_14B_FC.is_fun_control = True

i2v_14B_FC = copy.deepcopy(i2v_14B)
i2v_14B_FC.__name__ = "Config: Wan-Fun-Control I2V 14B"
i2v_14B_FC.in_dim = 48
i2v_14B_FC.is_fun_control = True

# ---- Main config dict ----

WAN_CONFIGS = {
    "t2v-14B": t2v_14B,
    "t2v-1.3B": t2v_1_3B,
    "i2v-14B": i2v_14B,
    "t2i-14B": t2i_14B,
    "flf2v-14B": flf2v_14B,
    "t2v-1.3B-FC": t2v_1_3B_FC,
    "t2v-14B-FC": t2v_14B_FC,
    "i2v-14B-FC": i2v_14B_FC,
    "i2v-A14B": i2v_A14B,
    "t2v-A14B": t2v_A14B,
}

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "1024*1024": (1024, 1024),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
    "704*1280": 704 * 1280,
    "1280*704": 1280 * 704,
}

SUPPORTED_SIZES = {
    "t2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-1.3B": ("480*832", "832*480"),
    "i2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2i-14B": tuple(SIZE_CONFIGS.keys()),
    "flf2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-1.3B-FC": ("480*832", "832*480"),
    "t2v-14B-FC": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-14B-FC": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
}
