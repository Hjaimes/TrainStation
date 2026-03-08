"""Kandinsky 5 task-based configuration dataclasses.

Ported from Musubi_Tuner's kandinsky5/configs.py. Inline configs - no
external YAML required. TASK_CONFIGS keys match the ``task`` model kwarg.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------

@dataclass
class TextEmbedderConfig:
    qwen_checkpoint: str
    qwen_max_length: int
    clip_checkpoint: str
    clip_max_length: int
    qwen_dim: int = 3584
    clip_dim: int = 768


@dataclass
class AttentionConfig:
    type: str
    chunk: bool
    causal: bool
    local: bool
    glob: bool
    window: int
    chunk_len: Optional[int] = None
    method: Optional[str] = None
    P: Optional[float] = None
    add_sta: Optional[bool] = None
    wT: Optional[int] = None
    wH: Optional[int] = None
    wW: Optional[int] = None


@dataclass
class VAEConfig:
    name: str
    checkpoint_path: str


@dataclass
class DiTParams:
    in_visual_dim: int
    out_visual_dim: int
    in_text_dim: int
    in_text_dim2: int
    time_dim: int
    patch_size: tuple[int, int, int]
    model_dim: int
    ff_dim: int
    num_text_blocks: int
    num_visual_blocks: int
    axes_dims: tuple[int, int, int]
    visual_cond: bool
    instruct_type: Optional[str] = None


@dataclass
class TaskConfig:
    name: str
    checkpoint_path: str
    num_steps: int
    guidance_weight: float
    resolution: int
    scale_factor: List[float]
    dit_params: DiTParams
    attention: AttentionConfig
    vae: VAEConfig
    text: TextEmbedderConfig
    scheduler_scale: Optional[float] = None
    magcache: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Helper constructors (avoids repeating the same paths everywhere)
# ---------------------------------------------------------------------------

def _text_default() -> TextEmbedderConfig:
    return TextEmbedderConfig(
        qwen_checkpoint="./weights/text_encoder/",
        qwen_max_length=256,
        clip_checkpoint="./weights/text_encoder2/",
        clip_max_length=77,
    )


def _text_image() -> TextEmbedderConfig:
    """Longer context for image models (t2i, i2i)."""
    return TextEmbedderConfig(
        qwen_checkpoint="./weights/text_encoder/",
        qwen_max_length=512,
        clip_checkpoint="./weights/text_encoder2/",
        clip_max_length=77,
    )


def _vae_hunyuan() -> VAEConfig:
    return VAEConfig(name="hunyuan", checkpoint_path="./weights/vae/")


def _attn_flash() -> AttentionConfig:
    return AttentionConfig(
        type="flash", chunk=False, causal=False,
        local=False, glob=False, window=3,
    )


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, TaskConfig] = {
    # ------------------------------------------------------------------
    # Lite image models
    # ------------------------------------------------------------------
    "k5-lite-t2i-hd": TaskConfig(
        name="k5-lite-t2i-hd",
        checkpoint_path="./weights/model/kandinsky5lite_t2i.safetensors",
        num_steps=50,
        guidance_weight=3.5,
        resolution=1024,
        scale_factor=[1.0, 1.0, 1.0],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=512,
            patch_size=(1, 2, 2),
            model_dim=2560,
            ff_dim=10240,
            num_text_blocks=2,
            num_visual_blocks=50,
            axes_dims=(32, 48, 48),
            visual_cond=False,
        ),
        attention=_attn_flash(),
        vae=VAEConfig(name="flux", checkpoint_path="./weights/flux/vae"),
        text=_text_image(),
        scheduler_scale=3.0,
    ),
    "k5-lite-i2i-hd": TaskConfig(
        name="k5-lite-i2i-hd",
        checkpoint_path="./weights/model/kandinsky5lite_i2i.safetensors",
        num_steps=50,
        guidance_weight=3.5,
        resolution=1024,
        scale_factor=[1.0, 1.0, 1.0],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=512,
            patch_size=(1, 2, 2),
            model_dim=2560,
            ff_dim=10240,
            num_text_blocks=2,
            num_visual_blocks=50,
            axes_dims=(32, 48, 48),
            visual_cond=False,
            instruct_type="channel",
        ),
        attention=_attn_flash(),
        vae=VAEConfig(name="flux", checkpoint_path="./weights/flux/vae"),
        text=_text_image(),
        scheduler_scale=3.0,
    ),
    # ------------------------------------------------------------------
    # Lite video models
    # ------------------------------------------------------------------
    "k5-lite-t2v-5s-sd": TaskConfig(
        name="k5-lite-t2v-5s-sd",
        checkpoint_path="./weights/model/kandinsky5lite_t2v_sft_5s.safetensors",
        num_steps=50,
        guidance_weight=5.0,
        resolution=512,
        scale_factor=[1.0, 2.0, 2.0],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=512,
            patch_size=(1, 2, 2),
            model_dim=1792,
            ff_dim=7168,
            num_text_blocks=2,
            num_visual_blocks=32,
            axes_dims=(16, 24, 24),
            visual_cond=True,
        ),
        attention=_attn_flash(),
        vae=_vae_hunyuan(),
        text=_text_default(),
        scheduler_scale=10.0,
    ),
    "k5-lite-t2v-10s-sd": TaskConfig(
        name="k5-lite-t2v-10s-sd",
        checkpoint_path="./weights/model/kandinsky5lite_t2v_sft_10s.safetensors",
        num_steps=50,
        guidance_weight=5.0,
        resolution=512,
        scale_factor=[1.0, 2.0, 2.0],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=512,
            patch_size=(1, 2, 2),
            model_dim=1792,
            ff_dim=7168,
            num_text_blocks=2,
            num_visual_blocks=32,
            axes_dims=(16, 24, 24),
            visual_cond=True,
        ),
        attention=AttentionConfig(
            type="nabla",
            chunk=False,
            causal=False,
            local=False,
            glob=False,
            window=3,
            method="topcdf",
            P=0.9,
            add_sta=True,
            wT=11,
            wH=3,
            wW=3,
        ),
        vae=_vae_hunyuan(),
        text=_text_default(),
        scheduler_scale=10.0,
    ),
    "k5-lite-i2v-5s-sd": TaskConfig(
        name="k5-lite-i2v-5s-sd",
        checkpoint_path="./weights/model/kandinsky5lite_i2v_5s.safetensors",
        num_steps=50,
        guidance_weight=5.0,
        resolution=512,
        scale_factor=[1.0, 2.0, 2.0],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=512,
            patch_size=(1, 2, 2),
            model_dim=1792,
            ff_dim=7168,
            num_text_blocks=2,
            num_visual_blocks=32,
            axes_dims=(16, 24, 24),
            visual_cond=True,
        ),
        attention=_attn_flash(),
        vae=_vae_hunyuan(),
        text=_text_default(),
        scheduler_scale=10.0,
    ),
    # ------------------------------------------------------------------
    # Pro video models (19B)
    # ------------------------------------------------------------------
    "k5-pro-t2v-5s-sd": TaskConfig(
        name="k5-pro-t2v-5s-sd",
        checkpoint_path="./weights/model/kandinsky5pro_t2v_sft_5s.safetensors",
        num_steps=50,
        guidance_weight=5.0,
        resolution=512,
        scale_factor=[1.0, 2.0, 2.0],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=1024,
            patch_size=(1, 2, 2),
            model_dim=4096,
            ff_dim=16384,
            num_text_blocks=4,
            num_visual_blocks=60,
            axes_dims=(32, 48, 48),
            visual_cond=True,
        ),
        attention=_attn_flash(),
        vae=_vae_hunyuan(),
        text=_text_default(),
        scheduler_scale=10.0,
    ),
    "k5-pro-t2v-5s-hd": TaskConfig(
        name="k5-pro-t2v-5s-hd",
        checkpoint_path="./weights/model/kandinsky5pro_t2v_sft_5s.safetensors",
        num_steps=50,
        guidance_weight=5.0,
        resolution=1024,
        scale_factor=[1.0, 3.16, 3.16],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=1024,
            patch_size=(1, 2, 2),
            model_dim=4096,
            ff_dim=16384,
            num_text_blocks=4,
            num_visual_blocks=60,
            axes_dims=(32, 48, 48),
            visual_cond=True,
        ),
        attention=AttentionConfig(
            type="nabla",
            chunk=False,
            causal=False,
            local=False,
            glob=False,
            window=3,
            method="topcdf",
            P=0.8,
            add_sta=True,
            wT=11,
            wH=3,
            wW=3,
        ),
        vae=_vae_hunyuan(),
        text=_text_default(),
        scheduler_scale=10.0,
    ),
    "k5-pro-t2v-10s-sd": TaskConfig(
        name="k5-pro-t2v-10s-sd",
        checkpoint_path="./weights/model/kandinsky5pro_t2v_sft_10s.safetensors",
        num_steps=50,
        guidance_weight=5.0,
        resolution=512,
        scale_factor=[1.0, 2.0, 2.0],
        dit_params=DiTParams(
            in_visual_dim=16,
            out_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=1024,
            patch_size=(1, 2, 2),
            model_dim=4096,
            ff_dim=16384,
            num_text_blocks=4,
            num_visual_blocks=60,
            axes_dims=(32, 48, 48),
            visual_cond=True,
        ),
        attention=AttentionConfig(
            type="nabla",
            chunk=False,
            causal=False,
            local=False,
            glob=False,
            window=3,
            method="topcdf",
            P=0.9,
            add_sta=True,
            wT=11,
            wH=3,
            wW=3,
        ),
        vae=_vae_hunyuan(),
        text=_text_default(),
        scheduler_scale=10.0,
    ),
}
