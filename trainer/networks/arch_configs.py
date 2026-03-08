# Architecture-specific network configurations.
# Extracted from Musubi_Tuner's per-arch lora files and network_arch.py.
#
# Each entry defines which module class names to target when walking the model,
# default exclude patterns (regex), and optional default include patterns.
# The NetworkContainer receives these values rather than doing arch detection itself.

from typing import Dict, List, TypedDict


class ArchNetworkConfig(TypedDict, total=False):
    target_modules: List[str]
    default_exclude_patterns: List[str]
    default_include_patterns: List[str]
    te_target_modules: List[str]           # Module class names for text encoder LoRA targeting


ARCH_NETWORK_CONFIGS: Dict[str, ArchNetworkConfig] = {
    # Wan 2.1
    "wan": {
        "target_modules": ["WanAttentionBlock"],
        "default_exclude_patterns": [
            r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*",
        ],
    },
    # HunyuanVideo (original - has both MMDoubleStreamBlock and MMSingleStreamBlock)
    "hunyuan_video": {
        "target_modules": ["MMDoubleStreamBlock", "MMSingleStreamBlock"],
        "default_exclude_patterns": [
            r".*(img_mod|txt_mod|modulation).*",
        ],
    },
    # HunyuanVideo 1.5 (only MMDoubleStreamBlock, no MMSingleStreamBlock)
    "hunyuan_video_1_5": {
        "target_modules": ["MMDoubleStreamBlock"],
        "default_exclude_patterns": [
            r".*(_in).*",
        ],
    },
    # FramePack (HunyuanVideo-based architecture)
    "framepack": {
        "target_modules": ["HunyuanVideoTransformerBlock", "HunyuanVideoSingleTransformerBlock"],
        "default_exclude_patterns": [
            r".*(norm).*",
        ],
    },
    # FLUX Kontext
    "flux_kontext": {
        "target_modules": ["DoubleStreamBlock", "SingleStreamBlock"],
        "default_exclude_patterns": [
            r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*",
            r".*(norm).*",
        ],
    },
    # FLUX 2 (same target/exclude config as FLUX Kontext)
    "flux_2": {
        "target_modules": ["DoubleStreamBlock", "SingleStreamBlock"],
        "default_exclude_patterns": [
            r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*",
            r".*(norm).*",
        ],
    },
    # Kandinsky 5 DiT (uses include_patterns by default for fine-grained targeting)
    "kandinsky5": {
        "target_modules": ["TransformerEncoderBlock", "TransformerDecoderBlock"],
        "default_exclude_patterns": [
            r".*modulation.*",
        ],
        "default_include_patterns": [
            r".*self_attention\.to_query.*",
            r".*self_attention\.to_key.*",
            r".*self_attention\.to_value.*",
            r".*self_attention\.out_layer.*",
            r".*cross_attention\.to_query.*",
            r".*cross_attention\.to_key.*",
            r".*cross_attention\.to_value.*",
            r".*cross_attention\.out_layer.*",
            r".*feed_forward\.in_layer.*",
            r".*feed_forward\.out_layer.*",
        ],
    },
    # Qwen-Image
    "qwen_image": {
        "target_modules": ["QwenImageTransformerBlock"],
        "default_exclude_patterns": [
            r".*(_mod_).*",
        ],
    },
    # Z-Image
    "zimage": {
        "target_modules": ["ZImageTransformerBlock"],
        "default_exclude_patterns": [
            r".*(_modulation|_refiner).*",
        ],
    },
    # SDXL (UNet-based - targets diffusers Transformer2DModel blocks)
    "sdxl": {
        "target_modules": ["Transformer2DModel"],
        "default_exclude_patterns": [
            r".*norm.*",
        ],
    },
    # SD3 (MMDiT - targets joint + single transformer blocks)
    "sd3": {
        "target_modules": ["JointTransformerBlock", "SD3SingleTransformerBlock"],
        "default_exclude_patterns": [
            r".*norm.*",
            r".*ada_norm.*",
        ],
    },
    # Flux 1 (dual-stream, distinct block names from Flux 2)
    "flux_1": {
        "target_modules": ["Flux1DoubleStreamBlock", "Flux1SingleStreamBlock"],
        "default_exclude_patterns": [
            r".*(modulation).*",
            r".*(norm).*",
        ],
    },
}


def get_arch_config(arch_name: str) -> ArchNetworkConfig:
    """Look up network configuration for an architecture.

    Args:
        arch_name: Architecture key (e.g. "wan", "hunyuan_video").

    Returns:
        The ArchNetworkConfig dict.

    Raises:
        ValueError: If the architecture is not found.
    """
    if arch_name not in ARCH_NETWORK_CONFIGS:
        available = ", ".join(sorted(ARCH_NETWORK_CONFIGS.keys()))
        raise ValueError(f"Unknown architecture '{arch_name}'. Available: {available}")
    return ARCH_NETWORK_CONFIGS[arch_name]
