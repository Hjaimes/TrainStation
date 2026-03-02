import logging

import torch

logger = logging.getLogger(__name__)


def load_sdxl_unet(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
):
    """Load SDXL UNet from a single-file checkpoint or diffusers directory.

    Returns a diffusers UNet2DConditionModel. The import is deferred to avoid
    requiring diffusers at module import time.
    """
    from diffusers import UNet2DConditionModel

    logger.info("Loading SDXL UNet from %s", model_path)

    if model_path.endswith(".safetensors"):
        model = UNet2DConditionModel.from_single_file(model_path, torch_dtype=dtype)
    else:
        model = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype)

    model = model.to(device=device)
    logger.info("SDXL UNet loaded: %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    return model
