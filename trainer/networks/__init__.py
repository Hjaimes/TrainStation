# trainer.networks - LoRA/LoHa/LoKr/DoRA network modules and container.

from typing import Type

import torch.nn as nn

from .lora import LoRAModule
from .loha import LoHaModule
from .lokr import LoKrModule
from .dora import DoRAModule
from .container import NetworkContainer
from .arch_configs import ARCH_NETWORK_CONFIGS, get_arch_config

__all__ = [
    "LoRAModule",
    "LoHaModule",
    "LoKrModule",
    "DoRAModule",
    "NetworkContainer",
    "ARCH_NETWORK_CONFIGS",
    "get_arch_config",
    "get_module_class",
]


def get_module_class(name: str) -> Type[nn.Module]:
    """Resolve a network module class by name.

    Args:
        name: One of "lora", "loha", "lokr", or "dora".

    Returns:
        The corresponding module class.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name == "lora":
        return LoRAModule
    elif name == "loha":
        return LoHaModule
    elif name == "lokr":
        return LoKrModule
    elif name == "dora":
        return DoRAModule
    else:
        raise ValueError(f"Unknown network module '{name}'. Available: lora, loha, lokr, dora")
