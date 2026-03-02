"""Exponential Moving Average tracker for model parameters.

Shadow params are stored on CPU in fp32 to avoid doubling GPU VRAM.
Uses warm-start decay from OneTrainer and non-blocking transfers.
"""
from __future__ import annotations

import logging
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

logger = logging.getLogger(__name__)


class EMATracker:
    """Tracks an exponential moving average of model parameters.

    Shadow parameters are kept in fp32 on the specified device (default: CPU).
    Updates use non-blocking copies to overlap with GPU compute.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        decay: float = 0.9999,
        device: str | torch.device = "cpu",
    ) -> None:
        self.max_decay = decay
        self.device = torch.device(device)
        self.shadow_params: list[Tensor] = [
            p.data.detach().clone().float().to(self.device) for p in parameters
        ]
        self._backup: list[Tensor] = []

    def get_decay(self, global_step: int) -> float:
        """Warm-start decay: ramps from ~0.1 to max_decay over early steps."""
        return min((1 + global_step) / (10 + global_step), self.max_decay)

    @torch.no_grad()
    def step(
        self,
        parameters: Iterable[Parameter],
        global_step: int,
    ) -> None:
        """Update shadow params toward current params."""
        decay = self.get_decay(global_step)
        for shadow, param in zip(self.shadow_params, parameters):
            param_fp32 = param.data.float().to(self.device, non_blocking=True)
            shadow.lerp_(param_fp32, 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[Parameter]) -> None:
        """Copy shadow params to model (for inference/saving). Backs up originals."""
        self._backup = []
        for shadow, param in zip(self.shadow_params, parameters):
            self._backup.append(param.data.clone())
            param.data.copy_(shadow.to(param.device, param.dtype, non_blocking=True))

    @torch.no_grad()
    def restore(self, parameters: Iterable[Parameter]) -> None:
        """Restore original params from backup (after inference/saving)."""
        for backup, param in zip(self._backup, parameters):
            param.data.copy_(backup)
        self._backup = []

    def state_dict(self) -> dict:
        return {
            "shadow_params": [s.clone() for s in self.shadow_params],
            "max_decay": self.max_decay,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow_params = [s.to(self.device) for s in state["shadow_params"]]
        self.max_decay = state.get("max_decay", self.max_decay)
