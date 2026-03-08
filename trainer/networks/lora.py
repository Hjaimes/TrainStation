# LoRA (Low-Rank Adaptation) network module
# Ported from Musubi_Tuner - math/forward logic identical to source.
#
# References:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
from typing import List, Optional

import torch
import torch.nn as nn


class LoRAModule(nn.Module):
    """Low-Rank Adaptation module that replaces the forward method of the
    original Linear or Conv2d layer, adding a low-rank residual path.

    Supports split_dims for mimicking split qkv of multi-head attention.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float | int | None = 1,
        dropout: float | None = None,
        rank_dropout: float | None = None,
        module_dropout: float | None = None,
        split_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        """Create a LoRA module wrapping ``org_module``.

        Args:
            lora_name: Unique name used as key in state dict.
            org_module: The original Linear or Conv2d to wrap.
            multiplier: Scale factor applied to the LoRA output.
            lora_dim: Rank of the low-rank decomposition.
            alpha: Scaling alpha; if 0 or None, defaults to lora_dim (no scaling).
            dropout: Standard dropout probability applied after lora_down.
            rank_dropout: Per-rank dropout probability (masks individual rank dims).
            module_dropout: Probability of skipping the entire LoRA path per forward.
            split_dims: If set, creates separate lora_down/lora_up pairs that are
                concatenated, mimicking split qkv. Only supported for Linear.
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            else:
                self.lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)

            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
        else:
            # Conv2d not supported with split_dims
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"

            self.lora_down = nn.ModuleList(
                [nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = nn.ModuleList(
                [nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims]
            )
            for lora_down in self.lora_down:
                nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                nn.init.zeros_(lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # persisted in state dict

        self.multiplier = multiplier
        self.org_module = org_module  # cleared in apply_to()
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self) -> None:
        """Replace the original module's forward with this module's forward."""
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_forward(x)

        # module dropout - skip entire LoRA path
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training:
                mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
            else:
                scale = self.scale

            lx = self.lora_up(lx)

            return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
                masks = [
                    torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                    for lx in lxs
                ]
                for i in range(len(lxs)):
                    if len(lxs[i].size()) == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lxs[i].size()) == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale
