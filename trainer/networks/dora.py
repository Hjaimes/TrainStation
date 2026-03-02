# DoRA (Weight-Decomposed Low-Rank Adaptation) network module
#
# Paper: "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024)
# https://arxiv.org/abs/2402.09353
#
# DoRA decomposes the pre-trained weight into magnitude and directional
# components, then adapts only the direction (via LoRA) while learning
# a separate per-output-feature magnitude parameter.
#
# Forward: out = magnitude * normalize(W_0 + scale * B@A) @ x
# Column-wise norm is detached per paper sec 4.3 to prevent gradient
# interference between magnitude and directional learning.
#
# Only supports Linear layers — Conv2d is not compatible with this
# decomposition without further modification.

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DoRAModule(nn.Module):
    """Weight-Decomposed LoRA module.

    Decomposes the weight update into magnitude and direction components.
    The magnitude vector is a learnable per-output-feature parameter
    initialized from the original weight's column norms.

    Only supports Linear layers. Conv2d will raise ValueError at construction.

    Args:
        lora_name: Unique name used as key in state dict.
        org_module: The original Linear layer to wrap.
        multiplier: Scale factor applied to the LoRA delta.
        lora_dim: Rank of the low-rank decomposition.
        alpha: Scaling alpha; if 0 or None, defaults to lora_dim.
        dropout: Standard dropout probability applied after lora_down.
        rank_dropout: Per-rank dropout probability (masks individual rank dims).
        module_dropout: Probability of skipping DoRA and returning original output.
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
        **kwargs,  # Accept and ignore extra kwargs for compatibility with container
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ != "Linear":
            raise ValueError(
                f"DoRA only supports Linear layers, got {org_module.__class__.__name__}. "
                "Set conv_rank=0 or use a different network_type for Conv2d layers."
            )

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.lora_dim = lora_dim

        # Standard LoRA down/up projections
        self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Alpha / scale — same convention as LoRA
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # bf16 safety
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # persisted in state dict

        # Magnitude parameter: per-output-feature, initialized from original weight column norms.
        # Column norm = norm of each row of W (each output feature's weight vector, dim=1).
        weight_f32 = org_module.weight.detach().float()
        self.magnitude = nn.Parameter(weight_f32.norm(dim=1))  # shape: (out_dim,)

        # Store references to original weight and bias — NOT as registered parameters.
        # Using a plain attribute list prevents nn.Module from registering them, which
        # would double-count parameters and create ownership conflicts.
        self._org_weight_ref = [org_module.weight]
        self._org_bias_ref = [org_module.bias]  # May be None

        self.multiplier = multiplier
        self.org_module = org_module  # Cleared in apply_to()
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    # ------------------------------------------------------------------
    # Properties for clean access to the referenced tensors
    # ------------------------------------------------------------------

    @property
    def _org_weight(self) -> torch.Tensor:
        return self._org_weight_ref[0]

    @property
    def _org_bias(self) -> Optional[torch.Tensor]:
        return self._org_bias_ref[0]

    # ------------------------------------------------------------------
    # Apply / restore
    # ------------------------------------------------------------------

    def apply_to(self) -> None:
        """Replace the original module's forward with this module's forward."""
        self.org_forward = self.org_module.forward
        # Refresh the references so they survive after del self.org_module
        self._org_weight_ref = [self.org_module.weight]
        self._org_bias_ref = [self.org_module.bias]
        self.org_module.forward = self.forward
        del self.org_module

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Module dropout — skip DoRA entirely, return original output
        if self.module_dropout is not None and self.training:
            if torch.rand(1).item() < self.module_dropout:
                return self.org_forward(x)

        # --- Low-rank projection ---
        lx = self.lora_down(x)

        if self.dropout is not None and self.training:
            lx = F.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            mask = torch.rand(lx.size(0), self.lora_dim, device=lx.device) > self.rank_dropout
            if lx.dim() == 3:
                mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        # --- DoRA weight composition ---
        # Merged weight: W_0 + scale * multiplier * (up_weight @ down_weight)
        # Cast org_weight to the LoRA parameter dtype to avoid dtype mismatches during training
        w0 = self._org_weight.to(self.lora_up.weight.dtype)
        lora_delta = self.lora_up.weight @ self.lora_down.weight  # (out_dim, in_dim)
        merged_weight = w0 + (scale * self.multiplier) * lora_delta

        # Column-wise L2 norm, detached per paper sec 4.3.
        # Detaching prevents gradient from flowing through the normalization denominator,
        # which would interfere with independent magnitude learning.
        weight_norm = merged_weight.norm(dim=1, keepdim=True).detach().clamp(min=1e-8)

        # Normalize direction, then scale by learned magnitude
        # magnitude: (out_dim,) -> (out_dim, 1) for broadcasting with (out_dim, in_dim)
        mag = self.magnitude.to(merged_weight.dtype)
        mag_weight = mag.unsqueeze(1) * (merged_weight / weight_norm)

        # Apply the composed weight + original bias
        return F.linear(x, mag_weight, self._org_bias)
