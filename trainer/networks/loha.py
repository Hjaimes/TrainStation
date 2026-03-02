# LoHa (Low-rank Hadamard Product) network module
# Ported from Musubi_Tuner — math/forward logic identical to source.
# Linear layers only (no Conv2d/Tucker decomposition).
#
# Reference: https://arxiv.org/abs/2108.06098
# Based on the LyCORIS project by KohakuBlueleaf
# https://github.com/KohakuBlueleaf/LyCORIS

import torch
import torch.nn as nn
import torch.nn.functional as F


class HadaWeight(torch.autograd.Function):
    """Efficient Hadamard product forward/backward for LoHa.

    Computes ((w1a @ w1b) * (w2a @ w2b)) * scale with a custom backward
    that recomputes intermediates instead of storing them.
    """

    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=None):
        if scale is None:
            scale = torch.tensor(1, device=w1a.device, dtype=w1a.dtype)
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        del temp
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


class LoHaModule(nn.Module):
    """LoHa module for training. Replaces forward method of the original Linear.

    Weight delta: dW = ((w1a @ w1b) * (w2a @ w2b)) * scale
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
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("LoHa Conv2d is not supported in this implementation")
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # Hadamard product parameters: dW = (w1a @ w1b) * (w2a @ w2b)
        self.hada_w1_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, in_dim))
        self.hada_w2_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, in_dim))

        # Initialization: w1_a normal(0.1), w1_b normal(1.0), w2_a = 0, w2_b normal(1.0)
        # Ensures dW = 0 at init since w2_a = 0
        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w1_b, std=1.0)
        nn.init.constant_(self.hada_w2_a, 0)
        nn.init.normal_(self.hada_w2_b, std=1.0)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

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

    def get_diff_weight(self) -> torch.Tensor:
        """Return materialized weight delta."""
        scale = torch.tensor(self.scale, dtype=self.hada_w1_a.dtype, device=self.hada_w1_a.device)
        return HadaWeight.apply(self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()

        # rank dropout
        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(
                diff_weight.dtype
            )
            drop = drop.view(-1, 1)
            diff_weight = diff_weight * drop
            # scaling for rank dropout
            scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            scale = 1.0

        return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale
