# LoKr (Low-rank Kronecker Product) network module
# Ported from Musubi_Tuner — math/forward logic identical to source.
# Linear layers only (no Conv2d/Tucker decomposition).
#
# Reference: https://arxiv.org/abs/2309.14859
# Based on the LyCORIS project by KohakuBlueleaf
# https://github.com/KohakuBlueleaf/LyCORIS

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Return a tuple of two values whose product equals dimension,
    optimized for balanced factors.

    In LoKr, the first value is for the weight scale (smaller),
    and the second value is for the weight (larger).

    Examples:
        factor=-1: 128 -> (8, 16), 512 -> (16, 32), 1024 -> (32, 32)
        factor=4:  128 -> (4, 32), 512 -> (4, 128)
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1: torch.Tensor, w2: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute Kronecker product of w1 and w2, scaled by scale."""
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


class LoKrModule(nn.Module):
    """LoKr module for training. Replaces forward method of the original Linear.

    Weight delta: dW = kron(w1, w2) * scale, where w2 may be low-rank (w2_a @ w2_b)
    or a full matrix depending on the rank relative to factored dimensions.
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
        factor: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("LoKr Conv2d is not supported in this implementation")
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        factor = int(factor)
        self.use_w2 = False

        # Factorize dimensions
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        # w1 is always a full matrix (the "scale" factor, small)
        self.lokr_w1 = nn.Parameter(torch.empty(out_l, in_m))

        # w2: low-rank decomposition if rank is small enough, otherwise full matrix
        if lora_dim < max(out_k, in_n) / 2:
            self.lokr_w2_a = nn.Parameter(torch.empty(out_k, lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_n))
        else:
            self.use_w2 = True
            self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n))
            if lora_dim >= max(out_k, in_n) / 2:
                logger.warning(
                    f"LoKr: lora_dim {lora_dim} is large for dim={max(in_dim, out_dim)} "
                    f"and factor={factor}, using full matrix mode."
                )

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        # if both w1 and w2 are full matrices, use scale = 1
        if self.use_w2:
            alpha = lora_dim
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialization
        nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        if self.use_w2:
            nn.init.constant_(self.lokr_w2, 0)
        else:
            nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            nn.init.constant_(self.lokr_w2_b, 0)
        # Ensures dW = kron(w1, 0) = 0 at init

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
        w1 = self.lokr_w1
        if self.use_w2:
            w2 = self.lokr_w2
        else:
            w2 = self.lokr_w2_a @ self.lokr_w2_b
        return make_kron(w1, w2, self.scale)

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
            scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            scale = 1.0

        return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale
