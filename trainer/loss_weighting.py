"""SNR-based loss weighting for diffusion training.

Supports flow matching (continuous t) and DDPM (discrete timesteps).
All functions operate on batched tensors for vectorized computation.

References:
- Min-SNR-gamma: Hang et al. 2023 "Efficient Diffusion Training via Min-SNR Weighting Strategy"
- Debiased estimation: "Perception Prioritized Training of Diffusion Models" (P2 weighting)
- P2 weighting: Choi et al. 2022 "Perception Prioritized Training of Diffusion Models"
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

WeightFn = Callable[[Tensor], Tensor]


def get_weight_fn(
    scheme: str,
    *,
    snr_gamma: float = 5.0,
    p2_gamma: float = 1.0,
) -> WeightFn | None:
    """Return a weight function for the given scheme.

    Args:
        scheme: One of "none", "min_snr_gamma", "debiased", "p2".
        snr_gamma: Gamma value for min-SNR weighting.
        p2_gamma: Gamma exponent for P2 weighting.

    Returns:
        A callable (snr_tensor) -> weight_tensor, or None for "none".
    """
    match scheme:
        case "none":
            return None
        case "min_snr_gamma":
            gamma = snr_gamma
            def _min_snr(snr: Tensor) -> Tensor:
                return min_snr_gamma_weights(snr, gamma)
            return _min_snr
        case "debiased":
            return debiased_estimation_weights
        case "p2":
            gamma = p2_gamma
            def _p2(snr: Tensor) -> Tensor:
                return p2_weights(snr, gamma)
            return _p2
        case _:
            raise ValueError(
                f"Unknown weighting scheme '{scheme}'. "
                f"Supported: 'none', 'min_snr_gamma', 'debiased', 'p2'."
            )


def compute_snr_flow_matching(t: Tensor) -> Tensor:
    """Compute SNR for flow matching: SNR(t) = (1-t)^2 / t^2.

    Args:
        t: Timestep values in [0, 1], shape (B,).

    Returns:
        SNR values, shape (B,). Clamped to avoid inf at t=0.
    """
    t_clamped = t.clamp(min=1e-6, max=1.0 - 1e-6)
    return ((1.0 - t_clamped) / t_clamped).square()


def compute_snr_ddpm(
    alphas_cumprod: Tensor,
    timesteps: Tensor,
) -> Tensor:
    """Compute SNR for DDPM: SNR(t) = alpha_bar(t) / (1 - alpha_bar(t)).

    Args:
        alphas_cumprod: Precomputed cumulative alpha products, shape (T,).
        timesteps: Integer timesteps, shape (B,).

    Returns:
        SNR values, shape (B,).
    """
    alpha_bar = alphas_cumprod[timesteps]
    return alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)


def min_snr_gamma_weights(snr: Tensor, gamma: float = 5.0) -> Tensor:
    """Min-SNR-gamma weighting: weight = min(SNR, gamma) / SNR.

    Downweights high-SNR (low-noise) timesteps where signal dominates.
    """
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr


def debiased_estimation_weights(snr: Tensor) -> Tensor:
    """Debiased estimation weighting: weight = 1 / sqrt(SNR).

    Equalizes the gradient contribution across noise levels.
    """
    return 1.0 / snr.sqrt().clamp(min=1e-6)


def p2_weights(snr: Tensor, gamma: float = 1.0, k: float = 1.0) -> Tensor:
    """P2 weighting: weight = 1 / (k + snr)^gamma.

    Downweights high-SNR (low-noise) timesteps where the target signal
    dominates. Setting gamma=0 recovers uniform weighting; higher gamma
    focuses training on noisier, perceptually harder timesteps.

    Reference: Choi et al. 2022 "Perception Prioritized Training of
    Diffusion Models" — Eq. (8), denominator form with k=1.

    Args:
        snr:   SNR values, shape (B,). Must be non-negative.
        gamma: Exponent controlling downweighting strength. Default 1.0.
        k:     Offset preventing division by zero at SNR=0. Default 1.0.

    Returns:
        Per-sample weights, shape (B,).
    """
    return 1.0 / (k + snr).pow(gamma)
