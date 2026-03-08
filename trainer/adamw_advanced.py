"""AdamW Advanced - a modular AdamW with 10+ toggleable features.

Inspired by OneTrainer's AdamW_Adv. All features are disabled by default,
making this behave identically to standard AdamW when no toggles are set.

Toggleable features:
    - AdEMAMix: Dual-momentum with slow EMA (beta3_ema) for stability
    - Cautious masking: Zero momentum where it contradicts the gradient
    - Cautious weight decay: Apply WD only where param & update signs agree
    - GRAMS moment: sign(grad) * |momentum| for direction-magnitude decoupling
    - OrthoGrad: Project gradient orthogonal to parameter vector
    - atan2 update: Bounded, scale-invariant update rule
    - SMMF factorization: Rank-1 compress optimizer states (~60-70% memory savings)
    - Kourkoutas dynamic beta2: Layer-wise adaptive beta2 based on gradient variance
    - Stochastic rounding: BF16-safe rounding (Nerogar, arXiv:2010.06192)
    - Bias correction toggle
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# atan2 scaling constant: 4/pi normalizes atan2 output to [-1, 1] range
_ATAN2_SCALE = 4.0 / math.pi


# ---------------------------------------------------------------------------
# Stochastic rounding utilities
# ---------------------------------------------------------------------------

_generators: dict[torch.device, torch.Generator] = {}


def _init_sr_generator(device: torch.device) -> None:
    if device not in _generators:
        _generators[device] = torch.Generator(device=device)
        _generators[device].manual_seed(42)


def _copy_stochastic_(target: Tensor, source: Tensor) -> None:
    """Copy fp32 source into bf16 target with stochastic rounding."""
    gen = _generators.get(source.device)
    if gen is None:
        _init_sr_generator(source.device)
        gen = _generators[source.device]
    rand_int = torch.randint(
        0, 1 << 16, source.shape,
        device=source.device, dtype=torch.int32, generator=gen,
    )
    rand_int.add_(source.view(dtype=torch.int32))
    rand_int.bitwise_and_(-65536)  # FFFF0000 mask
    target.copy_(rand_int.view(dtype=torch.float32))


# ---------------------------------------------------------------------------
# Gradient modification utilities
# ---------------------------------------------------------------------------

def _orthogonalize_gradient(p: Tensor, grad: Tensor) -> Tensor:
    """Project gradient orthogonal to parameter vector (OrthoGrad)."""
    w = p.view(-1).float()
    g = grad.view(-1).float()
    w_norm_sq = torch.dot(w, w).add_(1e-30)
    proj = torch.dot(w, g) / w_norm_sq
    g_orth = g.sub(w * proj)
    g_norm = g.norm(2)
    g_orth_norm = g_orth.norm(2).add_(1e-30)
    g_orth_scaled = g_orth * (g_norm / g_orth_norm)
    return g_orth_scaled.view(grad.shape).to(grad.dtype)


def _grams_update(mt: Tensor, grad: Tensor) -> Tensor:
    """GRAMS: sign(grad) * |momentum|. Returns new tensor."""
    return grad.sign().mul_(mt.abs())


def _cautious_update(mt: Tensor, grad: Tensor) -> Tensor:
    """Cautious masking: zero out momentum where it contradicts gradient."""
    mask = (mt * grad > 0).to(grad.dtype)
    mask.div_(mask.mean().clamp_min_(1e-3))
    return mt.mul(mask)


# ---------------------------------------------------------------------------
# SMMF factorization utilities
# ---------------------------------------------------------------------------

def _get_effective_shape(numel: int) -> tuple[int, int]:
    """Find two factors of numel closest to its square root."""
    for i in reversed(range(1, int(numel ** 0.5) + 1)):
        if numel % i == 0:
            return (numel // i, i)
    return (numel, 1)


def _nnmf(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Rank-1 non-negative matrix factorization."""
    M, N = matrix.shape
    mu = torch.sum(matrix, dim=1, dtype=torch.float32)
    mv = torch.sum(matrix, dim=0, dtype=torch.float32)
    eps = 1e-12
    if M < N:
        mu.div_(mu.sum().clamp_min_(eps))
    else:
        mv.div_(mv.sum().clamp_min_(eps))
    return mu, mv


def _pack_bools(tensor: Tensor) -> Tensor:
    """Pack boolean matrix into uint8 (1-bit per element)."""
    n, m = tensor.shape
    packed_m = (m + 7) // 8
    padded = torch.nn.functional.pad(tensor, (0, packed_m * 8 - m), "constant", 0)
    reshaped = padded.view(n, packed_m, 8)
    shifter = torch.arange(8, device=tensor.device, dtype=torch.uint8)
    return (reshaped.to(torch.uint8) * (2 ** shifter)).sum(dim=2).to(torch.uint8)


def _unpack_bools(packed: Tensor, original_m: int) -> Tensor:
    """Unpack uint8 back to boolean matrix."""
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    shifter = (2 ** torch.arange(8, device=packed.device, dtype=torch.uint8)).view(1, 1, 8)
    unpacked = (packed.unsqueeze(2) & shifter) != 0
    return unpacked.view(packed.shape[0], -1)[:, :original_m]


def _reconstruct_state(factors: tuple, signed: bool) -> Tensor:
    """Reconstruct full state from rank-1 factors + optional sign."""
    if signed:
        mu, mv, sign, d2 = factors
        full = torch.outer(mu.float(), mv.float())
        unpacked_sign = _unpack_bools(sign, original_m=d2)
        torch.where(unpacked_sign, full, -full, out=full)
        return full
    else:
        mu, mv = factors
        return torch.outer(mu.float(), mv.float())


def _factorize_state(full: Tensor, signed: bool) -> tuple:
    """Compress full state to rank-1 factors + optional 1-bit sign."""
    if signed:
        sign = _pack_bools(full > 0)
        mu, mv = _nnmf(full.abs_())
        return mu, mv, sign
    else:
        mu, mv = _nnmf(full.abs_())
        return mu, mv


# ---------------------------------------------------------------------------
# Kourkoutas dynamic beta2 helper
# ---------------------------------------------------------------------------

class _KourkoutasHelper:
    """Layer-wise adaptive beta2 based on gradient variance (sunspike detection)."""

    def __init__(self, optimizer: AdamWAdvanced) -> None:
        self.optimizer = optimizer
        self.layer_state: dict[Any, dict[str, Any]] = {}
        self.layer_info: dict[Any, dict[str, Any]] = {}
        self._current_step_prepared = -1
        self._build_layer_info()

    def _build_layer_info(self) -> None:
        key_fn = self.optimizer._layer_key_fn or (lambda p: tuple(p.shape))
        for group in self.optimizer.param_groups:
            if not group.get("kourkoutas_beta", False):
                continue
            for p in group["params"]:
                key = key_fn(p)
                if key not in self.layer_info:
                    self.layer_info[key] = {"params": [], "group_ref": group}
                self.layer_info[key]["params"].append(p)

    def maybe_prepare_step(self, current_step: int, device: torch.device) -> None:
        if self._current_step_prepared < current_step:
            self._prepare_step(current_step, device)
            self._current_step_prepared = current_step

    def _prepare_step(self, current_step: int, device: torch.device) -> None:
        defaults = self.optimizer.defaults
        for layer_key, info in self.layer_info.items():
            group = info["group_ref"]
            if layer_key not in self.layer_state:
                self.layer_state[layer_key] = {
                    "sum_sq_accumulator": torch.tensor(0.0, device=device, dtype=torch.float32),
                    "r_ema": torch.tensor(0.0, device=device, dtype=torch.float32),
                }

            ls = self.layer_state[layer_key]
            pooled_norm = torch.sqrt(ls["sum_sq_accumulator"])

            ema_alpha = group.get("ema_alpha", defaults.get("ema_alpha", 0.95))
            beta2_max = group["betas"][1]
            beta2_min = group.get("beta2_min", defaults.get("beta2_min", 0.9))
            tiny_spike = group.get("tiny_spike", defaults.get("tiny_spike", 1e-9))
            k_warmup = group.get("k_warmup_steps", defaults.get("k_warmup_steps", 0))

            ls["r_ema"].mul_(ema_alpha).add_(pooled_norm, alpha=1.0 - ema_alpha)

            if current_step < k_warmup:
                beta2 = beta2_max
            else:
                raw = pooled_norm / (ls["r_ema"] + tiny_spike)
                sun = raw / (1.0 + raw)
                beta2 = beta2_max - (beta2_max - beta2_min) * sun

            ls["dynamic_beta2"] = beta2.item() if isinstance(beta2, Tensor) else beta2
            ls["sum_sq_accumulator"].zero_()

    def accumulate_gradient_sq_norm(self, p: Tensor, grad: Tensor) -> None:
        key_fn = self.optimizer._layer_key_fn or (lambda p: tuple(p.shape))
        layer_key = key_fn(p)
        if layer_key in self.layer_state:
            self.layer_state[layer_key]["sum_sq_accumulator"] += grad.detach().pow(2).sum().float()

    def get_beta2(self, p: Tensor, group: dict) -> float:
        key_fn = self.optimizer._layer_key_fn or (lambda p: tuple(p.shape))
        layer_key = key_fn(p)
        default = group["betas"][1]
        return self.layer_state.get(layer_key, {}).get("dynamic_beta2", default)


# ---------------------------------------------------------------------------
# AdamW Advanced optimizer
# ---------------------------------------------------------------------------

class AdamWAdvanced(Optimizer):
    """AdamW with toggleable advanced features.

    All features default to off, giving standard AdamW behavior. Enable
    features individually via constructor args or ``optimizer_kwargs``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        # Feature toggles
        use_bias_correction: bool = True,
        cautious_wd: bool = False,
        cautious_mask: bool = False,
        grams_moment: bool = False,
        orthogonal_gradient: bool = False,
        use_atan2: bool = False,
        use_ademamix: bool = False,
        beta3_ema: float = 0.9999,
        ademamix_alpha: float = 5.0,
        kourkoutas_beta: bool = False,
        beta2_min: float = 0.9,
        ema_alpha: float = 0.95,
        tiny_spike: float = 1e-9,
        k_warmup_steps: int = 0,
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        stochastic_rounding: bool = False,
        layer_key_fn: Callable | None = None,
    ):
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Betas must be in [0, 1). Got {betas}")
        if kourkoutas_beta and betas[1] <= beta2_min:
            raise ValueError(
                f"Kourkoutas-beta requires betas[1] > beta2_min. "
                f"Got betas[1]={betas[1]}, beta2_min={beta2_min}"
            )
        if cautious_mask and grams_moment:
            logger.warning("cautious_mask and grams_moment are incompatible; disabling cautious_mask")
            cautious_mask = False

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            use_bias_correction=use_bias_correction,
            cautious_wd=cautious_wd,
            use_atan2=use_atan2,
            orthogonal_gradient=orthogonal_gradient,
            nnmf_factor=nnmf_factor,
            vector_reshape=vector_reshape,
            kourkoutas_beta=kourkoutas_beta,
            beta2_min=beta2_min,
            ema_alpha=ema_alpha,
            tiny_spike=tiny_spike,
            k_warmup_steps=k_warmup_steps,
            beta3_ema=beta3_ema,
            ademamix_alpha=ademamix_alpha,
        )

        # Instance-level flags (not per-group to keep it simple)
        self._cautious_mask = cautious_mask
        self._grams_moment = grams_moment
        self._use_ademamix = use_ademamix
        self._stochastic_rounding = stochastic_rounding
        self._layer_key_fn = layer_key_fn

        super().__init__(params, defaults)

        # Init Kourkoutas helper after super().__init__ so param_groups exist
        self._kourkoutas_helper: _KourkoutasHelper | None = None
        if kourkoutas_beta:
            self._kourkoutas_helper = _KourkoutasHelper(self)

        # Init stochastic rounding generators
        if self._stochastic_rounding:
            devices = {
                p.device for group in self.param_groups
                for p in group["params"] if p.dtype == torch.bfloat16
            }
            for device in devices:
                _init_sr_generator(device)

    @property
    def supports_fused_back_pass(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Per-parameter step (supports fused backward pass)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_parameter(self, p: Tensor, group: dict, i: int | None = None) -> None:
        """Step a single parameter. Called by step() or fused backward hooks."""
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]

        # --- State init ---
        if "step" not in state:
            state["step"] = 0
            factored = (
                group["nnmf_factor"]
                and not (p.dim() == 1 and not group["vector_reshape"])
            )
            state["factored"] = factored
            dtype = torch.float32 if factored else p.dtype
            device = p.device

            if factored:
                state["effective_shape"] = _get_effective_shape(p.numel())
                d1, d2 = state["effective_shape"]
                if group["betas"][0] > 0:
                    state["mu_m"] = torch.zeros(d1, device=device, dtype=dtype)
                    state["mv_m"] = torch.zeros(d2, device=device, dtype=dtype)
                    state["sign_m"] = torch.zeros((d1, (d2 + 7) // 8), dtype=torch.uint8, device=device)
                if self._use_ademamix:
                    state["mu_m_slow"] = torch.zeros(d1, device=device, dtype=dtype)
                    state["mv_m_slow"] = torch.zeros(d2, device=device, dtype=dtype)
                    state["sign_m_slow"] = torch.zeros((d1, (d2 + 7) // 8), dtype=torch.uint8, device=device)
                state["mu_v"] = torch.zeros(d1, device=device, dtype=dtype)
                state["mv_v"] = torch.zeros(d2, device=device, dtype=dtype)
            else:
                if group["betas"][0] > 0:
                    state["exp_avg"] = torch.zeros_like(p, dtype=dtype)
                if self._use_ademamix:
                    state["exp_avg_slow"] = torch.zeros_like(p, dtype=dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=dtype)

        beta1, beta2 = group["betas"]

        current_step = state["step"]
        if self._kourkoutas_helper is not None and group.get("kourkoutas_beta", False):
            self._kourkoutas_helper.maybe_prepare_step(current_step, p.device)
            beta2 = self._kourkoutas_helper.get_beta2(p, group)

        # Bias correction
        if group["use_bias_correction"]:
            step = current_step + 1
            bc1 = 1.0 - beta1 ** step
            sqrt_bc2 = (1.0 - group["betas"][1] ** step) ** 0.5
        else:
            bc1 = 1.0
            sqrt_bc2 = 1.0
        step_size = group["lr"] / bc1

        # --- Core step logic ---
        self._step_core(p, grad, state, group, step_size, beta1, beta2, sqrt_bc2)

        state["step"] += 1

    def _step_core(
        self, p: Tensor, grad: Tensor, state: dict, group: dict,
        step_size: float, beta1: float, beta2: float, sqrt_bc2: float,
    ) -> None:
        factored = state["factored"]

        if factored and grad.dtype != torch.float32:
            grad = grad.float()

        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        if self._kourkoutas_helper is not None and group.get("kourkoutas_beta", False):
            self._kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)

        ademamix = self._use_ademamix
        beta3 = group["beta3_ema"] if ademamix else 0.0
        alpha_mix = group["ademamix_alpha"] if ademamix else 0.0

        if factored:
            update = self._step_factored(
                grad, state, group, beta1, beta2, sqrt_bc2, ademamix, beta3, alpha_mix,
            )
            update = update.view(p.shape)
        else:
            update = self._step_standard(
                grad, state, group, beta1, beta2, sqrt_bc2, ademamix, beta3, alpha_mix,
            )

        scale = step_size * _ATAN2_SCALE if group["use_atan2"] else step_size
        update.mul_(scale)

        # Apply weight decay + parameter update
        self._apply_update(p, group, update, step_size)

    def _step_standard(
        self, grad: Tensor, state: dict, group: dict,
        beta1: float, beta2: float, sqrt_bc2: float,
        ademamix: bool, beta3: float, alpha_mix: float,
    ) -> Tensor:
        # First moment
        if beta1 > 0:
            exp_avg = state["exp_avg"]
            exp_avg.lerp_(grad, 1.0 - beta1)

            if self._grams_moment:
                update_mt = _grams_update(exp_avg, grad)
            elif self._cautious_mask:
                update_mt = _cautious_update(exp_avg, grad)
            else:
                update_mt = exp_avg.clone()
        else:
            update_mt = None

        # AdEMAMix slow moment
        if ademamix:
            exp_avg_slow = state["exp_avg_slow"]
            exp_avg_slow.lerp_(grad, 1.0 - beta3)
            if update_mt is not None:
                update = update_mt.add_(exp_avg_slow, alpha=alpha_mix)
            else:
                update = torch.add(grad, exp_avg_slow, alpha=alpha_mix)
        else:
            update = update_mt if update_mt is not None else grad.clone()

        # Second moment
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Denominator
        if group["use_atan2"]:
            denom = exp_avg_sq.sqrt().div_(sqrt_bc2)
            update.atan2_(denom)
        else:
            denom = exp_avg_sq.sqrt().div_(sqrt_bc2).add_(group["eps"])
            update.div_(denom)

        return update

    def _step_factored(
        self, grad: Tensor, state: dict, group: dict,
        beta1: float, beta2: float, sqrt_bc2: float,
        ademamix: bool, beta3: float, alpha_mix: float,
    ) -> Tensor:
        d1, d2 = state["effective_shape"]
        grad_2d = grad.view(d1, d2)

        # First moment
        if beta1 > 0:
            mt = _reconstruct_state((state["mu_m"], state["mv_m"], state["sign_m"], d2), signed=True)
            mt.lerp_(grad_2d, 1.0 - beta1)
            state["mu_m"], state["mv_m"], state["sign_m"] = _factorize_state(mt.clone(), signed=True)

            if self._grams_moment:
                update_mt = _grams_update(mt, grad_2d)
            elif self._cautious_mask:
                update_mt = _cautious_update(mt, grad_2d)
            else:
                update_mt = mt
        else:
            update_mt = None

        # Second moment
        vt = _reconstruct_state((state["mu_v"], state["mv_v"]), signed=False)
        vt.mul_(beta2).addcmul_(grad_2d, grad_2d, value=1.0 - beta2)

        # AdEMAMix slow moment
        if ademamix:
            mt_slow = _reconstruct_state(
                (state["mu_m_slow"], state["mv_m_slow"], state["sign_m_slow"], d2), signed=True,
            )
            mt_slow.lerp_(grad_2d, 1.0 - beta3)
            if update_mt is not None:
                update = update_mt.add_(mt_slow, alpha=alpha_mix)
            else:
                update = grad_2d.add(mt_slow, alpha=alpha_mix)
            state["mu_m_slow"], state["mv_m_slow"], state["sign_m_slow"] = _factorize_state(mt_slow, signed=True)
        else:
            update = update_mt if update_mt is not None else grad_2d.clone()

        # Re-factorize second moment
        state["mu_v"], state["mv_v"] = _factorize_state(vt, signed=False)

        # Denominator
        if group["use_atan2"]:
            denom = vt.sqrt_().div_(sqrt_bc2)
            update.atan2_(denom)
        else:
            denom = vt.sqrt_().div_(sqrt_bc2).add_(group["eps"])
            update.div_(denom)

        return update

    def _apply_update(
        self, p: Tensor, group: dict, update: Tensor, lr: float,
    ) -> None:
        """Apply weight decay and parameter update, with optional stochastic rounding."""
        wd = group["weight_decay"]
        cautious = group.get("cautious_wd", False)
        scaled_wd = wd * lr

        if p.dtype == torch.bfloat16 and self._stochastic_rounding:
            p_fp32 = p.float()
            update_fp32 = update.float()

            if wd != 0:
                if cautious:
                    mask = (update_fp32 * p_fp32 >= 0).float()
                    p_fp32.addcmul_(p_fp32, mask, value=-scaled_wd)
                else:
                    p_fp32.add_(p_fp32, alpha=-scaled_wd)

            p_fp32.add_(-update_fp32)
            _copy_stochastic_(p, p_fp32)
        else:
            if wd != 0:
                if cautious:
                    mask = (update * p >= 0).to(p.dtype)
                    p.addcmul_(p, mask, value=-scaled_wd)
                else:
                    p.add_(p, alpha=-scaled_wd)
            p.add_(-update)

    # ------------------------------------------------------------------
    # Standard step() interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)

        return loss
