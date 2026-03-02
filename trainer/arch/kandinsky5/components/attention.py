"""Kandinsky 5 attention backends.

Ported from Musubi_Tuner's kandinsky5/models/attention.py.
Improvements over source:
- Removed duplicate _ENABLE_COMPILE / _maybe_compile (defined centrally in nn.py).
- `SelfAttentionEngine.__init__` uses a match/case for clarity.
- Graceful degradation: flash/sage/xformers fall back rather than crash at import.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional attention back-end imports
# ---------------------------------------------------------------------------

try:
    from flash_attn import flash_attn_func as _flash_attention_2
except Exception:
    _flash_attention_2 = None  # type: ignore[assignment]

try:
    from flash_attn_interface import flash_attn_func as _flash_attention_3
except Exception:
    _flash_attention_3 = None  # type: ignore[assignment]

try:
    import sageattention as _sageattention
except Exception:
    _sageattention = None  # type: ignore[assignment]

try:
    import xformers.ops as _xops
except Exception:
    _xops = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Attention kernel wrappers (SDPA, sage, xformers)
# ---------------------------------------------------------------------------

def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None):
    """Standard PyTorch scaled-dot-product attention. Layout: [B, S, H, D]."""
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    return out.transpose(1, 2).contiguous()


def sage_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """SageAttention (NHD layout)."""
    return _sageattention.sageattn(q, k, v, tensor_layout="NHD", is_causal=False)  # type: ignore[union-attr]


def xformers_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None):
    """xFormers memory-efficient attention (NHD layout)."""
    if attn_mask is not None:
        return _xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask)  # type: ignore[union-attr]
    return _xops.memory_efficient_attention(q, k, v)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# SelfAttentionEngine — unified dispatch object
# ---------------------------------------------------------------------------

_VALID_ENGINES = frozenset({
    "auto", "flash_attention_2", "flash_attention_3", "sage", "sdpa", "xformers"
})


class SelfAttentionEngine:
    """Dispatch object that selects and wraps the best available attention kernel.

    The ``engine`` argument controls backend selection:
    - ``"auto"``               Picks the best available back-end.
    - ``"flash_attention_2"``  Uses flash-attn v2 (requires flash_attn package).
    - ``"flash_attention_3"``  Uses flash-attn v3 (requires flash_attn_interface).
    - ``"sage"``               Uses SageAttention.
    - ``"xformers"``           Uses xFormers memory-efficient attention.
    - ``"sdpa"``               Uses torch F.scaled_dot_product_attention.

    Attributes:
        attention_fn: Callable that matches the expected (q, k, v[, attn_mask]) signature.
        supports_mask: Whether attention_fn accepts an ``attn_mask`` argument.
    """

    def __init__(self, engine: str = "auto") -> None:
        if engine not in _VALID_ENGINES:
            raise ValueError(
                f"Unknown attention engine '{engine}'. "
                f"Choose from: {sorted(_VALID_ENGINES)}"
            )

        self.attention_fn = None
        self.supports_mask = False

        if engine == "flash_attention_2":
            if _flash_attention_2 is None:
                raise RuntimeError("flash_attention_2 selected but flash_attn is not installed.")
            self.attention_fn = _flash_attention_2

        elif engine == "flash_attention_3":
            if _flash_attention_3 is None:
                raise RuntimeError("flash_attention_3 selected but flash_attn_interface is not installed.")
            self.attention_fn = _flash_attention_3

        elif engine == "sage":
            if _sageattention is None:
                raise RuntimeError("sage selected but sageattention is not installed.")
            self.attention_fn = sage_attn

        elif engine == "xformers":
            if _xops is None:
                raise RuntimeError("xformers selected but xformers is not installed.")
            self.attention_fn = xformers_attn

        elif engine == "sdpa":
            self.attention_fn = sdpa
            self.supports_mask = True

        elif engine == "auto":
            # Preference order: flash3 > flash2 > sage > xformers > sdpa
            self.attention_fn = sdpa
            self.supports_mask = True
            if _xops is not None:
                self.attention_fn = xformers_attn
                self.supports_mask = False
            if _sageattention is not None:
                self.attention_fn = sage_attn
                self.supports_mask = False
            if _flash_attention_2 is not None:
                self.attention_fn = _flash_attention_2
                self.supports_mask = False
            if _flash_attention_3 is not None:
                self.attention_fn = _flash_attention_3
                self.supports_mask = False

        if self.attention_fn is None:
            # Should never reach here, but be safe.
            logger.warning("SelfAttentionEngine: no kernel resolved; falling back to sdpa.")
            self.attention_fn = sdpa
            self.supports_mask = True

    def get_attention(self):
        return self.attention_fn
