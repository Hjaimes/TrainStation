"""Shared utilities: dtype resolution, logging, timing."""
from __future__ import annotations

import collections
import logging
import time
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Dtype helpers (formerly util/dtype.py)
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "fp32": torch.float32, "float32": torch.float32,
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}

_REVERSE_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}


def resolve_dtype(dtype_str: str | None) -> torch.dtype | None:
    if dtype_str is None:
        return None
    result = _DTYPE_MAP.get(dtype_str.lower())
    if result is None:
        raise ValueError(f"Unknown dtype: '{dtype_str}'. Available: {sorted(_DTYPE_MAP.keys())}")
    return result


def dtype_to_str(dtype: torch.dtype) -> str:
    return _REVERSE_DTYPE_MAP.get(dtype, str(dtype).replace("torch.", ""))


# ---------------------------------------------------------------------------
# Pipe logging handler (formerly util/log_capture.py)
# ---------------------------------------------------------------------------

class PipeLoggingHandler(logging.Handler):
    """Logging handler that forwards log records through a multiprocessing pipe."""

    def __init__(self, pipe: Connection, level: int = logging.INFO):
        super().__init__(level)
        self.pipe = pipe

    def emit(self, record: logging.LogRecord) -> None:
        from trainer.events import LogEvent
        try:
            message = self.format(record)
            self.pipe.send(LogEvent(timestamp=record.created, level=record.levelname, message=message))
        except (BrokenPipeError, OSError, RecursionError):
            pass


# ---------------------------------------------------------------------------
# Training timer (formerly util/timer.py)
# ---------------------------------------------------------------------------

class TrainingTimer:
    def __init__(self, window: int = 100):
        self._timestamps: collections.deque = collections.deque(maxlen=window)
        self._total_steps: int = 0

    def start(self, total_steps: int) -> None:
        self._total_steps = total_steps
        self._timestamps.clear()

    def step(self) -> None:
        self._timestamps.append(time.time())

    @property
    def it_per_sec(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / dt if dt > 0 else 0.0

    @property
    def eta_seconds(self) -> float:
        speed = self.it_per_sec
        if speed <= 0:
            return 0.0
        remaining = self._total_steps - len(self._timestamps)
        return remaining / speed
