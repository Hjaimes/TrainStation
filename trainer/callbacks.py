"""Unified callback system. Trainer calls these at lifecycle points.
All callback methods use keyword-only args (except metrics) for safety."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any
import sys
import time

import torch


@dataclass
class StepMetrics:
    """All data produced at each optimization step."""
    step: int
    total_steps: int
    loss: float
    avg_loss: float                              # Moving average, computed by Trainer
    lr: float
    epoch: int = 0
    epoch_step: int = 0
    wall_time: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


class TrainingCallback:
    """Base class for training callbacks. Override what you need."""

    def on_training_start(
        self, *, architecture: str, method: str, total_steps: int,
        output_dir: str, config_dict: dict[str, Any],
    ) -> None:
        pass

    def on_step_end(self, metrics: StepMetrics) -> None:
        pass

    def on_epoch_start(self, *, epoch: int) -> None:
        pass

    def on_epoch_end(self, *, epoch: int, avg_loss: float) -> None:
        pass

    def on_sample_generated(self, *, path: str, step: int, prompt: str) -> None:
        pass

    def on_checkpoint_saved(self, *, path: str, step: int) -> None:
        pass

    def on_validation_end(self, *, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_log(self, *, level: str, message: str) -> None:
        pass

    def on_error(self, *, message: str, traceback_str: str, is_fatal: bool) -> None:
        pass

    def on_training_end(self, *, final_step: int, final_loss: float, output_dir: str) -> None:
        pass

    def check_for_commands(self) -> list:
        """Poll for external commands (StopCommand, PauseCommand, etc.).
        Only PipeCallback overrides this."""
        return []


class CLIProgressCallback(TrainingCallback):
    """Prints training progress to stdout."""

    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self._start_time: float = 0.0

    def on_training_start(self, *, architecture, method, total_steps, output_dir, config_dict):
        self._start_time = time.time()
        print(f"Training {architecture} ({method}), {total_steps} steps -> {output_dir}")

    def on_step_end(self, metrics: StepMetrics) -> None:
        if metrics.step % self.log_every == 0 or metrics.step == metrics.total_steps:
            elapsed = time.time() - self._start_time
            it_s = metrics.step / elapsed if elapsed > 0 else 0
            eta = (metrics.total_steps - metrics.step) / it_s if it_s > 0 else 0
            print(f"  Step {metrics.step}/{metrics.total_steps} "
                  f"loss={metrics.loss:.4f} avg={metrics.avg_loss:.4f} "
                  f"lr={metrics.lr:.2e} {it_s:.2f}it/s ETA {eta:.0f}s")

    def on_epoch_end(self, *, epoch, avg_loss):
        print(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.6f}")

    def on_error(self, *, message, traceback_str, is_fatal):
        print(f"{'FATAL ' if is_fatal else ''}ERROR: {message}", file=sys.stderr)

    def on_training_end(self, *, final_step, final_loss, output_dir):
        elapsed = time.time() - self._start_time
        print(f"Done. {final_step} steps, loss={final_loss:.4f}, "
              f"{elapsed:.1f}s. Output: {output_dir}")


# ---------------------------------------------------------------------------
# VRAM profiling callback (formerly callbacks_vram.py)
# ---------------------------------------------------------------------------

_vram_logger = logging.getLogger(__name__ + ".vram")


class VRAMProfilerCallback(TrainingCallback):
    """Track peak VRAM usage per training step.

    After each step the peak allocation is read, the counter is reset so the
    next step starts from zero, and the value is stored in
    ``metrics.extra["vram_peak_mb"]``.

    Every ``log_every`` steps a fuller breakdown (allocated + reserved) is also
    emitted at INFO level so it shows up in the training log without flooding it.

    When CUDA is not available the callback is a no-op so it is always safe to
    attach regardless of hardware.
    """

    def __init__(self, log_every: int = 50) -> None:
        self._log_every = log_every

    def on_step_end(self, metrics: StepMetrics) -> None:
        if not torch.cuda.is_available():
            return

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()

        metrics.extra["vram_peak_mb"] = peak_mb

        if metrics.step % self._log_every == 0:
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            _vram_logger.info(
                "VRAM step %d: peak=%.0fMB alloc=%.0fMB reserved=%.0fMB",
                metrics.step,
                peak_mb,
                allocated,
                reserved,
            )
