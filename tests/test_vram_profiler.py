"""Tests for trainer/callbacks_vram.py - VRAM profiling callback."""
from __future__ import annotations

import pytest
import torch

from trainer.callbacks import StepMetrics, VRAMProfilerCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metrics(step: int = 1, total_steps: int = 100) -> StepMetrics:
    return StepMetrics(
        step=step,
        total_steps=total_steps,
        loss=0.5,
        avg_loss=0.5,
        lr=1e-4,
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestVRAMProfilerCallbackInstantiation:
    def test_default_log_every(self):
        cb = VRAMProfilerCallback()
        assert cb._log_every == 50

    def test_custom_log_every(self):
        cb = VRAMProfilerCallback(log_every=10)
        assert cb._log_every == 10

    def test_is_training_callback(self):
        from trainer.callbacks import TrainingCallback
        cb = VRAMProfilerCallback()
        assert isinstance(cb, TrainingCallback)


# ---------------------------------------------------------------------------
# on_step_end - no CUDA
# ---------------------------------------------------------------------------

class TestOnStepEndNoCUDA:
    """Verify the callback is a no-op when CUDA is unavailable."""

    def test_does_not_crash_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        cb = VRAMProfilerCallback()
        metrics = make_metrics(step=1)
        cb.on_step_end(metrics)  # must not raise

    def test_extra_dict_unchanged_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        cb = VRAMProfilerCallback()
        metrics = make_metrics(step=1)
        cb.on_step_end(metrics)
        assert "vram_peak_mb" not in metrics.extra

    def test_multiple_steps_without_cuda_no_crash(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        cb = VRAMProfilerCallback(log_every=5)
        for step in range(1, 20):
            cb.on_step_end(make_metrics(step=step))


# ---------------------------------------------------------------------------
# on_step_end - with CUDA (mocked)
# ---------------------------------------------------------------------------

class TestOnStepEndWithCUDA:
    """Mock CUDA calls to verify the callback logic without real GPU."""

    @pytest.fixture(autouse=True)
    def patch_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 512 * (1024 ** 2))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 256 * (1024 ** 2))
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 1024 * (1024 ** 2))
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    def test_vram_peak_stored_in_extra(self):
        cb = VRAMProfilerCallback()
        metrics = make_metrics(step=1)
        cb.on_step_end(metrics)
        assert "vram_peak_mb" in metrics.extra

    def test_vram_peak_value_correct(self):
        cb = VRAMProfilerCallback()
        metrics = make_metrics(step=1)
        cb.on_step_end(metrics)
        assert metrics.extra["vram_peak_mb"] == pytest.approx(512.0)

    def test_does_not_crash_on_log_step(self):
        cb = VRAMProfilerCallback(log_every=10)
        metrics = make_metrics(step=10)
        cb.on_step_end(metrics)  # should trigger INFO log, not raise
        assert metrics.extra["vram_peak_mb"] == pytest.approx(512.0)

    def test_non_log_step_still_stores_value(self):
        cb = VRAMProfilerCallback(log_every=50)
        metrics = make_metrics(step=7)
        cb.on_step_end(metrics)
        assert "vram_peak_mb" in metrics.extra


# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------

class TestVRAMProfilingConfigField:
    def test_logging_config_has_vram_profiling_field(self):
        from trainer.config.schema import LoggingConfig
        cfg = LoggingConfig()
        assert hasattr(cfg, "vram_profiling")
        assert cfg.vram_profiling is False

    def test_vram_profiling_can_be_enabled(self):
        from trainer.config.schema import LoggingConfig
        cfg = LoggingConfig(vram_profiling=True)
        assert cfg.vram_profiling is True
