"""Tests for SubprocessTrainingRunner. All spawn-safe (module-level workers)."""
import multiprocessing as mp
from unittest.mock import MagicMock

import pytest

from trainer.events import StepEvent, LogEvent, TrainingCompleteEvent
from ui.runner import SubprocessTrainingRunner

_ctx = mp.get_context("spawn")


def _simple_worker(pipe):
    """Module-level worker that sends 3 known events and exits."""
    pipe.send(StepEvent(step=1, total_steps=10, loss=0.5, avg_loss=0.5, lr=1e-4))
    pipe.send(LogEvent(level="INFO", message="hello"))
    pipe.send(TrainingCompleteEvent(final_step=10, final_loss=0.1, output_dir="/out"))
    pipe.close()


class TestSubprocessRunner:
    def test_is_alive_false_initially(self):
        runner = SubprocessTrainingRunner()
        assert runner.is_alive() is False

    def test_poll_events_empty_initially(self):
        runner = SubprocessTrainingRunner()
        assert runner.poll_events() == []

    def test_start_and_poll_events(self):
        """Spawn a simple worker, verify events arrive via poll."""
        runner = SubprocessTrainingRunner()
        parent_conn, child_conn = _ctx.Pipe()
        runner._parent_conn = parent_conn

        proc = _ctx.Process(target=_simple_worker, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        runner._process = proc

        # Wait for subprocess to finish
        proc.join(timeout=10)

        events = runner.poll_events()
        assert len(events) == 3
        assert isinstance(events[0], StepEvent)
        assert events[0].step == 1
        assert isinstance(events[1], LogEvent)
        assert events[1].message == "hello"
        assert isinstance(events[2], TrainingCompleteEvent)
        assert events[2].final_step == 10

    def test_start_while_running_raises(self):
        runner = SubprocessTrainingRunner()
        runner._process = MagicMock()
        runner._process.is_alive.return_value = True

        with pytest.raises(RuntimeError, match="already running"):
            runner.start({"model": {"architecture": "wan", "base_model_path": "/x"}})

    def test_crash_message_oom(self):
        runner = SubprocessTrainingRunner()
        runner._process = MagicMock()
        runner._process.is_alive.return_value = False
        runner._process.exitcode = -9

        msg = runner.get_crash_message()
        assert msg is not None
        assert "OOM" in msg

    def test_crash_message_segfault(self):
        runner = SubprocessTrainingRunner()
        runner._process = MagicMock()
        runner._process.is_alive.return_value = False
        runner._process.exitcode = -11

        msg = runner.get_crash_message()
        assert msg is not None
        assert "Segmentation fault" in msg

    def test_crash_message_gpu_driver(self):
        runner = SubprocessTrainingRunner()
        runner._process = MagicMock()
        runner._process.is_alive.return_value = False
        runner._process.exitcode = 3221225477

        msg = runner.get_crash_message()
        assert msg is not None
        assert "GPU driver" in msg

    def test_crash_message_normal_exit(self):
        runner = SubprocessTrainingRunner()
        runner._process = MagicMock()
        runner._process.is_alive.return_value = False
        runner._process.exitcode = 0

        assert runner.get_crash_message() is None

    def test_crash_message_unknown_code(self):
        runner = SubprocessTrainingRunner()
        runner._process = MagicMock()
        runner._process.is_alive.return_value = False
        runner._process.exitcode = 42

        msg = runner.get_crash_message()
        assert msg is not None
        assert "42" in msg

    def test_exit_message_property(self):
        runner = SubprocessTrainingRunner()
        runner._process = MagicMock()
        runner._process.is_alive.return_value = False
        runner._process.exitcode = -9

        # First access computes and caches
        msg = runner.exit_message
        assert "OOM" in msg
        # Second access returns cached value
        assert runner.exit_message is msg
