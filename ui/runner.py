"""Subprocess training runner. Manages the training process lifecycle."""
from __future__ import annotations
import logging
import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection

from trainer.events import (
    ErrorEvent, StopCommand, PauseCommand, ResumeCommand, SaveCommand,
)

logger = logging.getLogger(__name__)

_ctx = mp.get_context("spawn")

# Exit code -> crash message mapping
_CRASH_MESSAGES: dict[int, str] = {
    -9: "Process killed (likely OOM). Try reducing batch_size or enabling block_swap.",
    -11: "Segmentation fault (likely GPU driver issue). Update your GPU drivers.",
    3221225477: "Access violation (likely GPU driver crash). Update your GPU drivers.",
}


def _training_worker(pipe: Connection, config_dict: dict, mode: str) -> None:
    """Run training in a subprocess. Module-level for spawn picklability.

    Order matters:
    1. Install PipeLoggingHandler FIRST so all logs route through pipe
    2. Deferred imports to avoid CUDA fork issues
    3. Run training session
    """
    # 1. Capture all logging into the pipe
    from trainer.util import PipeLoggingHandler
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(PipeLoggingHandler(pipe))

    try:
        # 2. Deferred imports (avoid top-level torch/CUDA in parent)
        from trainer.config.io import load_config_from_dict
        from trainer.training.session import TrainingSession
        from trainer.util.pipe_callback import PipeCallback

        config = load_config_from_dict(config_dict)
        session = TrainingSession()
        session.start(config, callbacks=[PipeCallback(pipe)], mode=mode)
    except Exception as exc:
        try:
            pipe.send(ErrorEvent(
                message=str(exc),
                traceback_str=traceback.format_exc(),
                is_fatal=True,
            ))
        except (BrokenPipeError, OSError):
            pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


class SubprocessTrainingRunner:
    """Manages a training subprocess from the UI process."""

    def __init__(self) -> None:
        self._process: mp.Process | None = None
        self._parent_conn: Connection | None = None
        self._exit_message: str | None = None

    def start(self, config_dict: dict, mode: str = "train") -> None:
        if self.is_alive():
            raise RuntimeError("Training is already running.")

        parent_conn, child_conn = _ctx.Pipe()
        self._parent_conn = parent_conn
        self._exit_message = None

        self._process = _ctx.Process(
            target=_training_worker,
            args=(child_conn, config_dict, mode),
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # Parent must close child end or poll() hangs
        logger.info("Training subprocess started (pid=%d)", self._process.pid)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def poll_events(self) -> list:
        if self._parent_conn is None:
            return []
        events = []
        try:
            while self._parent_conn.poll():
                events.append(self._parent_conn.recv())
        except (EOFError, OSError):
            pass
        return events

    def send_stop(self) -> None:
        self._send_command(StopCommand())

    def send_pause(self) -> None:
        self._send_command(PauseCommand())

    def send_resume(self) -> None:
        self._send_command(ResumeCommand())

    def send_save(self) -> None:
        self._send_command(SaveCommand())

    def stop(self, timeout: float = 30.0) -> None:
        if not self.is_alive():
            return
        # Phase 1: graceful stop via command
        self.send_stop()
        self._process.join(timeout=timeout)
        # Phase 2: force terminate if still alive
        if self._process.is_alive():
            logger.warning("Training subprocess didn't stop gracefully, terminating.")
            self._process.terminate()
            self._process.join(timeout=5.0)

    def get_crash_message(self) -> str | None:
        if self._process is None:
            return None
        if self._process.is_alive():
            return None
        code = self._process.exitcode
        if code is None or code == 0:
            return None
        return _CRASH_MESSAGES.get(code, f"Process exited with code {code}")

    @property
    def exit_message(self) -> str | None:
        if self._exit_message is None:
            self._exit_message = self.get_crash_message()
        return self._exit_message

    def _send_command(self, command) -> None:
        if self._parent_conn is None:
            return
        try:
            self._parent_conn.send(command)
        except (BrokenPipeError, OSError):
            pass
