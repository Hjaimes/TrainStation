"""Bridges TrainingCallback to multiprocessing.Pipe events."""
from __future__ import annotations
from multiprocessing.connection import Connection
from trainer.callbacks import TrainingCallback, StepMetrics
from trainer.events import (
    TrainingStartedEvent, StepEvent, EpochEvent, SampleEvent,
    CheckpointEvent, LogEvent, ErrorEvent, TrainingCompleteEvent,
)


class PipeCallback(TrainingCallback):
    def __init__(self, pipe: Connection):
        self.pipe = pipe

    def on_training_start(self, *, architecture, method, total_steps, output_dir, config_dict):
        self._send(TrainingStartedEvent(
            architecture=architecture, method=method,
            total_steps=total_steps, output_dir=output_dir))

    def on_step_end(self, metrics: StepMetrics) -> None:
        self._send(StepEvent(
            step=metrics.step, total_steps=metrics.total_steps,
            loss=metrics.loss, avg_loss=metrics.avg_loss, lr=metrics.lr,
            epoch=metrics.epoch, metrics=metrics.extra))

    def on_epoch_start(self, *, epoch):
        self._send(EpochEvent(epoch=epoch, is_start=True))

    def on_epoch_end(self, *, epoch, avg_loss):
        self._send(EpochEvent(epoch=epoch, avg_loss=avg_loss, is_start=False))

    def on_sample_generated(self, *, path, step, prompt):
        self._send(SampleEvent(path=path, step=step, prompt=prompt))

    def on_checkpoint_saved(self, *, path, step):
        self._send(CheckpointEvent(path=path, step=step))

    def on_log(self, *, level, message):
        self._send(LogEvent(level=level, message=message))

    def on_error(self, *, message, traceback_str, is_fatal):
        self._send(ErrorEvent(message=message, traceback_str=traceback_str, is_fatal=is_fatal))

    def on_training_end(self, *, final_step, final_loss, output_dir):
        self._send(TrainingCompleteEvent(
            final_step=final_step, final_loss=final_loss, output_dir=output_dir))

    def check_for_commands(self) -> list:
        commands = []
        try:
            while self.pipe.poll():
                commands.append(self.pipe.recv())
        except (EOFError, OSError):
            pass
        return commands

    def _send(self, event) -> None:
        try:
            self.pipe.send(event)
        except (BrokenPipeError, OSError):
            pass
