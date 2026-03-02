"""IPC protocol for subprocess communication. All dataclasses, all picklable.
Events: Training -> UI. Commands: UI -> Training."""
from __future__ import annotations
from dataclasses import dataclass, field
import time as _time


# --- Events: Training -> UI ---

@dataclass
class TrainingEvent:
    timestamp: float = field(default_factory=_time.time)

@dataclass
class TrainingStartedEvent(TrainingEvent):
    architecture: str = ""
    method: str = ""
    total_steps: int = 0
    output_dir: str = ""

@dataclass
class StepEvent(TrainingEvent):
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    avg_loss: float = 0.0
    lr: float = 0.0
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

@dataclass
class EpochEvent(TrainingEvent):
    epoch: int = 0
    avg_loss: float = 0.0
    is_start: bool = True

@dataclass
class SampleEvent(TrainingEvent):
    path: str = ""
    step: int = 0
    prompt: str = ""

@dataclass
class CheckpointEvent(TrainingEvent):
    path: str = ""
    step: int = 0

@dataclass
class LogEvent(TrainingEvent):
    level: str = "INFO"
    message: str = ""

@dataclass
class ErrorEvent(TrainingEvent):
    message: str = ""
    traceback_str: str = ""
    is_fatal: bool = True

@dataclass
class TrainingCompleteEvent(TrainingEvent):
    final_step: int = 0
    final_loss: float = 0.0
    output_dir: str = ""


# --- Commands: UI -> Training ---

@dataclass
class StopCommand:
    """Graceful stop: finish current step, save checkpoint, exit."""
    pass

@dataclass
class PauseCommand:
    pass

@dataclass
class ResumeCommand:
    pass

@dataclass
class SampleCommand:
    """Request on-demand sample generation."""
    prompt: str = ""
    seed: int = 42

@dataclass
class SaveCommand:
    """Request immediate checkpoint save."""
    pass
