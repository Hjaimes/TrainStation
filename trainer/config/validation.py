"""Pre-flight validation. Checks filesystem, VRAM, arch availability.
Separate from Pydantic because it produces UI-actionable results."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from trainer.config.schema import TrainConfig


@dataclass
class ValidationIssue:
    level: str                              # "error", "warning", "info"
    message: str
    go_to_tab: str | None = None
    field_path: str | None = None
    fix_label: str | None = None
    fix_action: Callable | None = None


@dataclass
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def can_train(self) -> bool:
        return not self.has_errors

    def all_issues(self) -> list[ValidationIssue]:
        return self.errors + self.warnings + self.info


def validate_config(config: TrainConfig) -> ValidationResult:
    """Run all pre-flight checks. Returns structured results for UI display."""
    result = ValidationResult()

    # Path validation
    model_path = Path(config.model.base_model_path)
    if not model_path.exists():
        result.errors.append(ValidationIssue(
            level="error",
            message=f"Base model path does not exist: {model_path}",
            go_to_tab="Model", field_path="model.base_model_path",
        ))

    if config.model.vae_path:
        vae_path = Path(config.model.vae_path)
        if not vae_path.exists():
            result.errors.append(ValidationIssue(
                level="error",
                message=f"VAE path does not exist: {vae_path}",
                go_to_tab="Model", field_path="model.vae_path",
            ))

    output_dir = Path(config.saving.output_dir)
    if not output_dir.parent.exists():
        result.warnings.append(ValidationIssue(
            level="warning",
            message=f"Output directory parent does not exist: {output_dir.parent}. It will be created.",
            go_to_tab="Output", field_path="saving.output_dir",
        ))

    # Architecture availability
    from trainer.registry import list_models
    available = list_models()
    if available and config.model.architecture not in available:
        result.errors.append(ValidationIssue(
            level="error",
            message=f"Unknown architecture: '{config.model.architecture}'. Available: {available}",
            go_to_tab="Model", field_path="model.architecture",
        ))

    # VRAM warning
    effective_batch = config.training.batch_size * config.training.gradient_accumulation_steps
    if effective_batch > 8:
        result.warnings.append(ValidationIssue(
            level="warning",
            message=f"Effective batch size is {effective_batch}. May require significant VRAM.",
            go_to_tab="Training",
        ))

    # Sampling without prompts
    if config.sampling.enabled and len(config.sampling.prompts) == 0 and not config.sampling.prompts_file:
        result.warnings.append(ValidationIssue(
            level="warning",
            message="Sampling is enabled but no prompts are configured.",
            go_to_tab="Sampling",
        ))

    # Resume path
    if config.training.resume_from:
        if not Path(config.training.resume_from).exists():
            result.errors.append(ValidationIssue(
                level="error",
                message=f"Resume path does not exist: {config.training.resume_from}",
                go_to_tab="Training", field_path="training.resume_from",
            ))

    # Network rank sanity
    if config.network is not None and config.network.rank < 1:
        result.errors.append(ValidationIssue(
            level="error", message="network.rank must be >= 1",
            go_to_tab="Training", field_path="network.rank",
        ))

    # Info summary
    result.info.append(ValidationIssue(
        level="info",
        message=f"Architecture: {config.model.architecture}, Method: {config.training.method}, "
                f"Mixed precision: {config.training.mixed_precision}",
    ))

    return result
