"""Configuration models and utilities."""
from trainer.config.schema import (
    TrainConfig, ModelConfig, TrainingConfig, OptimizerConfig,
    NetworkConfig, DataConfig, DatasetEntry, SavingConfig,
    SamplingConfig, LoggingConfig,
)
from trainer.config.io import load_config, save_config, apply_overrides
from trainer.config.validation import validate_config, ValidationResult, ValidationIssue
