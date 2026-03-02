"""Single entry point for both CLI and UI subprocess."""
from __future__ import annotations
import logging
from typing import Any

from trainer.config.schema import TrainConfig
from trainer.callbacks import TrainingCallback
from trainer.errors import ConfigError, ModelLoadError, TrainerError

logger = logging.getLogger(__name__)


class TrainingSession:
    """Orchestrates: discover -> resolve strategy -> load model -> prepare method -> train.
    Both CLI and subprocess use this. No duplicated orchestration."""

    def start(
        self,
        config: TrainConfig,
        callbacks: TrainingCallback | list[TrainingCallback] | None = None,
        mode: str = "train",
    ) -> None:
        """Run training or preprocessing.

        Args:
            config: Complete training configuration.
            callbacks: One or more TrainingCallback instances.
            mode: "train", "cache-latents", "cache-text", "cache-all"
        """
        if isinstance(callbacks, TrainingCallback):
            callbacks = [callbacks]
        callbacks = callbacks or []

        try:
            self._run(config, callbacks, mode)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except (ConfigError, ModelLoadError, TrainerError):
            raise
        except Exception as e:
            raise TrainerError(f"Training failed: {e}") from e

    def _run(
        self, config: TrainConfig, callbacks: list[TrainingCallback], mode: str,
    ) -> None:
        # 1. Freeze config
        frozen = config.freeze()

        # 2. Pre-flight validation
        from trainer.config.validation import validate_config
        result = validate_config(frozen)
        if result.has_errors:
            msgs = [issue.message for issue in result.errors]
            raise ConfigError("Pre-flight validation failed:\n" +
                              "\n".join(f"  - {m}" for m in msgs))
        for issue in result.warnings:
            logger.warning(f"Config warning: {issue.message}")

        # 3. Discover and resolve strategy
        from trainer.registry import get_model_strategy
        StrategyClass = get_model_strategy(frozen.model.architecture)
        strategy = StrategyClass(frozen)

        # 4. Load model components
        logger.info(f"Loading model: {frozen.model.architecture}")
        self._notify_log(callbacks, "INFO", f"Loading {frozen.model.architecture} model...")
        components = strategy.setup()

        # 5. Handle preprocessing modes
        if mode != "train":
            self._run_preprocessing(mode, strategy, components, frozen, callbacks)
            return

        # 6. Prepare training method (LoRA injection or full finetune)
        from trainer.training.methods import create_training_method
        method = create_training_method(frozen)
        method_result = method.prepare(
            model=components.model,
            arch=strategy.architecture,
            learning_rate=frozen.optimizer.learning_rate,
            text_encoders=components.text_encoders or None,
        )

        # 7. Create and run trainer
        from trainer.training.trainer import Trainer
        trainer = Trainer(
            config=frozen,
            strategy=strategy,
            method_result=method_result,
            components=components,
            callbacks=callbacks,
        )
        trainer.run()

    def _run_preprocessing(self, mode, strategy, components, config, callbacks):
        if mode in ("cache-latents", "cache-all"):
            self._notify_log(callbacks, "INFO", "Caching latents...")
            # Strategy handles the iteration over the dataset
        if mode in ("cache-text", "cache-all"):
            self._notify_log(callbacks, "INFO", "Caching text encoder outputs...")
        self._notify_log(callbacks, "INFO", "Preprocessing complete.")

    @staticmethod
    def _notify_log(callbacks, level, message):
        for cb in callbacks:
            try:
                cb.on_log(level=level, message=message)
            except Exception:
                pass
