"""The training engine. Created by TrainingSession, not by users directly."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import logging
import math
import shutil
import time

import collections
import torch
from accelerate import Accelerator

from trainer.config.schema import TrainConfig
from trainer.arch.base import ModelStrategy, ModelComponents, TrainStepOutput
from trainer.training.methods import TrainingMethodResult
from trainer.callbacks import TrainingCallback, StepMetrics
from trainer.events import StopCommand, PauseCommand, ResumeCommand, SampleCommand, SaveCommand
from trainer.errors import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Minimal DTO for checkpoint resume."""
    global_step: int = 0
    current_epoch: int = 0
    best_loss: float = float("inf")


class _LossRecorder:
    """Moving average loss tracker."""
    def __init__(self, window: int = 100):
        self._losses: collections.deque[float] = collections.deque(maxlen=window)

    def add(self, loss: float) -> None:
        self._losses.append(loss)

    @property
    def moving_average(self) -> float:
        return sum(self._losses) / max(len(self._losses), 1)


def _get_optimizer_mode_fns(optimizer):
    """Get train/eval mode functions for optimizers that support them (e.g. Prodigy)."""
    if hasattr(optimizer, "train") and callable(optimizer.train):
        return optimizer.train, optimizer.eval
    return (lambda: None), (lambda: None)


def _scale_lr(base_lr: float, batch_size: int, grad_accum: int, method: str) -> float:
    """Scale learning rate by effective batch size.

    Args:
        base_lr: The base learning rate from config.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        method: Scaling method - "none", "linear", or "sqrt".

    Returns:
        Scaled learning rate.

    Raises:
        ConfigError: If ``method`` is not a recognised scaling mode.
    """
    effective = batch_size * grad_accum
    match method:
        case "none":
            return base_lr
        case "linear":
            return base_lr * effective
        case "sqrt":
            return base_lr * math.sqrt(effective)
        case _:
            raise ConfigError(f"Unknown lr_scaling method: {method!r}. Valid options: none, linear, sqrt")


class Trainer:
    """Orchestrates Accelerate + training loop + checkpointing + sampling."""

    def __init__(
        self, config: TrainConfig, strategy: ModelStrategy,
        method_result: TrainingMethodResult, components: ModelComponents,
        callbacks: list[TrainingCallback],
    ):
        self.config = config
        self.strategy = strategy
        self.method_result = method_result
        self.components = components
        self.callbacks = callbacks
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_loss: float = float("inf")
        self._stop_requested: bool = False
        self._paused: bool = False
        self._pending_sample: bool = False
        self._pending_save: bool = False
        self._ema_tracker = None

    def run(self) -> None:
        """Full training run. Blocking."""
        cfg = self.config

        # 1. Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            mixed_precision=cfg.training.mixed_precision,
            log_with=cfg.logging.log_with,
            project_dir=cfg.logging.logging_dir,
        )

        # 2. Seed
        if cfg.training.seed is not None:
            from accelerate.utils import set_seed
            set_seed(cfg.training.seed)

        # 3. Optimizer
        from trainer.optimizers import create_optimizer
        lr = _scale_lr(
            cfg.optimizer.learning_rate,
            cfg.training.batch_size,
            cfg.training.gradient_accumulation_steps,
            cfg.optimizer.lr_scaling,
        )
        if cfg.optimizer.lr_scaling != "none":
            logger.info(
                "LR scaling (%s): %.6e -> %.6e (batch=%d, grad_accum=%d)",
                cfg.optimizer.lr_scaling, cfg.optimizer.learning_rate, lr,
                cfg.training.batch_size, cfg.training.gradient_accumulation_steps,
            )
        optimizer = create_optimizer(
            cfg.optimizer.optimizer_type,
            self.method_result.trainable_params,
            lr=lr,
            weight_decay=cfg.optimizer.weight_decay,
            **cfg.optimizer.optimizer_kwargs,
        )

        if cfg.training.stochastic_rounding:
            from trainer.util.stochastic_rounding import register_stochastic_rounding_hook
            register_stochastic_rounding_hook(optimizer)
            logger.info("Stochastic rounding enabled for BF16 parameters")

        optimizer_train_fn, optimizer_eval_fn = _get_optimizer_mode_fns(optimizer)

        # Fused backward: step each param immediately during backward to avoid
        # holding all gradients in memory at once (~25-40% VRAM savings).
        fused_bwd = None
        if cfg.training.fused_backward:
            from trainer.training.fused_backward import FusedBackwardManager
            if accelerator.num_processes > 1:
                logger.warning(
                    "Fused backward is single-GPU only and bypasses Accelerate "
                    "gradient sync. Disabling for multi-GPU run."
                )
            else:
                if cfg.training.gradient_accumulation_steps > 1:
                    logger.warning(
                        "Fused backward has no VRAM benefit with "
                        "gradient_accumulation_steps > 1 because full gradients "
                        "must be accumulated across micro-steps."
                    )
                if cfg.training.max_grad_norm > 0:
                    logger.warning(
                        "Fused backward is incompatible with gradient clipping "
                        "(max_grad_norm=%.4f). Gradients are freed immediately "
                        "during backward and cannot be globally normalised.",
                        cfg.training.max_grad_norm,
                    )
                fused_bwd = FusedBackwardManager(optimizer)
                fused_bwd.register()
                logger.info("Fused backward enabled")

        # Activation offloading setup
        activation_offload_ctx = None
        if cfg.model.activation_offloading:
            from trainer.util.activation_offload import ActivationOffloadContext
            activation_offload_ctx = ActivationOffloadContext(enabled=True)
            logger.info("Activation offloading enabled (saved tensors offloaded to CPU pinned memory)")

        # EMA setup
        if cfg.training.ema_enabled:
            from trainer.ema import EMATracker
            self._ema_tracker = EMATracker(
                self.method_result.get_trainable_params_flat(),
                decay=cfg.training.ema_decay,
                device=cfg.training.ema_device,
            )
            logger.info("EMA enabled: decay=%.6f, device=%s", cfg.training.ema_decay, cfg.training.ema_device)

        # 4. Dataloader
        from trainer.data.loader import create_dataloader
        dataloader = create_dataloader(
            config=cfg.data, strategy=self.strategy,
            components=self.components, batch_size=cfg.training.batch_size,
        )

        # 4b. Validation dataloader + runner (optional)
        val_runner = None
        if cfg.validation.enabled and cfg.validation.data_path:
            from trainer.training.validation import ValidationRunner
            from trainer.config.schema import DataConfig
            val_data_cfg = cfg.data.model_copy(
                update={"dataset_config_path": cfg.validation.data_path, "datasets": []}
            )
            val_dataloader = create_dataloader(
                config=val_data_cfg, strategy=self.strategy,
                components=self.components, batch_size=cfg.training.batch_size,
            )
            val_runner = ValidationRunner(
                strategy=self.strategy, components=self.components,
                dataloader=val_dataloader,
                num_steps=cfg.validation.num_steps,
            )
            logger.info(
                "Validation enabled: every %d steps, %d batches per run",
                cfg.validation.interval_steps, cfg.validation.num_steps,
            )

        # 4c. Regularization dataloader (optional, for prior preservation)
        reg_iter = None
        if cfg.data.reg_data_path:
            from trainer.data.loader import RegDataIterator
            reg_data_cfg = cfg.data.model_copy(
                update={"dataset_config_path": cfg.data.reg_data_path, "datasets": []}
            )
            reg_dataloader = create_dataloader(
                config=reg_data_cfg, strategy=self.strategy,
                components=self.components, batch_size=cfg.training.batch_size,
            )
            reg_iter = RegDataIterator(reg_dataloader)
            logger.info(
                "Regularization data loaded from %s (%d batches)",
                cfg.data.reg_data_path, len(reg_iter),
            )

        # 5. Before accelerate prepare
        hints = self.strategy.on_before_accelerate_prepare(self.components, accelerator)

        # 6. Accelerate prepare
        prepared_model = accelerator.prepare(
            self.components.model, device_placement=[hints.get("device_placement", True)],
        )
        self.components.model = prepared_model

        if self.method_result.network is not None:
            network, optimizer, dataloader = accelerator.prepare(
                self.method_result.network, optimizer, dataloader)
            self.method_result.network = network
            training_model = network
        else:
            optimizer, dataloader = accelerator.prepare(optimizer, dataloader)
            training_model = prepared_model

        # 7. After accelerate prepare
        self.strategy.on_after_accelerate_prepare(self.components, accelerator)

        # 8. Total steps + scheduler
        total_steps = self._calc_total_steps(cfg, dataloader)
        warmup = cfg.optimizer.warmup_steps or int(total_steps * cfg.optimizer.warmup_ratio)
        from trainer.schedulers import create_scheduler
        scheduler = create_scheduler(
            cfg.optimizer.scheduler_type, optimizer,
            num_training_steps=total_steps, warmup_steps=warmup,
            min_lr_ratio=cfg.optimizer.min_lr_ratio,
        )
        scheduler = accelerator.prepare(scheduler)

        # 9. Accelerate hooks + resume
        self._register_accelerate_hooks(accelerator)
        if cfg.training.resume_from:
            self._resume(accelerator, cfg.training.resume_from)

        # 10. Notify start
        self._notify("on_training_start",
                      architecture=self.strategy.architecture, method=cfg.training.method,
                      total_steps=total_steps, output_dir=str(cfg.saving.output_dir),
                      config_dict=cfg.to_dict())

        # 11. Sample at first
        if cfg.sampling.sample_at_first and cfg.sampling.enabled:
            optimizer_eval_fn()
            self._generate_samples(accelerator)
            optimizer_train_fn()

        # 12. Training loop
        _MAX_OOM_RETRIES = 3
        num_epochs = self._calc_num_epochs(total_steps, dataloader)
        optimizer_train_fn()
        loss_recorder = _LossRecorder()

        try:
            for epoch in range(self.current_epoch, num_epochs):
                if self._stop_requested:
                    break
                self.current_epoch = epoch
                self._notify("on_epoch_start", epoch=epoch)
                oom_count = 0

                for step_in_epoch, batch in enumerate(dataloader):
                    self._check_commands()
                    if self._stop_requested:
                        optimizer_eval_fn()
                        self._save_checkpoint(accelerator, f"stop-{self.global_step}")
                        break
                    while self._paused and not self._stop_requested:
                        time.sleep(0.1)
                        self._check_commands()

                    if self._pending_save:
                        optimizer_eval_fn()
                        self._save_checkpoint(accelerator, f"manual-{self.global_step}")
                        optimizer_train_fn()
                        self._pending_save = False
                    if self._pending_sample:
                        optimizer_eval_fn()
                        self._generate_samples(accelerator)
                        optimizer_train_fn()
                        self._pending_sample = False

                    try:
                        with accelerator.accumulate(training_model):
                            self.strategy.on_before_training_step(self.components)
                            if activation_offload_ctx is not None:
                                with activation_offload_ctx:
                                    output = self.strategy.training_step(self.components, batch, self.global_step)
                            else:
                                output = self.strategy.training_step(self.components, batch, self.global_step)

                            # Regularization: separate forward pass, combine losses before backward
                            if reg_iter is not None:
                                reg_batch = reg_iter.next_batch()
                                if activation_offload_ctx is not None:
                                    with activation_offload_ctx:
                                        reg_output = self.strategy.training_step(
                                            self.components, reg_batch, self.global_step
                                        )
                                else:
                                    reg_output = self.strategy.training_step(
                                        self.components, reg_batch, self.global_step
                                    )
                                combined_loss = output.loss + reg_output.loss * cfg.data.prior_loss_weight
                                output = TrainStepOutput(
                                    loss=combined_loss,
                                    metrics={**output.metrics, "reg_loss": reg_output.loss.detach()},
                                )

                            accelerator.backward(output.loss)

                            if fused_bwd is None:
                                # Normal path: clip grads, step, zero.
                                if accelerator.sync_gradients and cfg.training.max_grad_norm > 0:
                                    accelerator.clip_grad_norm_(
                                        self.method_result.get_trainable_params_flat(),
                                        cfg.training.max_grad_norm)
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad(set_to_none=True)
                            else:
                                # Fused path: hooks already stepped and zeroed
                                # each param during backward. Just advance the
                                # scheduler; do NOT call optimizer.step() or
                                # optimizer.zero_grad().
                                scheduler.step()

                        oom_count = 0  # reset on success
                    except torch.cuda.OutOfMemoryError:
                        oom_count += 1
                        logger.warning(
                            "OOM on step %d (%d/%d retries)",
                            self.global_step, oom_count, _MAX_OOM_RETRIES)
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        if oom_count >= _MAX_OOM_RETRIES:
                            raise
                        continue

                    # Weight norm scaling
                    if (accelerator.sync_gradients and cfg.network
                            and cfg.network.scale_weight_norms is not None
                            and self.method_result.network is not None):
                        unwrapped = accelerator.unwrap_model(self.method_result.network)
                        if hasattr(unwrapped, "apply_max_norm_regularization"):
                            unwrapped.apply_max_norm_regularization(
                                cfg.network.scale_weight_norms, accelerator.device)

                    if accelerator.sync_gradients:
                        self.global_step += 1

                        # EMA update
                        if self._ema_tracker is not None:
                            self._ema_tracker.step(
                                self.method_result.get_trainable_params_flat(), self.global_step)

                        loss_val = output.loss.detach().item()
                        loss_recorder.add(loss_val)
                        if loss_val < self.best_loss:
                            self.best_loss = loss_val
                        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0

                        self._notify("on_step_end", metrics=StepMetrics(
                            step=self.global_step, total_steps=total_steps,
                            loss=loss_val, avg_loss=loss_recorder.moving_average,
                            lr=lr, epoch=epoch, epoch_step=step_in_epoch,
                            wall_time=time.time(), extra=output.metrics))

                        should_sample = self._should_sample()
                        should_save = self._should_save_step()
                        if should_sample or should_save:
                            optimizer_eval_fn()
                            if should_sample:
                                self._generate_samples(accelerator)
                            if should_save:
                                self._save_checkpoint(accelerator, f"step-{self.global_step}")
                            optimizer_train_fn()

                        if (val_runner is not None
                                and self.global_step % cfg.validation.interval_steps == 0):
                            optimizer_eval_fn()
                            val_metrics = val_runner.run(self.global_step)
                            self._notify("on_validation_end", step=self.global_step, metrics=val_metrics)
                            optimizer_train_fn()

                        if self.global_step >= total_steps:
                            break

                # Epoch end
                self._notify("on_epoch_end", epoch=epoch, avg_loss=loss_recorder.moving_average)
                optimizer_eval_fn()
                if self._should_save_epoch(epoch):
                    self._save_checkpoint(accelerator, f"epoch-{epoch + 1}")
                optimizer_train_fn()

                if self.global_step >= total_steps:
                    break

        except torch.cuda.OutOfMemoryError:
            logger.error(f"GPU OOM at step {self.global_step}")
            if fused_bwd is not None and fused_bwd.is_registered:
                fused_bwd.remove()
            try:
                self._save_checkpoint(accelerator, "emergency-oom")
            except Exception:
                logger.error("Failed to save emergency checkpoint.", exc_info=True)
            raise
        except Exception:
            logger.error(f"Error at step {self.global_step}", exc_info=True)
            if fused_bwd is not None and fused_bwd.is_registered:
                fused_bwd.remove()
            try:
                self._save_checkpoint(accelerator, "emergency-error")
            except Exception:
                logger.error("Failed to save emergency checkpoint.", exc_info=True)
            raise

        # Finalize
        if fused_bwd is not None and fused_bwd.is_registered:
            fused_bwd.remove()
        optimizer_eval_fn()
        self._save_final(accelerator)
        self._notify("on_training_end",
                      final_step=self.global_step, final_loss=loss_recorder.moving_average,
                      output_dir=str(cfg.saving.output_dir))
        accelerator.end_training()

    # --- Notifications ---

    def _notify(self, callback_method: str, metrics=None, **kwargs) -> None:
        for cb in self.callbacks:
            try:
                fn = getattr(cb, callback_method)
                if metrics is not None:
                    fn(metrics)
                else:
                    fn(**kwargs)
            except Exception as e:
                logger.warning(f"Callback error in {callback_method}: {e}")

    # --- Commands ---

    def _check_commands(self) -> None:
        for cb in self.callbacks:
            for cmd in cb.check_for_commands():
                if isinstance(cmd, StopCommand):
                    self._stop_requested = True
                elif isinstance(cmd, PauseCommand):
                    self._paused = True
                    logger.info("Training paused.")
                elif isinstance(cmd, ResumeCommand):
                    self._paused = False
                    logger.info("Training resumed.")
                elif isinstance(cmd, SampleCommand):
                    self._pending_sample = True
                elif isinstance(cmd, SaveCommand):
                    self._pending_save = True

    # --- Checkpointing ---

    def _save_checkpoint(self, accelerator, tag: str) -> None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_dir = Path(self.config.saving.output_dir) / f"checkpoint-{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        params_flat = self.method_result.get_trainable_params_flat()
        if self._ema_tracker is not None:
            self._ema_tracker.copy_to(params_flat)
        try:
            self.method_result.save_fn(
                str(ckpt_dir / f"{self.config.saving.output_name}.safetensors"),
                {"arch": self.strategy.architecture, "method": self.config.training.method,
                 "global_step": str(self.global_step), "epoch": str(self.current_epoch)})
        finally:
            if self._ema_tracker is not None:
                self._ema_tracker.restore(params_flat)

        state = CheckpointState(self.global_step, self.current_epoch, self.best_loss)
        torch.save(asdict(state), ckpt_dir / "training_state.pt")
        if self._ema_tracker is not None:
            torch.save(self._ema_tracker.state_dict(), ckpt_dir / "ema_state.pt")
        accelerator.save_state(str(ckpt_dir / "accelerate_state"))
        self._notify("on_checkpoint_saved", path=str(ckpt_dir), step=self.global_step)
        self._rotate_checkpoints()

    def _save_final(self, accelerator) -> None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        out = Path(self.config.saving.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        final_path = out / f"{self.config.saving.output_name}.safetensors"

        params_flat = self.method_result.get_trainable_params_flat()
        if self._ema_tracker is not None:
            self._ema_tracker.copy_to(params_flat)
        try:
            self.method_result.save_fn(str(final_path), {
                "arch": self.strategy.architecture, "total_steps": str(self.global_step)})
        finally:
            if self._ema_tracker is not None:
                self._ema_tracker.restore(params_flat)

        self.method_result.cleanup_fn()
        logger.info("Final weights saved: %s", final_path)

    def _rotate_checkpoints(self) -> None:
        max_keep = self.config.saving.max_keep_ckpts
        if max_keep is None:
            return
        output_dir = Path(self.config.saving.output_dir)
        ckpts = sorted(
            [d for d in output_dir.glob("checkpoint-step-*") if d.is_dir()],
            key=lambda p: p.stat().st_mtime)
        while len(ckpts) > max_keep:
            shutil.rmtree(ckpts.pop(0))

    # --- Accelerate hooks ---

    def _register_accelerate_hooks(self, accelerator) -> None:
        net = self.method_result.network
        if net is None:
            return
        net_type = type(accelerator.unwrap_model(net))
        def save_hook(models, weights, output_dir):
            for i in reversed(range(len(models))):
                if not isinstance(models[i], net_type) and len(weights) > i:
                    weights.pop(i)
        def load_hook(models, input_dir):
            for i in reversed(range(len(models))):
                if not isinstance(models[i], net_type):
                    models.pop(i)
        accelerator.register_save_state_pre_hook(save_hook)
        accelerator.register_load_state_pre_hook(load_hook)

    def _resume(self, accelerator, path: str) -> None:
        ckpt_dir = Path(path)
        state_path = ckpt_dir / "training_state.pt"
        accel_path = ckpt_dir / "accelerate_state"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
            state = CheckpointState(**state_dict)
            self.global_step = state.global_step
            self.current_epoch = state.current_epoch
            self.best_loss = state.best_loss
            logger.info(f"Resumed: step={self.global_step}, epoch={self.current_epoch}")
        if accel_path.exists():
            accelerator.load_state(str(accel_path))
        else:
            logger.warning(f"No accelerate_state at {path}, starting fresh.")

    # --- Schedule helpers ---

    def _calc_total_steps(self, cfg, dataloader) -> int:
        if cfg.training.max_steps is not None:
            return cfg.training.max_steps
        spe = max(math.ceil(len(dataloader) / cfg.training.gradient_accumulation_steps), 1)
        return spe * cfg.training.epochs

    def _calc_num_epochs(self, total_steps, dataloader) -> int:
        spe = max(math.ceil(len(dataloader) / self.config.training.gradient_accumulation_steps), 1)
        return max(math.ceil(total_steps / spe), 1)

    def _should_sample(self) -> bool:
        s = self.config.sampling
        if not s.enabled:
            return False
        if s.sample_every_n_steps is not None and self.global_step > 0:
            return self.global_step % s.sample_every_n_steps == 0
        return False

    def _should_save_step(self) -> bool:
        s = self.config.saving
        return (s.save_every_n_steps is not None and self.global_step > 0
                and self.global_step % s.save_every_n_steps == 0)

    def _should_save_epoch(self, epoch) -> bool:
        s = self.config.saving
        return s.save_every_n_epochs is not None and (epoch + 1) % s.save_every_n_epochs == 0

    def _generate_samples(self, accelerator) -> None:
        params_flat = self.method_result.get_trainable_params_flat()
        if self._ema_tracker is not None:
            self._ema_tracker.copy_to(params_flat)
        self.strategy.on_before_sampling(self.components)
        try:
            prompts = list(self.config.sampling.prompts)
            if not prompts and self.config.sampling.prompts_file:
                p = Path(self.config.sampling.prompts_file)
                if p.exists():
                    prompts = [l.strip() for l in p.read_text("utf-8").splitlines() if l.strip()]

            for i, prompt in enumerate(prompts):
                result = self.strategy.generate_sample(
                    self.components, prompt,
                    width=self.config.sampling.width, height=self.config.sampling.height,
                    num_frames=self.config.sampling.num_frames,
                    num_inference_steps=self.config.sampling.num_inference_steps,
                    guidance_scale=self.config.sampling.guidance_scale,
                    seed=self.config.sampling.seed,
                )
                sample_dir = Path(self.config.saving.output_dir) / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_path = sample_dir / f"step{self.global_step:06d}_p{i}.png"

                if hasattr(result, "save"):
                    result.save(str(save_path))
                elif isinstance(result, str):
                    save_path = Path(result)

                self._notify("on_sample_generated", path=str(save_path), step=self.global_step, prompt=prompt)

        except NotImplementedError:
            logger.debug("Sampling not implemented for %s.", self.strategy.architecture)
        except Exception as e:
            logger.warning("Sample generation failed: %s", e)
        finally:
            self.strategy.on_after_sampling(self.components)
            if self._ema_tracker is not None:
                self._ema_tracker.restore(params_flat)
