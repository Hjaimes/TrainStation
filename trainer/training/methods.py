"""Training method implementations: LoRA and Full Finetune.
Two-step pattern: create_training_method(config) -> method.prepare(model, arch, lr) -> result."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import logging
import torch
import torch.nn as nn

from trainer.config.schema import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingMethodResult:
    """What a training method produces after preparing the model."""
    trainable_params: list[dict]                 # Optimizer parameter groups
    network: nn.Module | None
    save_fn: Callable[[str, dict[str, str]], None]
    cleanup_fn: Callable[[], None]

    _flat_params_cache: list[torch.nn.Parameter] | None = field(default=None, init=False, repr=False)

    def get_trainable_params_flat(self) -> list[torch.nn.Parameter]:
        if self._flat_params_cache is None:
            params = []
            for group in self.trainable_params:
                params.extend(group["params"])
            self._flat_params_cache = params
        return self._flat_params_cache


def create_training_method(config: TrainConfig):
    """Factory for training methods."""
    method = config.training.method
    if method == "lora":
        return LoRAMethod(config)
    elif method == "full_finetune":
        return FullFinetuneMethod(config)
    else:
        raise ValueError(f"Unknown training method: '{method}'. Available: lora, full_finetune")


class LoRAMethod:
    def __init__(self, config: TrainConfig):
        self.config = config

    def prepare(
        self,
        model: nn.Module,
        arch: str,
        learning_rate: float,
        text_encoders: list[nn.Module] | None = None,
    ) -> TrainingMethodResult:
        net_cfg = self.config.network
        if net_cfg is None:
            raise ValueError("LoRA training requires a network config section")

        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Freeze text encoders (TE LoRA will selectively enable params)
        if text_encoders:
            for te in text_encoders:
                for param in te.parameters():
                    param.requires_grad = False

        # Get architecture LoRA targeting
        from trainer.networks.arch_configs import ARCH_NETWORK_CONFIGS
        arch_config = ARCH_NETWORK_CONFIGS.get(arch)
        if arch_config is None:
            raise ValueError(f"No network config for '{arch}'. Available: {list(ARCH_NETWORK_CONFIGS.keys())}")

        # Build exclude patterns
        exclude = list(arch_config.get("default_exclude_patterns", []))
        if net_cfg.exclude_patterns:
            exclude.extend(net_cfg.exclude_patterns)

        # Create and apply network
        from trainer.networks.container import NetworkContainer
        from trainer.networks import get_module_class

        # use_dora=True with network_type="lora" activates DoRA (Weight-Decomposed LoRA)
        module_type = net_cfg.network_type
        if net_cfg.use_dora and module_type == "lora":
            module_type = "dora"
            logger.info("DoRA enabled: using DoRAModule (weight-decomposed LoRA)")

        network = NetworkContainer(
            module_class=get_module_class(module_type),
            target_modules=list(arch_config["target_modules"]),
            exclude_patterns=exclude,
            include_patterns=net_cfg.include_patterns or None,
            rank=net_cfg.rank,
            alpha=net_cfg.alpha,
            dropout=net_cfg.dropout,
            rank_dropout=net_cfg.rank_dropout,
            module_dropout=net_cfg.module_dropout,
        )
        network.apply_to(model)

        if net_cfg.network_weights:
            logger.info("Loading pre-trained network weights from %s", net_cfg.network_weights)
            network.load_weights(net_cfg.network_weights)

        if net_cfg.loraplus_lr_ratio is not None:
            network.set_loraplus_lr_ratio(net_cfg.loraplus_lr_ratio)

        trainable_params, _ = network.prepare_optimizer_params(
            unet_lr=learning_rate,
            block_lr_multipliers=net_cfg.block_lr_multipliers,
        )

        num_params = sum(sum(p.numel() for p in g["params"]) for g in trainable_params)
        logger.info(f"LoRA: {len(network.lora_modules)} modules, {num_params:,} params, rank={net_cfg.rank}")

        # Text encoder LoRA (when text encoder training is enabled)
        te_network: NetworkContainer | None = None
        te_networks: list[NetworkContainer] = []
        if (self.config.training.train_text_encoder
                and text_encoders
                and "te_target_modules" in arch_config):
            te_lr = self.config.training.text_encoder_lr or learning_rate
            te_target_modules = arch_config["te_target_modules"]

            for i, te_model in enumerate(text_encoders):
                te_net = NetworkContainer(
                    module_class=get_module_class(module_type),
                    target_modules=te_target_modules,
                    rank=net_cfg.rank,
                    alpha=net_cfg.alpha,
                    dropout=net_cfg.dropout,
                    rank_dropout=net_cfg.rank_dropout,
                    module_dropout=net_cfg.module_dropout,
                    prefix=f"lora_te{i}" if len(text_encoders) > 1 else "lora_te",
                )
                te_net.apply_to(te_model)
                te_networks.append(te_net)

                te_params, _ = te_net.prepare_optimizer_params(unet_lr=te_lr)
                trainable_params.extend(te_params)

                te_num = sum(sum(p.numel() for p in g["params"]) for g in te_params)
                logger.info(
                    "TE LoRA [%d]: %d modules, %d params, lr=%.2e",
                    i, len(te_net.lora_modules), te_num, te_lr,
                )

            # Keep first for backward compat in save_fn
            te_network = te_networks[0] if te_networks else None

        from trainer.util import resolve_dtype
        save_dtype = resolve_dtype(net_cfg.save_dtype) if net_cfg.save_dtype else None

        # Build save function that includes TE network weights
        _te_networks_for_save = list(te_networks)
        def _save_fn(path: str, meta: dict[str, str]) -> None:
            if _te_networks_for_save:
                # Merge all network state dicts (DiT + TE) into one file
                merged_sd = dict(network.state_dict())
                for te_net in _te_networks_for_save:
                    merged_sd.update(te_net.state_dict())
                if save_dtype is not None:
                    merged_sd = {k: v.detach().clone().to("cpu").to(save_dtype) for k, v in merged_sd.items()}
                else:
                    merged_sd = {k: v.detach().clone().to("cpu") for k, v in merged_sd.items()}
                import os
                if os.path.splitext(path)[1] == ".safetensors":
                    from safetensors.torch import save_file
                    save_file(merged_sd, path, meta if meta else None)
                else:
                    torch.save(merged_sd, path)
            else:
                network.save_weights(path, dtype=save_dtype, metadata=meta)

        # Build cleanup function that restores all networks
        _text_encoders_for_cleanup = list(text_encoders) if text_encoders else []
        def _cleanup_fn() -> None:
            network.restore(model)
            for te_net, te_model in zip(_te_networks_for_save, _text_encoders_for_cleanup):
                te_net.restore(te_model)

        return TrainingMethodResult(
            trainable_params=trainable_params,
            network=network,
            save_fn=_save_fn,
            cleanup_fn=_cleanup_fn,
        )


class FullFinetuneMethod:
    def __init__(self, config: TrainConfig):
        self.config = config

    def prepare(
        self,
        model: nn.Module,
        arch: str,
        learning_rate: float,
        text_encoders: list[nn.Module] | None = None,
    ) -> TrainingMethodResult:
        for param in model.parameters():
            param.requires_grad = True

        overrides = self.config.optimizer.component_lr_overrides
        if overrides:
            trainable_params = self._build_component_param_groups(model, learning_rate, overrides)
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            trainable_params = [{"params": trainable, "lr": learning_rate}]

        # Text encoder full finetune (when enabled)
        if self.config.training.train_text_encoder and text_encoders:
            te_lr = self.config.training.text_encoder_lr or learning_rate
            for i, te_model in enumerate(text_encoders):
                for param in te_model.parameters():
                    param.requires_grad = True
                te_params = [p for p in te_model.parameters() if p.requires_grad]
                if te_params:
                    trainable_params.append({"params": te_params, "lr": te_lr})
                    te_num = sum(p.numel() for p in te_params)
                    logger.info(
                        "TE full finetune [%d]: %d params, lr=%.2e", i, te_num, te_lr,
                    )

        num_params = sum(sum(p.numel() for p in g["params"]) for g in trainable_params)
        logger.info(f"Full finetune: {num_params:,} trainable parameters")
        if overrides:
            logger.info(f"Full finetune: {len(trainable_params)} param groups from component_lr_overrides")

        def save_fn(path: str, metadata: dict[str, str]) -> None:
            import safetensors.torch
            state_dict = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            safetensors.torch.save_file(state_dict, path, metadata=metadata or {})

        return TrainingMethodResult(
            trainable_params=trainable_params,
            network=None,
            save_fn=save_fn,
            cleanup_fn=lambda: None,
        )

    @staticmethod
    def _build_component_param_groups(
        model: nn.Module,
        base_lr: float,
        overrides: dict[str, float],
    ) -> list[dict]:
        """Build optimizer param groups using component name pattern matching.

        Each pattern in overrides is matched as a substring of the parameter name.
        Patterns are applied in order; the first match wins. Unmatched parameters
        fall into a final group at the base learning rate.

        Args:
            model: The model whose parameters will be grouped.
            base_lr: Learning rate for parameters not matched by any pattern.
            overrides: Mapping of name pattern -> learning rate override.

        Returns:
            List of optimizer param group dicts with 'params' and 'lr' keys.
        """
        groups: list[dict] = []
        matched_ids: set[int] = set()

        for pattern, override_lr in overrides.items():
            group_params: list[torch.nn.Parameter] = []
            for name, param in model.named_parameters():
                if not param.requires_grad or id(param) in matched_ids:
                    continue
                if pattern in name:
                    group_params.append(param)
                    matched_ids.add(id(param))
            if group_params:
                groups.append({"params": group_params, "lr": override_lr})
                logger.debug(
                    "component_lr_overrides: pattern=%r lr=%g params=%d",
                    pattern, override_lr, len(group_params),
                )

        # Remaining params at base lr
        remaining = [
            p for _n, p in model.named_parameters()
            if p.requires_grad and id(p) not in matched_ids
        ]
        if remaining:
            groups.append({"params": remaining, "lr": base_lr})

        return groups
