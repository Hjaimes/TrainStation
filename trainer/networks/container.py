# NetworkContainer — unified container for LoRA/LoHa/LoKr modules.
# Ported and simplified from Musubi_Tuner's LoRANetwork class.
# Does NOT do architecture detection — receives target_modules from arch_configs.

import logging
import os
import re
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NetworkContainer(nn.Module):
    """Manages a collection of LoRA/LoHa/LoKr modules applied to a model.

    This container walks a model's named modules, finds matching Linear/Conv2d
    layers within target module classes, creates adapter modules, and replaces
    their forward methods.
    """

    def __init__(
        self,
        module_class: Type[nn.Module],
        target_modules: List[str],
        rank: int = 4,
        alpha: float = 1.0,
        multiplier: float = 1.0,
        dropout: float | None = None,
        rank_dropout: float | None = None,
        module_dropout: float | None = None,
        conv_rank: int | None = None,
        conv_alpha: float | None = None,
        exclude_patterns: List[str] | None = None,
        include_patterns: List[str] | None = None,
        module_kwargs: Dict[str, Any] | None = None,
        prefix: str = "lora_unet",
        verbose: bool = False,
    ):
        """Create a NetworkContainer.

        Args:
            module_class: The adapter class (LoRAModule, LoHaModule, or LoKrModule).
            target_modules: List of module class names to target (e.g. ["WanAttentionBlock"]).
            rank: Rank (dim) for the low-rank decomposition.
            alpha: Scaling alpha for the adapter.
            multiplier: Multiplier applied to the adapter output.
            dropout: Standard dropout probability.
            rank_dropout: Per-rank dropout probability.
            module_dropout: Probability of skipping the entire adapter per forward.
            conv_rank: Separate rank for Conv2d layers. None = skip Conv2d 3x3.
            conv_alpha: Separate alpha for Conv2d layers.
            exclude_patterns: Regex patterns; matching module names are excluded.
            include_patterns: Regex patterns; matching module names are always included
                even if they match an exclude pattern.
            module_kwargs: Extra keyword arguments passed to module_class constructor.
            prefix: Prefix for lora_name keys in state dict.
            verbose: If True, log detailed module creation info.
        """
        super().__init__()
        self.module_class = module_class
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.conv_rank = conv_rank
        self.conv_alpha = conv_alpha
        self.module_kwargs = module_kwargs or {}
        self.prefix = prefix
        self.verbose = verbose

        self.loraplus_lr_ratio: float | None = None

        # Compile regex patterns
        self._exclude_re: List[re.Pattern] = []
        if exclude_patterns:
            for pattern in exclude_patterns:
                try:
                    self._exclude_re.append(re.compile(pattern))
                except re.error as e:
                    logger.error(f"Invalid exclude pattern '{pattern}': {e}")

        self._include_re: List[re.Pattern] = []
        if include_patterns:
            for pattern in include_patterns:
                try:
                    self._include_re.append(re.compile(pattern))
                except re.error as e:
                    logger.error(f"Invalid include pattern '{pattern}': {e}")

        # Populated by apply_to()
        self._lora_modules: List[nn.Module] = []
        self._original_forwards: Dict[str, Any] = {}  # lora_name -> (org_module, org_forward)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lora_modules(self) -> List[nn.Module]:
        """List of all created adapter modules."""
        return list(self._lora_modules)

    # ------------------------------------------------------------------
    # Core: apply_to / restore
    # ------------------------------------------------------------------

    def apply_to(self, model: nn.Module) -> None:
        """Walk the model, create adapter modules for matching layers, and
        replace their forward methods.

        Args:
            model: The model (e.g. UNet/DiT) to apply adapters to.
        """
        loras: List[nn.Module] = []
        skipped: List[str] = []

        for name, module in model.named_modules():
            if module.__class__.__name__ not in self.target_modules:
                continue

            for child_name, child_module in module.named_modules():
                is_linear = child_module.__class__.__name__ == "Linear"
                is_conv2d = child_module.__class__.__name__ == "Conv2d"
                is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                if not (is_linear or is_conv2d):
                    continue

                original_name = (name + "." if name else "") + child_name
                lora_name = f"{self.prefix}.{original_name}".replace(".", "_")

                # Exclude/include filtering
                excluded = any(p.search(original_name) for p in self._exclude_re)
                included = any(p.search(original_name) for p in self._include_re)
                if excluded and not included:
                    if self.verbose:
                        logger.info(f"exclude: {original_name}")
                    continue

                # Determine rank and alpha for this layer
                dim: int | None = None
                alpha_val: float | None = None

                if is_linear or is_conv2d_1x1:
                    dim = self.rank
                    alpha_val = self.alpha
                elif is_conv2d and self.conv_rank is not None:
                    dim = self.conv_rank
                    alpha_val = self.conv_alpha

                if dim is None or dim == 0:
                    if is_linear or is_conv2d_1x1 or self.conv_rank is not None:
                        skipped.append(lora_name)
                    continue

                lora = self.module_class(
                    lora_name,
                    child_module,
                    self.multiplier,
                    dim,
                    alpha_val,
                    dropout=self.dropout,
                    rank_dropout=self.rank_dropout,
                    module_dropout=self.module_dropout,
                    **self.module_kwargs,
                )
                loras.append(lora)

        if len(loras) == 0:
            raise RuntimeError(
                f"No adapter modules were created. Check target_modules={self.target_modules} "
                f"and exclude/include patterns."
            )

        # Verify no duplicate names
        names = set()
        for lora in loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        # Apply: replace forward methods and register as submodules
        for lora in loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        self._lora_modules = loras

        logger.info(f"Created {len(loras)} adapter modules ({self.module_class.__name__}).")
        if self.verbose:
            for lora in loras:
                logger.info(f"\t{lora.lora_name:50} dim={lora.lora_dim}, alpha={lora.alpha.item()}")

        if self.verbose and skipped:
            logger.warning(f"Skipped {len(skipped)} modules (dim=0):")
            for name in skipped:
                logger.info(f"\t{name}")

    def restore(self, model: nn.Module) -> None:
        """Restore original forward methods on all wrapped modules.

        Note: This walks the model again and restores any modules whose forward
        was replaced by one of our adapters. The adapter modules remain registered
        as submodules of this container but are no longer called.
        """
        # Build a set of our forward functions for identification
        our_forwards = {lora.forward for lora in self._lora_modules}

        for _name, module in model.named_modules():
            if hasattr(module, "forward") and module.forward in our_forwards:
                # Each adapter stored org_forward during apply_to
                for lora in self._lora_modules:
                    if module.forward is lora.forward and hasattr(lora, "org_forward"):
                        module.forward = lora.org_forward
                        break

        logger.info("Restored original forward methods.")

    # ------------------------------------------------------------------
    # Optimizer params
    # ------------------------------------------------------------------

    def set_loraplus_lr_ratio(self, ratio: float) -> None:
        """Set LoRA+ learning rate ratio. The 'up' parameters get lr * ratio."""
        self.loraplus_lr_ratio = ratio
        logger.info(f"LoRA+ LR Ratio: {ratio}")

    # Pre-compiled regex for extracting block index from lora_name.
    _BLOCK_RE: re.Pattern = re.compile(r"blocks?[._](\d+)", re.IGNORECASE)

    def prepare_optimizer_params(
        self,
        unet_lr: float = 1e-4,
        block_lr_multipliers: list[float] | None = None,
    ) -> tuple[List[Dict], List[str]]:
        """Return optimizer param groups and their descriptions.

        Supports:
        - LoRA+ (separate lr for lora_up weights) when loraplus_lr_ratio is set.
        - Per-block LR multipliers when block_lr_multipliers is provided.

        When block_lr_multipliers is set, the effective LR for each LoRA module is
        ``unet_lr * block_lr_multipliers[block_idx]`` where block_idx is parsed from
        the module's lora_name using the pattern ``blocks?[._](\\d+)``. Modules with
        no parseable block index, or whose block index exceeds the multiplier list,
        fall back to the base unet_lr. LoRA+ ratio is applied on top of the
        block-specific LR.

        Params with the same effective LR are merged into a single param group to
        minimise the number of optimizer state entries.

        Args:
            unet_lr: Base learning rate for LoRA parameters.
            block_lr_multipliers: Optional per-block multipliers indexed by block number.

        Returns:
            Tuple of (param_groups, descriptions) where param_groups is a list of
            dicts with 'params' and optional 'lr' keys.
        """
        self.requires_grad_(True)

        if block_lr_multipliers is not None:
            return self._prepare_params_with_block_lr(unet_lr, block_lr_multipliers)

        # --- Fast path: no per-block multipliers ---
        param_groups: Dict[str, Dict[str, torch.nn.Parameter]] = {"lora": {}, "plus": {}}
        for lora in self._lora_modules:
            for name, param in lora.named_parameters():
                if self.loraplus_lr_ratio is not None and "lora_up" in name:
                    param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                else:
                    param_groups["lora"][f"{lora.lora_name}.{name}"] = param

        if self.loraplus_lr_ratio is not None and len(param_groups["plus"]) == 0:
            logger.warning("LoRA+ is not effective for this network type (no 'lora_up' parameters found)")

        all_params: List[Dict] = []
        descriptions: List[str] = []

        for key in param_groups:
            params = list(param_groups[key].values())
            if not params:
                continue

            param_data: Dict[str, Any] = {"params": params}

            if unet_lr is not None:
                if key == "plus" and self.loraplus_lr_ratio is not None:
                    param_data["lr"] = unet_lr * self.loraplus_lr_ratio
                else:
                    param_data["lr"] = unet_lr

            if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                logger.info("LR is 0 or None, skipping param group")
                continue

            all_params.append(param_data)
            desc = "unet"
            if key == "plus":
                desc += " plus"
            descriptions.append(desc)

        return all_params, descriptions

    def _get_block_lr(self, lora_name: str, base_lr: float, multipliers: list[float]) -> float:
        """Extract block index from lora_name and return the effective LR.

        Args:
            lora_name: The lora module name, e.g. ``lora_unet_blocks_0_self_attn_q_proj``.
            base_lr: Fallback learning rate when no block index is found or index is out of range.
            multipliers: List of per-block multipliers; index maps to block number.

        Returns:
            Effective learning rate for this module.
        """
        match = self._BLOCK_RE.search(lora_name)
        if match is None:
            return base_lr
        block_idx = int(match.group(1))
        if block_idx >= len(multipliers):
            return base_lr
        return base_lr * multipliers[block_idx]

    def _prepare_params_with_block_lr(
        self,
        unet_lr: float,
        block_lr_multipliers: list[float],
    ) -> tuple[List[Dict], List[str]]:
        """Build param groups when per-block LR multipliers are active.

        Params with the same effective LR are merged into one group (keyed by the
        rounded float value) to keep the optimizer state compact.
        """
        has_plus_params = False

        # lr_key -> list[param]  (separate buckets for lora and plus)
        lora_buckets: Dict[float, list[torch.nn.Parameter]] = {}
        plus_buckets: Dict[float, list[torch.nn.Parameter]] = {}

        for lora in self._lora_modules:
            block_lr = self._get_block_lr(lora.lora_name, unet_lr, block_lr_multipliers)

            for name, param in lora.named_parameters():
                is_up = self.loraplus_lr_ratio is not None and "lora_up" in name
                if is_up:
                    effective_lr = block_lr * self.loraplus_lr_ratio
                    bucket = plus_buckets.setdefault(effective_lr, [])
                    bucket.append(param)
                    has_plus_params = True
                else:
                    bucket = lora_buckets.setdefault(block_lr, [])
                    bucket.append(param)

        if self.loraplus_lr_ratio is not None and not has_plus_params:
            logger.warning("LoRA+ is not effective for this network type (no 'lora_up' parameters found)")

        all_params: List[Dict] = []
        descriptions: List[str] = []

        for lr_val, params in sorted(lora_buckets.items()):
            if lr_val == 0:
                logger.info("Block LR resolved to 0, skipping param group")
                continue
            all_params.append({"params": params, "lr": lr_val})
            descriptions.append(f"unet block_lr={lr_val:.2e}")

        for lr_val, params in sorted(plus_buckets.items()):
            if lr_val == 0:
                logger.info("Block LR (plus) resolved to 0, skipping param group")
                continue
            all_params.append({"params": params, "lr": lr_val})
            descriptions.append(f"unet plus block_lr={lr_val:.2e}")

        return all_params, descriptions

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_weights(self, path: str, dtype: torch.dtype | None = None, metadata: Dict[str, str] | None = None) -> None:
        """Save adapter weights to a safetensors file.

        Args:
            path: Output file path. Must end with .safetensors.
            dtype: If set, cast all weights to this dtype before saving.
            metadata: Optional metadata dict to embed in the safetensors file.
        """
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(path)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, path, metadata)
        else:
            torch.save(state_dict, path)

    def load_weights(self, path: str) -> Any:
        """Load adapter weights from a file.

        Args:
            path: Path to .safetensors or .pt/.bin file.

        Returns:
            Info from load_state_dict (missing/unexpected keys).
        """
        if os.path.splitext(path)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(path)
        else:
            weights_sd = torch.load(path, map_location="cpu")

        info = self.load_state_dict(weights_sd, strict=False)
        return info

    # ------------------------------------------------------------------
    # Max norm regularization
    # ------------------------------------------------------------------

    def apply_max_norm_regularization(self, max_norm: float, device: torch.device) -> tuple[int, float, float]:
        """Clamp LoRA weight norms to prevent explosion (scale_weight_norms).

        Only supported for LoRA (lora_down/lora_up parameterization).

        Args:
            max_norm: Maximum allowed norm for the composed up@down weight.
            device: Device to perform computation on.

        Returns:
            Tuple of (keys_scaled, average_norm, max_observed_norm).
        """
        # Collect (down_weight, up_weight, alpha) directly from modules — no state_dict() copy
        triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for lora in self._lora_modules:
            down = getattr(lora, "lora_down", None)
            up = getattr(lora, "lora_up", None)
            alpha = getattr(lora, "alpha", None)
            if down is None or up is None or alpha is None:
                continue
            down_w = down.weight if hasattr(down, "weight") else down
            up_w = up.weight if hasattr(up, "weight") else up
            triples.append((down_w, up_w, alpha))

        if not triples:
            logger.warning("max_norm_regularization is only supported for LoRA")
            return 0, 0.0, 0.0

        norms: List[float] = []
        keys_scaled = 0

        for down, up, alpha in triples:
            down = down.to(device)
            up = up.to(device)
            alpha_val = alpha.to(device)
            dim = down.shape[0]
            scale = alpha_val / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm / 2)
            desired = torch.clamp(norm, max=max_norm)
            ratio = desired / norm
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                up.data *= sqrt_ratio
                down.data *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def set_multiplier(self, multiplier: float) -> None:
        """Update the multiplier on all adapter modules."""
        self.multiplier = multiplier
        for lora in self._lora_modules:
            lora.multiplier = multiplier

    def prepare_grad_etc(self) -> None:
        """Enable gradients for training."""
        self.requires_grad_(True)

    def on_epoch_start(self) -> None:
        """Called at the start of each training epoch."""
        self.train()

    def get_trainable_params(self):
        """Return all trainable parameters."""
        return self.parameters()
