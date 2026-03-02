"""Preset system — builtin + user presets as partial YAML files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from trainer.config.schema import (
    TrainConfig,
    ModelConfig,
    NetworkConfig,
    DataConfig,
    DatasetEntry,
)

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override wins for non-dict values."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class PresetManager:
    """Manages builtin and user presets stored as partial YAML files."""

    def __init__(
        self,
        builtin_dir: str = "presets/builtin",
        user_dir: str = "presets/user",
    ) -> None:
        self._builtin = Path(builtin_dir)
        self._user = Path(user_dir)
        self._user.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> list[dict[str, str]]:
        """List all available presets with name and category."""
        result: list[dict[str, str]] = []
        for directory, category in [
            (self._builtin, "builtin"),
            (self._user, "user"),
        ]:
            if not directory.exists():
                continue
            for f in sorted(directory.glob("*.yaml")):
                result.append({
                    "name": f.stem,
                    "category": category,
                    "filename": f.name,
                })
        return result

    def load_preset(self, category: str, name: str) -> dict[str, Any]:
        """Load a preset and merge with TrainConfig defaults."""
        directory = self._builtin if category == "builtin" else self._user
        path = directory / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {category}/{name}")

        with open(path) as f:
            partial = yaml.safe_load(f) or {}

        return self._merge_with_defaults(partial)

    def save_user_preset(self, name: str, config: dict[str, Any]) -> None:
        """Save a config as a user preset."""
        path = self._user / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved user preset: %s", name)

    def delete_user_preset(self, name: str) -> None:
        """Delete a user preset."""
        path = self._user / f"{name}.yaml"
        if path.exists():
            path.unlink()
            logger.info("Deleted user preset: %s", name)

    def _merge_with_defaults(self, partial: dict[str, Any]) -> dict[str, Any]:
        """Merge a partial config over TrainConfig defaults."""
        arch = partial.get("model", {}).get("architecture", "wan")
        base_model_path = partial.get("model", {}).get(
            "base_model_path", "<select model>"
        )
        defaults = TrainConfig(
            model=ModelConfig(architecture=arch, base_model_path=base_model_path),
            network=NetworkConfig(),
            data=DataConfig(datasets=[DatasetEntry(path="<select dataset>")]),
        ).model_dump()
        return _deep_merge(defaults, partial)
