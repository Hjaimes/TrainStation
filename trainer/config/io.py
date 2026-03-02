"""Config loading, saving, override application, and env var substitution."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any

from trainer.config.schema import TrainConfig


def load_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from YAML or JSON. Supports ${ENV_VAR} substitution."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text)
    elif path.suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unknown config format: {path.suffix}. Use .yaml or .json")

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(raw).__name__}")

    raw.pop("_preset", None)  # Strip preset metadata if present
    raw = _substitute_env_vars(raw)
    return TrainConfig.model_validate(raw)


def load_config_from_dict(d: dict[str, Any]) -> TrainConfig:
    """Construct TrainConfig from a plain dict. Used by subprocess worker."""
    return TrainConfig.model_validate(d)


def save_config(config: TrainConfig, path: str | Path) -> None:
    """Save a TrainConfig to YAML or JSON."""
    path = Path(path)
    data = config.model_dump(exclude_none=False)

    if path.suffix in (".yaml", ".yml"):
        import yaml
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    elif path.suffix == ".json":
        text = json.dumps(data, indent=2)
    else:
        raise ValueError(f"Unknown config format: {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def apply_overrides(config: TrainConfig, overrides: list[str]) -> TrainConfig:
    """Apply dot-notation overrides like 'training.batch_size=4'."""
    data = config.model_dump()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override}")
        key, value = override.split("=", 1)
        parts = key.strip().split(".")
        target = data
        for part in parts[:-1]:
            if part not in target:
                raise ValueError(f"Unknown config path: {key}")
            target = target[part]
        field_name = parts[-1]
        if field_name not in target:
            raise ValueError(f"Unknown config field: {key}")
        target[field_name] = _coerce(value.strip(), target[field_name])
    return TrainConfig.model_validate(data)


def _coerce(value: str, reference: Any) -> Any:
    """Coerce string value to match the type of reference."""
    if reference is None:
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    if isinstance(reference, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, list):
        return json.loads(value)
    return value


def _substitute_env_vars(data: Any) -> Any:
    """Recursively substitute ${VAR_NAME} in strings."""
    if isinstance(data, str):
        def replacer(m):
            val = os.environ.get(m.group(1))
            if val is None:
                raise ValueError(f"Environment variable '{m.group(1)}' is not set")
            return val
        return re.sub(r"\$\{(\w+)\}", replacer, data)
    if isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_substitute_env_vars(i) for i in data]
    return data
