"""Flat dict intermediary between frontend and Pydantic config."""
from __future__ import annotations
from typing import Any, Callable


class ConfigBinder:
    """Bridges flat dotted-key dicts (frontend) and nested config dicts (Pydantic).

    Usage:
        binder = ConfigBinder()
        binder.load_from_dict({"model": {"architecture": "wan", "base_model_path": "/m"}})
        binder.get("model.architecture")  # "wan"
        binder.set("model.architecture", "flux")  # fires on_change callbacks
        nested = binder.to_config_dict()  # back to nested dict
    """

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._callbacks: list[Callable[[str, Any], None]] = []

    def load_from_dict(self, nested: dict[str, Any]) -> None:
        self._values.clear()
        self._flatten(nested, "")

    def to_config_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for dotted_key, value in self._values.items():
            parts = dotted_key.split(".")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def get(self, path: str, default: Any = None) -> Any:
        return self._values.get(path, default)

    def set(self, path: str, value: Any) -> None:
        self._values[path] = value
        for cb in self._callbacks:
            cb(path, value)

    def update_many(self, updates: dict[str, Any]) -> None:
        for path, value in updates.items():
            self._values[path] = value

    def on_change(self, callback: Callable[[str, Any], None]) -> None:
        self._callbacks.append(callback)

    def keys(self) -> list[str]:
        return list(self._values.keys())

    def _flatten(self, obj: dict[str, Any], prefix: str) -> None:
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten(value, full_key)
            else:
                # Lists, scalars, None - all are leaf values, never recursed
                self._values[full_key] = value
