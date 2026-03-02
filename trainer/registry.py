"""Decorator-based model registry with auto-discovery."""
from __future__ import annotations
from typing import Type
import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)
_model_strategies: dict[str, Type] = {}
_discovered: bool = False


def register_model(name: str):
    """Decorator: @register_model("wan") on a ModelStrategy subclass."""
    def decorator(cls):
        if name in _model_strategies:
            logger.warning(f"Model registry: overwriting '{name}' ({_model_strategies[name].__name__} -> {cls.__name__})")
        _model_strategies[name] = cls
        return cls
    return decorator


def get_model_strategy(name: str) -> Type:
    """Get a registered strategy class by name. Auto-discovers if needed."""
    if not _discovered:
        discover_architectures()
    if name not in _model_strategies:
        available = ", ".join(sorted(_model_strategies.keys()))
        raise KeyError(f"Model '{name}' not found. Available: [{available}]")
    return _model_strategies[name]


def list_models() -> list[str]:
    """List all registered architecture names."""
    if not _discovered:
        discover_architectures()
    return sorted(_model_strategies.keys())


def discover_architectures() -> None:
    """Auto-import all architecture packages under trainer.arch."""
    global _discovered
    if _discovered:
        return
    try:
        import trainer.arch as arch_pkg
    except ImportError:
        logger.warning("trainer.arch package not found.")
        _discovered = True
        return
    for _, name, is_pkg in pkgutil.iter_modules(arch_pkg.__path__):
        if is_pkg and not name.startswith("_"):
            try:
                importlib.import_module(f"trainer.arch.{name}")
            except Exception as e:
                logger.error(f"Failed to load architecture '{name}': {e}", exc_info=True)
    _discovered = True
    logger.info(f"Discovered {len(_model_strategies)} architecture(s): {list(_model_strategies.keys())}")
