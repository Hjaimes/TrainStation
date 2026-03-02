"""Tests for preset system (Task 10)."""
from __future__ import annotations

from pathlib import Path

import yaml


def test_preset_manager_import():
    from ui.presets import PresetManager  # noqa: F401


def test_preset_manager_list_empty(tmp_path):
    from ui.presets import PresetManager

    mgr = PresetManager(
        builtin_dir=str(tmp_path / "builtin"),
        user_dir=str(tmp_path / "user"),
    )
    presets = mgr.list_presets()
    assert presets == []


def test_preset_manager_list_builtin(tmp_path):
    from ui.presets import PresetManager

    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "wan-lora-24gb.yaml").write_text(
        yaml.dump({"model": {"architecture": "wan"}})
    )
    mgr = PresetManager(builtin_dir=str(builtin), user_dir=str(tmp_path / "user"))
    presets = mgr.list_presets()
    assert len(presets) == 1
    assert presets[0]["name"] == "wan-lora-24gb"
    assert presets[0]["category"] == "builtin"


def test_preset_manager_save_and_list_user(tmp_path):
    from ui.presets import PresetManager

    mgr = PresetManager(
        builtin_dir=str(tmp_path / "builtin"),
        user_dir=str(tmp_path / "user"),
    )
    mgr.save_user_preset("my-preset", {"model": {"architecture": "wan"}})
    presets = mgr.list_presets()
    assert len(presets) == 1
    assert presets[0]["name"] == "my-preset"
    assert presets[0]["category"] == "user"


def test_preset_manager_load_merges_defaults(tmp_path):
    from ui.presets import PresetManager

    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "test.yaml").write_text(
        yaml.dump({"model": {"architecture": "wan", "base_model_path": "/my/model"}})
    )
    mgr = PresetManager(builtin_dir=str(builtin), user_dir=str(tmp_path / "user"))
    config = mgr.load_preset("builtin", "test")
    assert config["model"]["architecture"] == "wan"
    assert config["model"]["base_model_path"] == "/my/model"
    # Default sections should be present from merge
    assert "training" in config
    assert "optimizer" in config
    assert "saving" in config


def test_preset_manager_load_not_found(tmp_path):
    from ui.presets import PresetManager
    import pytest

    mgr = PresetManager(
        builtin_dir=str(tmp_path / "builtin"),
        user_dir=str(tmp_path / "user"),
    )
    with pytest.raises(FileNotFoundError):
        mgr.load_preset("builtin", "nonexistent")


def test_preset_manager_delete_user(tmp_path):
    from ui.presets import PresetManager

    mgr = PresetManager(
        builtin_dir=str(tmp_path / "builtin"),
        user_dir=str(tmp_path / "user"),
    )
    mgr.save_user_preset("delete-me", {"model": {"architecture": "wan"}})
    assert len(mgr.list_presets()) == 1
    mgr.delete_user_preset("delete-me")
    assert len(mgr.list_presets()) == 0


def test_preset_manager_delete_nonexistent(tmp_path):
    """Deleting a non-existent preset should not raise."""
    from ui.presets import PresetManager

    mgr = PresetManager(
        builtin_dir=str(tmp_path / "builtin"),
        user_dir=str(tmp_path / "user"),
    )
    mgr.delete_user_preset("never-existed")  # Should not raise


def test_deep_merge():
    from ui.presets import _deep_merge

    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}
