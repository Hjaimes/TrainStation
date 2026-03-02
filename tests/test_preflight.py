"""Tests for pre-flight validation endpoint (Task 12)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ui.server import app


@pytest.fixture
def client():
    """TestClient with mocked runner to avoid subprocess spawning."""
    mock_runner = MagicMock()
    mock_runner.is_alive.return_value = False
    mock_runner.exit_message = None
    mock_runner.poll_events.return_value = []

    with TestClient(app) as c:
        app.state.runner = mock_runner
        yield c


def _valid_config(tmp_path) -> dict:
    """Build a minimal valid config dict with real paths."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    return {
        "model": {
            "architecture": "wan",
            "base_model_path": str(model_dir),
        },
        "training": {"method": "lora"},
        "network": {"rank": 16, "alpha": 16.0},
        "data": {"datasets": [{"path": str(dataset_dir)}]},
        "saving": {"output_dir": str(output_dir)},
    }


def test_preflight_valid_config(client, tmp_path):
    """All checks pass with valid paths and config."""
    config = _valid_config(tmp_path)
    resp = client.post("/api/preflight/check", json={"config": config})
    assert resp.status_code == 200
    data = resp.json()
    assert data["can_start"] is True
    assert all(c["status"] != "error" for c in data["checks"])
    # Should have at least config validation, model path, dataset, output checks
    assert len(data["checks"]) >= 4


def test_preflight_missing_model_path(client, tmp_path):
    """Error when model path does not exist."""
    config = _valid_config(tmp_path)
    config["model"]["base_model_path"] = str(tmp_path / "nonexistent_model")
    resp = client.post("/api/preflight/check", json={"config": config})
    assert resp.status_code == 200
    data = resp.json()
    model_check = next(c for c in data["checks"] if c["name"] == "Model path")
    assert model_check["status"] == "error"
    assert "Not found" in model_check["message"]


def test_preflight_missing_dataset(client, tmp_path):
    """Error when dataset path does not exist."""
    config = _valid_config(tmp_path)
    config["data"]["datasets"] = [{"path": str(tmp_path / "nonexistent_dataset")}]
    resp = client.post("/api/preflight/check", json={"config": config})
    assert resp.status_code == 200
    data = resp.json()
    ds_check = next(c for c in data["checks"] if c["name"].startswith("Dataset"))
    assert ds_check["status"] == "error"
    assert "Not found" in ds_check["message"]


def test_preflight_invalid_config(client):
    """Pydantic validation error returns early with can_start=False."""
    # Missing required 'model' section entirely
    resp = client.post("/api/preflight/check", json={"config": {}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["can_start"] is False
    config_check = next(c for c in data["checks"] if c["name"] == "Config validation")
    assert config_check["status"] == "error"


def test_preflight_can_start_false_when_errors(client, tmp_path):
    """can_start is False when any check has error status."""
    config = _valid_config(tmp_path)
    # Make model path invalid to trigger an error
    config["model"]["base_model_path"] = str(tmp_path / "does_not_exist")
    resp = client.post("/api/preflight/check", json={"config": config})
    assert resp.status_code == 200
    data = resp.json()
    assert data["can_start"] is False
    has_error = any(c["status"] == "error" for c in data["checks"])
    assert has_error


def test_preflight_output_dir_warning(client, tmp_path):
    """Output directory that will be created gets a warning or ok, not error."""
    config = _valid_config(tmp_path)
    # Use a path that doesn't exist but whose parent is writable
    config["saving"]["output_dir"] = str(tmp_path / "new_output")
    resp = client.post("/api/preflight/check", json={"config": config})
    assert resp.status_code == 200
    data = resp.json()
    out_check = next(c for c in data["checks"] if c["name"] == "Output directory")
    assert out_check["status"] in ("ok", "warning")


def test_preflight_cross_validation_error(client, tmp_path):
    """Config cross-validation errors (e.g., lora without network) are caught."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    config = {
        "model": {
            "architecture": "wan",
            "base_model_path": str(model_dir),
        },
        "training": {"method": "lora"},
        # Missing 'network' section -- cross-validation should fail
        "data": {"datasets": [{"path": str(dataset_dir)}]},
    }
    resp = client.post("/api/preflight/check", json={"config": config})
    assert resp.status_code == 200
    data = resp.json()
    assert data["can_start"] is False
    config_check = next(c for c in data["checks"] if c["name"] == "Config validation")
    assert config_check["status"] == "error"
