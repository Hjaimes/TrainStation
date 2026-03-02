"""Tests for FastAPI routes using TestClient."""
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


class TestModelsEndpoint:
    def test_models_returns_list(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert "wan" in data["models"]


class TestConfigEndpoints:
    def test_validate_valid_config(self, client):
        config = {
            "config": {
                "model": {"architecture": "wan", "base_model_path": "."},
                "training": {"method": "lora"},
                "network": {"rank": 16, "alpha": 16.0},
                "data": {"datasets": [{"path": "."}]},
            }
        }
        resp = client.post("/api/config/validate", json=config)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)

    def test_validate_invalid_config(self, client):
        resp = client.post("/api/config/validate", json={"config": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_defaults_endpoint(self, client):
        resp = client.get("/api/config/defaults/wan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"]["architecture"] == "wan"


class TestTrainingEndpoints:
    def test_status_idle(self, client):
        resp = client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alive"] is False
        assert data["exit_message"] is None

    def test_start_while_running(self, client):
        app.state.runner.start.side_effect = RuntimeError("Training is already running.")
        resp = client.post("/api/training/start", json={"config": {}, "mode": "train"})
        assert resp.status_code == 400
        assert "already running" in resp.json()["error"]

    def test_stop(self, client):
        resp = client.post("/api/training/stop")
        assert resp.status_code == 200
        app.state.runner.send_stop.assert_called_once()

    def test_pause(self, client):
        resp = client.post("/api/training/pause")
        assert resp.status_code == 200
        app.state.runner.send_pause.assert_called_once()

    def test_resume(self, client):
        resp = client.post("/api/training/resume")
        assert resp.status_code == 200
        app.state.runner.send_resume.assert_called_once()

    def test_save(self, client):
        resp = client.post("/api/training/save")
        assert resp.status_code == 200
        app.state.runner.send_save.assert_called_once()
