"""Shared test fixtures."""
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def minimal_config_path(fixtures_dir):
    return fixtures_dir / "minimal_train.yaml"
