"""Tests for samples API (Task 15)."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path


def test_samples_route_import():
    from ui.routes.samples import router  # noqa: F401


def test_list_samples_empty(tmp_path):
    """Empty directory returns empty list."""
    from ui.routes.samples import list_samples

    result = asyncio.get_event_loop().run_until_complete(
        list_samples(output_dir=str(tmp_path))
    )
    assert result == []


def test_list_samples_nonexistent_dir():
    """Non-existent directory returns empty list."""
    from ui.routes.samples import list_samples

    result = asyncio.get_event_loop().run_until_complete(
        list_samples(output_dir="/nonexistent/path")
    )
    assert result == []


def test_list_samples_finds_samples(tmp_path):
    """Finds files with 'sample' in name and image extensions."""
    from ui.routes.samples import list_samples

    # Create sample files
    (tmp_path / "sample_step100.png").write_bytes(b"fake png")
    (tmp_path / "sample_step200.jpg").write_bytes(b"fake jpg")
    # This should NOT be found (no 'sample' in name)
    (tmp_path / "checkpoint.safetensors").write_bytes(b"fake model")
    # This should NOT be found (wrong extension)
    (tmp_path / "sample_data.txt").write_text("not an image")

    result = asyncio.get_event_loop().run_until_complete(
        list_samples(output_dir=str(tmp_path))
    )
    assert len(result) == 2
    filenames = {s["filename"] for s in result}
    assert "sample_step100.png" in filenames
    assert "sample_step200.jpg" in filenames


def test_list_samples_sorted_most_recent_first(tmp_path):
    """Samples are sorted by modified time, most recent first."""
    from ui.routes.samples import list_samples

    # Create files with different modification times
    f1 = tmp_path / "sample_old.png"
    f1.write_bytes(b"old")
    time.sleep(0.1)
    f2 = tmp_path / "sample_new.png"
    f2.write_bytes(b"new")

    result = asyncio.get_event_loop().run_until_complete(
        list_samples(output_dir=str(tmp_path))
    )
    assert len(result) == 2
    assert result[0]["filename"] == "sample_new.png"
    assert result[1]["filename"] == "sample_old.png"


def test_list_samples_finds_in_subdirs(tmp_path):
    """Finds samples in subdirectories."""
    from ui.routes.samples import list_samples

    sub = tmp_path / "run1" / "samples"
    sub.mkdir(parents=True)
    (sub / "sample_001.png").write_bytes(b"nested")

    result = asyncio.get_event_loop().run_until_complete(
        list_samples(output_dir=str(tmp_path))
    )
    assert len(result) == 1


def test_get_media_type():
    from ui.routes.samples import _get_media_type

    assert _get_media_type(".png") == "image/png"
    assert _get_media_type(".jpg") == "image/jpeg"
    assert _get_media_type(".mp4") == "video/mp4"
    assert _get_media_type(".xyz") == "application/octet-stream"


def test_sample_extensions():
    from ui.routes.samples import SAMPLE_EXTENSIONS

    assert ".png" in SAMPLE_EXTENSIONS
    assert ".jpg" in SAMPLE_EXTENSIONS
    assert ".mp4" in SAMPLE_EXTENSIONS
