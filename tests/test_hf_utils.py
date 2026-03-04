"""Tests for HuggingFace model ID detection and resolution utilities."""
import os
import tempfile
from pathlib import Path

import pytest

from trainer.util.hf_utils import is_huggingface_id, find_safetensors_in_dir


class TestIsHuggingFaceId:
    """Test HuggingFace model ID detection."""

    def test_standard_hf_ids(self):
        assert is_huggingface_id("Wan-AI/Wan2.1-T2V-14B")
        assert is_huggingface_id("stabilityai/stable-diffusion-xl-base-1.0")
        assert is_huggingface_id("black-forest-labs/FLUX.1-dev")
        assert is_huggingface_id("runwayml/stable-diffusion-v1-5")

    def test_hf_id_with_subfolder(self):
        assert is_huggingface_id("org/model/subfolder")

    def test_simple_org_model(self):
        assert is_huggingface_id("org/model")
        assert is_huggingface_id("a/b")

    def test_rejects_empty(self):
        assert not is_huggingface_id("")
        assert not is_huggingface_id("   ")

    def test_rejects_unix_paths(self):
        assert not is_huggingface_id("/home/user/models/wan")
        assert not is_huggingface_id("./models/wan")
        assert not is_huggingface_id("../models/wan")
        assert not is_huggingface_id("~/models/wan")

    def test_rejects_windows_paths(self):
        assert not is_huggingface_id("C:\\Users\\models\\wan")
        assert not is_huggingface_id("D:\\models")
        assert not is_huggingface_id("C:/Users/models/wan")

    def test_rejects_relative_windows_paths(self):
        assert not is_huggingface_id(".\\models\\wan")
        assert not is_huggingface_id("..\\models")

    def test_rejects_backslash_paths(self):
        assert not is_huggingface_id("models\\wan\\checkpoint")

    def test_rejects_single_word(self):
        assert not is_huggingface_id("model-name")

    def test_strips_whitespace(self):
        assert is_huggingface_id("  org/model  ")

    def test_dots_and_underscores(self):
        assert is_huggingface_id("my_org/my.model_v2")
        assert is_huggingface_id("org.name/model-v1.5")


class TestFindSafetensorsInDir:
    """Test safetensors file discovery in directories."""

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            st_file = Path(tmpdir) / "model.safetensors"
            st_file.write_bytes(b"dummy content")
            result = find_safetensors_in_dir(tmpdir)
            assert result == str(st_file)

    def test_no_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "readme.md").write_text("hello")
            result = find_safetensors_in_dir(tmpdir)
            assert result is None

    def test_multiple_files_returns_largest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            small = Path(tmpdir) / "small.safetensors"
            large = Path(tmpdir) / "large.safetensors"
            small.write_bytes(b"x" * 10)
            large.write_bytes(b"x" * 1000)
            result = find_safetensors_in_dir(tmpdir)
            assert result == str(large)

    def test_sharded_with_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "shard-00001.safetensors").write_bytes(b"x" * 100)
            (Path(tmpdir) / "shard-00002.safetensors").write_bytes(b"x" * 100)
            (Path(tmpdir) / "model.safetensors.index.json").write_text("{}")
            result = find_safetensors_in_dir(tmpdir)
            assert result == tmpdir  # Returns directory for sharded

    def test_file_path_passthrough(self):
        # If given a file path (not a directory), returns it as-is
        result = find_safetensors_in_dir("/some/model.safetensors")
        assert result == "/some/model.safetensors"


class TestValidationIntegration:
    """Test that config validation correctly handles HF IDs."""

    def _make_config(self, **model_kwargs):
        from trainer.config.schema import TrainConfig
        return TrainConfig(
            model={"architecture": "wan", **model_kwargs},
            training={"method": "lora"},
            network={"network_type": "lora", "rank": 16, "alpha": 16.0},
            data={"datasets": [{"path": "/tmp/fake"}]},
        )

    def test_hf_id_skips_path_check(self):
        from trainer.config.validation import validate_config

        config = self._make_config(base_model_path="Wan-AI/Wan2.1-T2V-14B")
        result = validate_config(config)
        # Should NOT have an error about base_model_path not existing
        path_errors = [
            e for e in result.errors
            if "base_model_path" in (e.field_path or "") or "Base model path" in e.message
        ]
        assert len(path_errors) == 0

        # Should have an info message about HF model ID
        hf_info = [i for i in result.info if "HuggingFace" in i.message]
        assert len(hf_info) >= 1

    def test_hf_vae_id_skips_path_check(self):
        from trainer.config.validation import validate_config

        config = self._make_config(
            base_model_path="Wan-AI/Wan2.1-T2V-14B",
            vae_path="stabilityai/sd-vae-ft-mse",
        )
        result = validate_config(config)
        vae_errors = [
            e for e in result.errors
            if "vae_path" in (e.field_path or "") or "VAE path" in e.message
        ]
        assert len(vae_errors) == 0

    def test_local_path_still_validated(self):
        from trainer.config.validation import validate_config

        config = self._make_config(base_model_path="/nonexistent/path/to/model")
        result = validate_config(config)
        path_errors = [e for e in result.errors if "Base model path" in e.message]
        assert len(path_errors) == 1
