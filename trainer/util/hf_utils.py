"""HuggingFace Hub integration. Detects HF model IDs and resolves them to local paths."""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# HF repo ID: org/model or org/model (with dots, hyphens, underscores)
# Must NOT look like a filesystem path (no drive letters, no leading / or ./ or ../)
_HF_REPO_PATTERN = re.compile(
    r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*$"
)


def is_huggingface_id(path: str) -> bool:
    """Check if a string looks like a HuggingFace model ID rather than a local path.

    HF IDs look like: 'Wan-AI/Wan2.1-T2V-14B', 'stabilityai/stable-diffusion-xl-base-1.0'
    Local paths look like: 'C:\\models\\wan', '/home/user/models', './models/wan'
    """
    if not path or not path.strip():
        return False

    path = path.strip()

    # Obvious local path indicators
    if path.startswith(("/", "\\", "./", ".\\", "../", "..\\", "~")):
        return False
    # Windows drive letter (C:, D:, etc.)
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        return False
    # Contains backslashes - filesystem path
    if "\\" in path:
        return False

    return bool(_HF_REPO_PATTERN.match(path))


def resolve_hf_model_path(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    cache_dir: str | None = None,
) -> str:
    """Download/cache a HuggingFace model and return the local path.

    Uses huggingface_hub.snapshot_download() which caches models in
    ~/.cache/huggingface/hub/ by default. Subsequent calls for the same
    model return the cached path instantly.

    Args:
        repo_id: HuggingFace repository ID (e.g. 'Wan-AI/Wan2.1-T2V-14B')
        revision: Git revision (branch, tag, or commit hash). None = main.
        allow_patterns: Only download files matching these patterns.
            e.g. ['*.safetensors', '*.json'] to skip large bin files.
        cache_dir: Override default cache directory.

    Returns:
        Local filesystem path to the downloaded/cached model directory.

    Raises:
        ImportError: If huggingface_hub is not installed.
        Exception: If download fails (network, auth, repo not found, etc.)
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HuggingFace model IDs. "
            "Install it with: pip install huggingface-hub"
        )

    logger.info(f"Resolving HuggingFace model: {repo_id}")

    local_path = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        local_files_only=False,
    )

    logger.info(f"HuggingFace model cached at: {local_path}")
    return local_path


def resolve_path_if_hf(path: str, **kwargs) -> str:
    """If path is a HuggingFace ID, resolve it to a local path. Otherwise return as-is.

    This is the main entry point used by the training pipeline.
    """
    if is_huggingface_id(path):
        return resolve_hf_model_path(path, **kwargs)
    return path


def find_safetensors_in_dir(directory: str) -> str | None:
    """Find the primary safetensors file in a HuggingFace model directory.

    Many HF repos have a single large .safetensors file or multiple sharded ones.
    Returns the path to the single file, or the directory itself if sharded.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return directory  # Already a file path

    safetensors_files = list(dir_path.glob("*.safetensors"))
    if not safetensors_files:
        return None

    # Single file - return it directly
    if len(safetensors_files) == 1:
        return str(safetensors_files[0])

    # Check for model index (sharded model)
    index_file = dir_path / "model.safetensors.index.json"
    if index_file.exists():
        return str(dir_path)  # Return directory for sharded models

    # Multiple files without index - return the largest one
    largest = max(safetensors_files, key=lambda f: f.stat().st_size)
    return str(largest)
