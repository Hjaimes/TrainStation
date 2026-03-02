"""Pre-flight validation -- checks config, paths, cache files before training."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from pydantic import ValidationError

router = APIRouter(prefix="/api/preflight", tags=["preflight"])
logger = logging.getLogger(__name__)


@router.post("/check")
async def preflight_check(body: dict):
    """Run pre-flight checks on config before starting training.

    Returns a list of checks with status (ok/warning/error) and whether
    training can start (can_start=True only when no errors).
    """
    from trainer.config.schema import TrainConfig
    from trainer.config.validation import validate_config

    checks: list[dict[str, str]] = []

    # 1. Config validation (Pydantic + cross-validation)
    try:
        config = TrainConfig(**body.get("config", body))
        result = validate_config(config)
        if result.errors:
            checks.append({
                "name": "Config validation",
                "status": "error",
                "message": "; ".join(i.message for i in result.errors),
            })
        elif result.warnings:
            checks.append({
                "name": "Config validation",
                "status": "warning",
                "message": "; ".join(i.message for i in result.warnings),
            })
        else:
            checks.append({
                "name": "Config validation",
                "status": "ok",
                "message": "Valid",
            })
    except (ValidationError, ValueError) as exc:
        checks.append({
            "name": "Config validation",
            "status": "error",
            "message": str(exc),
        })
        return {"checks": checks, "can_start": False}
    except Exception as exc:
        logger.exception("Unexpected error during preflight config validation")
        checks.append({
            "name": "Config validation",
            "status": "error",
            "message": f"Internal validation error: {exc}",
        })
        return {"checks": checks, "can_start": False}

    # 2. Model path
    model_path = config.model.base_model_path
    if os.path.exists(model_path):
        checks.append({"name": "Model path", "status": "ok", "message": model_path})
    else:
        checks.append({
            "name": "Model path",
            "status": "error",
            "message": f"Not found: {model_path}",
        })

    # 3. Datasets
    if config.data.datasets:
        for i, ds in enumerate(config.data.datasets):
            exists = os.path.isdir(ds.path)
            checks.append({
                "name": f"Dataset {i + 1}",
                "status": "ok" if exists else "error",
                "message": ds.path if exists else f"Not found: {ds.path}",
            })

    # 4. Output directory
    out_dir = config.saving.output_dir
    if os.path.isdir(out_dir) or os.access(os.path.dirname(out_dir) or ".", os.W_OK):
        checks.append({"name": "Output directory", "status": "ok", "message": out_dir})
    else:
        checks.append({
            "name": "Output directory",
            "status": "warning",
            "message": f"Will be created: {out_dir}",
        })

    can_start = all(c["status"] != "error" for c in checks)
    return {"checks": checks, "can_start": can_start}
