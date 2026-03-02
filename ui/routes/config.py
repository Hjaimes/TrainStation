"""Config validation and defaults endpoints."""
from __future__ import annotations
from fastapi import APIRouter
from pydantic import ValidationError

router = APIRouter(prefix="/api/config", tags=["config"])


@router.post("/validate")
async def validate_config_endpoint(body: dict):
    from trainer.config.schema import TrainConfig
    from trainer.config.validation import validate_config

    errors: list[str] = []
    warnings: list[str] = []

    # Stage 1: Pydantic structural validation
    try:
        config = TrainConfig(**body.get("config", body))
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Stage 2: Pre-flight validation
    result = validate_config(config)
    errors.extend(issue.message for issue in result.errors)
    warnings.extend(issue.message for issue in result.warnings)

    return {"valid": not errors, "errors": errors, "warnings": warnings}


@router.get("/defaults/{arch}")
async def get_defaults(arch: str):
    from trainer.config.schema import (
        TrainConfig, ModelConfig, NetworkConfig, DataConfig, DatasetEntry,
    )

    config = TrainConfig(
        model=ModelConfig(architecture=arch, base_model_path="<select model>"),
        network=NetworkConfig(),
        data=DataConfig(datasets=[DatasetEntry(path="<select dataset>")]),
    )
    return config.model_dump()
