"""Model listing endpoint."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def get_models():
    from trainer.registry import list_models
    return {"models": list_models()}
