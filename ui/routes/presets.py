"""Preset management API routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/presets", tags=["presets"])


def _get_manager(request: Request):
    """Lazy-init PresetManager on app state."""
    if not hasattr(request.app.state, "preset_manager"):
        from ui.presets import PresetManager

        request.app.state.preset_manager = PresetManager()
    return request.app.state.preset_manager


@router.get("")
async def list_presets(request: Request):
    """List all builtin and user presets."""
    return _get_manager(request).list_presets()


@router.get("/{category}/{name}")
async def load_preset(category: str, name: str, request: Request):
    """Load a preset merged with defaults."""
    try:
        return _get_manager(request).load_preset(category, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/user")
async def save_preset(body: dict, request: Request):
    """Save a user preset."""
    name = body.get("name", "").strip()
    config = body.get("config", {})
    if not name:
        raise HTTPException(status_code=400, detail="Preset name is required")
    _get_manager(request).save_user_preset(name, config)
    return {"status": "saved", "name": name}


@router.delete("/user/{name}")
async def delete_preset(name: str, request: Request):
    """Delete a user preset."""
    _get_manager(request).delete_user_preset(name)
    return {"status": "deleted", "name": name}
