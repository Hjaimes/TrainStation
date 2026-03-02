"""Training control endpoints and WebSocket event stream."""
from __future__ import annotations
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start")
async def start_training(body: dict, request: Request):
    runner = request.app.state.runner
    config = body.get("config", {})
    mode = body.get("mode", "train")
    try:
        runner.start(config, mode=mode)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"status": "started"}


@router.post("/stop")
async def stop_training(request: Request):
    request.app.state.runner.send_stop()
    return {"status": "stopping"}


@router.post("/pause")
async def pause_training(request: Request):
    request.app.state.runner.send_pause()
    return {"status": "paused"}


@router.post("/resume")
async def resume_training(request: Request):
    request.app.state.runner.send_resume()
    return {"status": "resumed"}


@router.post("/save")
async def save_checkpoint(request: Request):
    request.app.state.runner.send_save()
    return {"status": "saving"}


@router.get("/status")
async def training_status(request: Request):
    runner = request.app.state.runner
    return {
        "alive": runner.is_alive(),
        "exit_message": runner.exit_message,
    }


ws_router = APIRouter()


@ws_router.websocket("/ws/training")
async def training_ws(websocket: WebSocket, request: Request):
    await websocket.accept()
    request.app.state.ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        request.app.state.ws_clients.discard(websocket)
