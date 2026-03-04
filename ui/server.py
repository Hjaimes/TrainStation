"""FastAPI app with WebSocket broadcasting and static SPA serving."""
from __future__ import annotations
import asyncio
import dataclasses
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from ui.runner import SubprocessTrainingRunner
from ui.routes.training import router as training_router, ws_router
from ui.routes.config import router as config_router
from ui.routes.models import router as models_router
from ui.routes.presets import router as presets_router
from ui.routes.samples import router as samples_router
from ui.routes.queue import router as queue_router
from ui.routes.preflight import router as preflight_router
from ui.routes.browse import router as browse_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def dataclass_to_dict(event) -> dict:
    """Convert a dataclass event to JSON-serializable dict with type key."""
    d = dataclasses.asdict(event)
    d["type"] = type(event).__name__
    return d


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runner = SubprocessTrainingRunner()
    app.state.ws_clients: set[WebSocket] = set()
    poll_task = asyncio.create_task(_poll_loop(app))
    logger.info("Training UI server started")
    yield
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass
    app.state.runner.stop(timeout=5.0)
    logger.info("Training UI server stopped")


async def _poll_loop(app: FastAPI) -> None:
    """Poll runner for events every 100ms, broadcast to all WS clients."""
    while True:
        try:
            runner = app.state.runner
            events = runner.poll_events()

            # Detect subprocess death
            if not runner.is_alive() and runner.exit_message:
                from trainer.events import ErrorEvent
                events.append(ErrorEvent(
                    message=runner.exit_message,
                    is_fatal=True,
                ))
                # Clear exit message to avoid re-broadcasting
                runner._exit_message = None

            if events and app.state.ws_clients:
                dead: set[WebSocket] = set()
                for event in events:
                    msg = json.dumps(dataclass_to_dict(event))
                    for ws in app.state.ws_clients:
                        try:
                            await ws.send_text(msg)
                        except Exception:
                            dead.add(ws)
                app.state.ws_clients -= dead

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in poll loop")

        await asyncio.sleep(0.1)


app = FastAPI(title="TrainStation", lifespan=lifespan)

# Routes MUST be included BEFORE static mount
app.include_router(training_router)
app.include_router(ws_router)
app.include_router(config_router)
app.include_router(models_router)
app.include_router(presets_router)
app.include_router(samples_router)
app.include_router(queue_router)
app.include_router(preflight_router)
app.include_router(browse_router)

# Static SPA fallback — serves index.html for unknown paths
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
