# Phase 4: Subprocess + UI Skeleton — Design

**Date:** 2026-03-01
**Status:** Approved
**Gate:** UI launches training subprocess, receives events via WebSocket, displays progress, survives simulated OOM.

---

## Data Flow

```
Browser (Svelte)              FastAPI Server                    Training Subprocess
-----------------             --------------                    -------------------
                              SubprocessTrainingRunner
                              |  parent_conn (Pipe)  ------->   _training_worker()
                              |                                  |  PipeCallback
POST /api/training/start  --> runner.start(config_dict)          |    sends events -->
                              |                                  |
                              |  background_poll_task()           |
WS /ws/training  <----------  |  polls runner.poll_events()  <---+
  (receives JSON events)      |  broadcasts to all WS clients
                              |
POST /api/training/stop   --> runner.send_stop()  ----------->   check_for_commands()
POST /api/training/pause  --> runner.send_pause()
POST /api/config/validate --> validate_config()
GET  /api/models          --> list_models()
```

- Subprocess uses `mp.get_context("spawn")` — clean interpreter, no CUDA fork issues
- Events are pickled dataclasses over `multiprocessing.Pipe`
- Commands flow in reverse via REST -> pipe
- Single asyncio background task polls every 100ms, broadcasts to connected WS clients

---

## Backend Architecture

### File Layout

```
ui/
├── __init__.py
├── server.py          # FastAPI app, lifespan, static mount
├── runner.py          # SubprocessTrainingRunner + _training_worker
├── binding.py         # ConfigBinder
└── routes/
    ├── __init__.py
    ├── training.py    # POST start/stop/pause/resume/save, WS /ws/training
    ├── config.py      # POST validate, GET/PUT config values
    └── models.py      # GET /api/models
```

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/training/start | Body: {config, mode}. Calls runner.start() |
| POST | /api/training/stop | Calls runner.send_stop() |
| POST | /api/training/pause | Calls runner.send_pause() |
| POST | /api/training/resume | Calls runner.send_resume() |
| POST | /api/training/save | Calls runner.send_save() |
| GET | /api/training/status | Returns {alive, exit_message} |
| WS | /ws/training | Event stream — JSON-serialized training events |
| POST | /api/config/validate | Body: config dict. Returns validation result |
| GET | /api/models | Returns {models: ["wan", ...]} |
| GET | /api/config/defaults/{arch} | Returns default config for architecture |

### WebSocket Broadcasting

- FastAPI lifespan starts a background asyncio task
- Task polls `runner.poll_events()` every 100ms
- Events serialized to JSON (dataclass -> dict) and sent to all connected WS clients
- On subprocess death: detect via `not runner.is_alive()`, broadcast ErrorEvent with crash message
- Connected clients tracked in a `set[WebSocket]`

---

## Frontend Architecture

### Stack
- SvelteKit with Vite
- Built output -> `ui/static/`, served by FastAPI
- Dev mode: Vite proxy forwards `/api` and `/ws` to FastAPI backend

### Phase 4 Scope
- **Sidebar nav** — 4 tabs (Model, Data, Training, Output), only Training functional
- **Training page** — start/stop/pause buttons, live metrics, progress bar, log viewer
- **Config section** — load YAML, edit key fields (batch_size, lr, method, arch), validate
- **Connection indicator** — WebSocket status (connected/reconnecting/disconnected)
- **Auto-reconnect** — exponential backoff on WS disconnect

### NOT in Phase 4
- Loss chart, GPU monitor, sample gallery, preset system, full per-tab config editing (Phase 6)

---

## Error Handling

- **OOM/crash:** `get_crash_message()` maps exit codes to actionable messages. Background poller detects dead process and broadcasts ErrorEvent.
- **Broken pipe:** All pipe ops wrapped in try/except — silent degradation.
- **WS disconnect:** Client removed from broadcast set. Svelte auto-reconnects.
- **Start while running:** Returns 409 Conflict.

---

## Testing

| File | Scope |
|------|-------|
| test_subprocess.py | Runner lifecycle, worker with mock session, crash messages, OOM sim |
| test_pipe_callback.py | Event serialization over real Pipe, command polling |
| test_config_binder.py | Flatten/unflatten, change callbacks, update_many |
| test_api_routes.py | FastAPI TestClient — all endpoints |
| test_imports.py | Updated with canary imports for new modules |

### Gate Verification
1. `python -m pytest tests/ -v` — all tests pass (149 existing + ~25 new)
2. `python run_ui.py` — server starts, browser opens
3. Load YAML, click Start, see live progress via WebSocket
4. Click Stop, training stops gracefully
5. Kill subprocess (signal 9), UI shows OOM crash message
