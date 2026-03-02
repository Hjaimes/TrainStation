# Phase 6: Full UI Design

**Goal:** Build a production-quality web UI for the training app with form-based configuration, live training dashboard, sample gallery, job queue, and preset system.

**Audience:** Mixed — progressive disclosure (essential fields visible, advanced behind toggles).

**Tech stack:** SvelteKit 5 (Svelte 5 runes), FastAPI backend, Chart.js for loss plots, WebSocket for live updates.

---

## Pages

| Page | Route | Purpose |
|------|-------|---------|
| Configure | `/configure` | Form-based config editor with sections |
| Monitor | `/monitor` | Live training dashboard (loss chart, metrics, logs, sample preview) |
| Samples | `/samples` | Sample gallery with lightbox |
| Queue | `/queue` | Job queue management |
| Settings | `/settings` | App-level settings |

---

## Configure Page

Inner sidebar/tab bar for config sections: Model, Network, Data, Training, Sampling, Output.

Each section shows essential fields (~5-8) by default, with "Advanced" toggle for the rest. A "Raw Config" tab shows full YAML/JSON for power users.

Top bar: Preset dropdown (left), Validate button, "Start Training" / "Add to Queue" button (right).

### Field Organization

**Model (essential):** Architecture dropdown, base model path, model dtype, quantization.
**Model (advanced):** Gradient checkpointing, activation offloading, weight bouncing, block swap.

**Network (essential):** Training method (LoRA/LoHa/LoKr/Full Finetune), rank, alpha, save dtype.
**Network (advanced):** DoRA toggle, dropout, target modules filter, block LR multipliers, save format.

**Data (essential):** Dataset path(s), dataset config TOML path, batch size.
**Data (advanced):** Bucketing, num_workers, persistent_workers, augmentations, masked training, regularization path, tag shuffling, token dropout.

**Training (essential):** Optimizer, learning rate, LR scheduler, steps/epochs, gradient accumulation.
**Training (advanced):** Warmup steps, min LR ratio, noise offset, timestep distribution, loss function, loss weighting, P2 gamma, zero terminal SNR, stochastic rounding, fused backward, LR scaling, progressive timesteps, text encoder training + TE LR.

**Sampling (essential):** Sample every N steps, prompt list (add/remove rows with prompt + seed), width/height/frames.

**Output (essential):** Output directory, save every N steps, max saves to keep.
**Output (advanced):** Checkpoint format, save filename prefix, validation settings.

---

## Training Monitor (Dashboard Grid)

2x2 layout when training is active:

```
┌─────────────────────────────────┬──────────────────────┐
│ Loss Chart (Chart.js)           │ Metrics Cards        │
│ - Raw loss (faint line)         │ Step, Loss, Avg Loss │
│ - Smoothed loss (bold line)     │ LR, Epoch, ETA       │
│ - Zoom/pan                      │ VRAM peak + bar      │
│                                 │ Progress bar         │
├─────────────────────────────────┼──────────────────────┤
│ Log Viewer                      │ Sample Preview       │
│ - Scrolling, level-colored      │ - Latest thumbnail   │
│ - Auto-scroll w/ manual override│ - Click → Samples    │
└─────────────────────────────────┴──────────────────────┘
```

When not training: last run summary or placeholder with link to Configure.

Loss chart accumulates StepEvent data in the Svelte store (persists across navigation). Smoothed line uses configurable EMA window.

---

## Job Queue

Three job states: Queued, Running, Completed (includes failed).

- Queued: config name, architecture, method. Drag to reorder. Delete button.
- Running: progress bar, current step/total, loss. Click → Monitor.
- Completed: final loss, duration, output path. "Re-run" and "Clone" buttons.

**Backend:** `ui/queue.py`. Jobs stored as JSON files in `jobs/` directory. Each job: `{id, name, status, config, created_at, started_at, completed_at, result}`.

**Execution:** `QueueManager` wraps `SubprocessTrainingRunner`. Runs one job at a time, auto-starts next on completion/failure.

**API:**
- `POST /api/queue/add` — add config to queue
- `GET /api/queue` — list all jobs
- `DELETE /api/queue/{id}` — remove queued job
- `POST /api/queue/{id}/reorder` — move job position
- `POST /api/queue/{id}/clone` — duplicate config back to queue

---

## Preset System

```
presets/
  builtin/     # Shipped, read-only in UI (~15-20 files)
  user/        # User-created, full CRUD
```

Built-in presets: one per active architecture × 2-3 VRAM tiers. Partial YAML — deep-merged over TrainConfig defaults on load.

UI: dropdown at top of Configure page, grouped by Built-in / My Presets. "Save as Preset" button for user presets.

**API:**
- `GET /api/presets` — list all
- `GET /api/presets/{category}/{name}` — load merged config
- `POST /api/presets/user` — save
- `DELETE /api/presets/user/{name}` — delete

---

## Sample Gallery

Grid of thumbnail cards (most recent first). Each: image, truncated prompt, step, timestamp. Click opens lightbox with full-size image, full prompt, seed, step, dimensions, left/right navigation.

Filter: current run / all runs, prompt text search. Video architectures show first frame as thumbnail.

**API:**
- `GET /api/samples` — list sample files
- `GET /api/samples/{filename}` — serve image

Live updates via `SampleEvent` over WebSocket.

---

## Pre-flight Validation Modal

Triggered before training starts. Checklist:
1. Config validation (Pydantic + `validate_config()`)
2. Model path exists and accessible
3. Dataset path exists with cache files
4. Output directory writable
5. VRAM estimate (rough heuristic)

Green/yellow/red indicators. Warnings non-blocking, errors block Start.
