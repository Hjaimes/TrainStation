"""Job queue API routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/queue", tags=["queue"])


def _get_manager(request: Request):
    """Lazy-init QueueManager on app state."""
    if not hasattr(request.app.state, "queue_manager"):
        from ui.queue import QueueManager

        request.app.state.queue_manager = QueueManager()
    return request.app.state.queue_manager


@router.get("")
async def list_jobs(request: Request):
    """List all jobs."""
    return _get_manager(request).list_jobs()


@router.post("/add")
async def add_job(body: dict, request: Request):
    """Add a new job to the queue."""
    name = body.get("name", "Untitled")
    config = body.get("config", {})
    return _get_manager(request).add_job(name, config)


@router.get("/{job_id}")
async def get_job(job_id: str, request: Request):
    """Get a single job."""
    job = _get_manager(request).get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.delete("/{job_id}")
async def remove_job(job_id: str, request: Request):
    """Remove a job."""
    _get_manager(request).remove_job(job_id)
    return {"status": "removed"}


@router.post("/{job_id}/reorder")
async def reorder_job(job_id: str, body: dict, request: Request):
    """Move a job to a new queue position."""
    index = body.get("index", 0)
    _get_manager(request).reorder_job(job_id, index)
    return {"status": "reordered"}


@router.post("/{job_id}/clone")
async def clone_job(job_id: str, request: Request):
    """Clone a job's config into a new queued job."""
    try:
        return _get_manager(request).clone_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
