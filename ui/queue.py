"""Job queue manager with JSON file persistence."""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QueueManager:
    """Manages a queue of training jobs persisted as JSON files.

    Each job is a JSON file in the jobs directory. The queue order
    is stored in a separate _order.json file.
    """

    def __init__(self, jobs_dir: str = "jobs") -> None:
        self._dir = Path(jobs_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._order_file = self._dir / "_order.json"

    def add_job(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Add a new job to the queue. Returns the job dict."""
        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "id": job_id,
            "name": name,
            "status": "queued",
            "config": config,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "result": None,
        }
        self._save_job(job)
        order = self._load_order()
        order.append(job_id)
        self._save_order(order)
        logger.info("Added job %s: %s", job_id, name)
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs in queue order, plus completed jobs not in queue."""
        order = self._load_order()
        jobs: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Queued/running jobs in order
        for job_id in order:
            job = self._load_job(job_id)
            if job:
                jobs.append(job)
                seen.add(job_id)

        # Completed/failed jobs not in order
        for f in sorted(self._dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            job_id = f.stem
            if job_id not in seen:
                job = self._load_job(job_id)
                if job:
                    jobs.append(job)

        return jobs

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a single job by ID."""
        return self._load_job(job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove a job from the queue and delete its file."""
        path = self._dir / f"{job_id}.json"
        if path.exists():
            path.unlink()
        order = self._load_order()
        self._save_order([j for j in order if j != job_id])
        logger.info("Removed job %s", job_id)

    def reorder_job(self, job_id: str, new_index: int) -> None:
        """Move a job to a new position in the queue."""
        order = self._load_order()
        if job_id in order:
            order.remove(job_id)
        order.insert(max(0, min(new_index, len(order))), job_id)
        self._save_order(order)

    def clone_job(self, job_id: str) -> dict[str, Any]:
        """Clone a job's config into a new queued job."""
        original = self._load_job(job_id)
        if not original:
            raise FileNotFoundError(f"Job not found: {job_id}")
        return self.add_job(f"{original['name']} (copy)", original["config"])

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update specific fields on a job."""
        job = self._load_job(job_id)
        if job:
            job.update(fields)
            self._save_job(job)

    def get_next_queued(self) -> dict[str, Any] | None:
        """Get the next queued job (first in order with status='queued')."""
        for job_id in self._load_order():
            job = self._load_job(job_id)
            if job and job["status"] == "queued":
                return job
        return None

    def _save_job(self, job: dict[str, Any]) -> None:
        path = self._dir / f"{job['id']}.json"
        path.write_text(json.dumps(job, indent=2))

    def _load_job(self, job_id: str) -> dict[str, Any] | None:
        path = self._dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load job %s", job_id)
            return None

    def _load_order(self) -> list[str]:
        if not self._order_file.exists():
            return []
        try:
            return json.loads(self._order_file.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def _save_order(self, order: list[str]) -> None:
        self._order_file.write_text(json.dumps(order))
