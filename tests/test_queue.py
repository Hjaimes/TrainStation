"""Tests for job queue manager (Task 13)."""
from __future__ import annotations


def test_queue_manager_import():
    from ui.queue import QueueManager  # noqa: F401


def test_queue_add_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    job = mgr.add_job("test-run", {"model": {"architecture": "wan"}})
    assert job["status"] == "queued"
    assert job["name"] == "test-run"
    assert "id" in job
    assert job["config"]["model"]["architecture"] == "wan"
    assert job["created_at"] is not None
    assert job["started_at"] is None


def test_queue_list_jobs(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    mgr.add_job("job-1", {"arch": "wan"})
    mgr.add_job("job-2", {"arch": "flux_2"})
    jobs = mgr.list_jobs()
    assert len(jobs) == 2
    assert jobs[0]["name"] == "job-1"
    assert jobs[1]["name"] == "job-2"


def test_queue_list_empty(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    assert mgr.list_jobs() == []


def test_queue_remove_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    job = mgr.add_job("delete-me", {})
    mgr.remove_job(job["id"])
    assert len(mgr.list_jobs()) == 0


def test_queue_get_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    job = mgr.add_job("find-me", {"key": "value"})
    found = mgr.get_job(job["id"])
    assert found is not None
    assert found["name"] == "find-me"


def test_queue_get_job_not_found(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    assert mgr.get_job("nonexistent") is None


def test_queue_reorder(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    j1 = mgr.add_job("first", {})
    j2 = mgr.add_job("second", {})
    j3 = mgr.add_job("third", {})
    # Move third to position 0
    mgr.reorder_job(j3["id"], 0)
    jobs = mgr.list_jobs()
    assert jobs[0]["id"] == j3["id"]
    assert jobs[1]["id"] == j1["id"]
    assert jobs[2]["id"] == j2["id"]


def test_queue_clone_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    j1 = mgr.add_job("original", {"model": {"architecture": "wan"}})
    j2 = mgr.clone_job(j1["id"])
    assert j2["name"] == "original (copy)"
    assert j2["config"] == j1["config"]
    assert j2["id"] != j1["id"]
    assert j2["status"] == "queued"
    assert len(mgr.list_jobs()) == 2


def test_queue_clone_not_found(tmp_path):
    from ui.queue import QueueManager
    import pytest
    mgr = QueueManager(jobs_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        mgr.clone_job("nonexistent")


def test_queue_update_job(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    job = mgr.add_job("update-me", {})
    mgr.update_job(job["id"], status="running", started_at=123.456)
    updated = mgr.get_job(job["id"])
    assert updated is not None
    assert updated["status"] == "running"
    assert updated["started_at"] == 123.456


def test_queue_get_next_queued(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    j1 = mgr.add_job("first", {})
    j2 = mgr.add_job("second", {})
    mgr.update_job(j1["id"], status="completed")
    nxt = mgr.get_next_queued()
    assert nxt is not None
    assert nxt["id"] == j2["id"]


def test_queue_get_next_queued_empty(tmp_path):
    from ui.queue import QueueManager
    mgr = QueueManager(jobs_dir=str(tmp_path))
    assert mgr.get_next_queued() is None
