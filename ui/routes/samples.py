"""Sample gallery API — lists and serves generated sample files."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/samples", tags=["samples"])

SAMPLE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".mp4", ".gif", ".webp"}


@router.get("")
async def list_samples(output_dir: str = Query(default="./output")):
    """List all sample files in the output directory.

    Returns a list of sample metadata sorted most-recent-first.
    Looks for files with 'sample' in the name and image/video extensions.
    """
    base = Path(output_dir)
    if not base.exists():
        return []

    samples: list[dict] = []
    for f in base.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SAMPLE_EXTENSIONS:
            continue
        # Only include files that look like samples
        if "sample" not in f.name.lower():
            continue

        stat = f.stat()
        samples.append({
            "filename": str(f.relative_to(base)),
            "absolute_path": str(f),
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })

    # Most recent first
    samples.sort(key=lambda s: s["modified"], reverse=True)
    return samples


@router.get("/file")
async def serve_sample(path: str = Query(..., description="Absolute path to the sample file")):
    """Serve a sample file by its absolute path.

    The path must point to an existing file with a valid sample extension.
    """
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")
    if not p.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if p.suffix.lower() not in SAMPLE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {p.suffix}")

    return FileResponse(
        str(p),
        media_type=_get_media_type(p.suffix.lower()),
    )


def _get_media_type(suffix: str) -> str:
    """Map file extension to MIME type."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
    }.get(suffix, "application/octet-stream")
