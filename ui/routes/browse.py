"""Native file/folder picker dialog endpoints."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/browse", tags=["browse"])


def _open_directory_dialog(initial_dir: str | None = None) -> str | None:
    """Open a native OS directory picker. Returns selected path or None."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    kwargs: dict = {"title": "Select Folder"}
    if initial_dir and Path(initial_dir).is_dir():
        kwargs["initialdir"] = initial_dir

    result = filedialog.askdirectory(**kwargs)
    root.destroy()
    return result or None


def _open_file_dialog(
    initial_dir: str | None = None,
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Open a native OS file picker. Returns selected path or None."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    kwargs: dict = {"title": "Select File"}
    if initial_dir and Path(initial_dir).is_dir():
        kwargs["initialdir"] = initial_dir
    if filetypes:
        kwargs["filetypes"] = filetypes + [("All files", "*.*")]

    result = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return result or None


@router.post("/directory")
async def browse_directory(body: dict):
    """Open a native folder picker dialog."""
    initial_dir = body.get("initial_dir")
    try:
        path = await asyncio.to_thread(_open_directory_dialog, initial_dir)
    except Exception as exc:
        logger.exception("Failed to open directory dialog")
        return JSONResponse({"error": str(exc)}, status_code=500)

    if path is None:
        return {"path": None, "cancelled": True}
    return {"path": path.replace("\\", "/"), "cancelled": False}


@router.post("/file")
async def browse_file(body: dict):
    """Open a native file picker dialog."""
    initial_dir = body.get("initial_dir")
    extensions = body.get("extensions")

    filetypes = None
    if extensions:
        ext_str = " ".join(f"*.{e.lstrip('.')}" for e in extensions)
        filetypes = [("Matching files", ext_str)]

    try:
        path = await asyncio.to_thread(_open_file_dialog, initial_dir, filetypes)
    except Exception as exc:
        logger.exception("Failed to open file dialog")
        return JSONResponse({"error": str(exc)}, status_code=500)

    if path is None:
        return {"path": None, "cancelled": True}
    return {"path": path.replace("\\", "/"), "cancelled": False}
