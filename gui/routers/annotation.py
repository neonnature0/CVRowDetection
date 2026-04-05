"""Annotation endpoints — queue, load/save, launch editor subprocess."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from gui.config import ANNOTATIONS_DIR
from gui.services import block_registry, task_runner

logger = logging.getLogger(__name__)
router = APIRouter()


def _annotation_path(name: str) -> Path:
    return ANNOTATIONS_DIR / f"{name}.json"


@router.get("/queue")
def get_annotation_queue():
    """List blocks that need annotation (stage is 'detected' — detection done but not annotated)."""
    blocks = block_registry.list_blocks()
    queue = [b for b in blocks if b.get("stage") == "detected"]
    return queue


@router.get("/{name}")
def get_annotation(name: str):
    """Get the annotation JSON for a block."""
    path = _annotation_path(name)
    if not path.exists():
        raise HTTPException(404, "No annotation file for this block")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/{name}")
def save_annotation(name: str, data: dict):
    """Save annotation JSON for a block."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _annotation_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Update stage
    block_registry.update_block(name, {"stage": "annotated"})
    logger.info("Saved annotation for %s", name)
    return {"status": "saved"}


@router.post("/{name}/launch-editor")
def launch_editor(name: str):
    """Spawn annotate.py as a subprocess for this block.

    The matplotlib window opens in its own OS window.
    Poll /editor-status to check when it exits.
    """
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    # Record annotation mtime before launching
    path = _annotation_path(name)
    mtime_before = path.stat().st_mtime if path.exists() else None

    launched = task_runner.launch_editor(name)
    if not launched:
        raise HTTPException(409, "Editor already running for this block")

    return {"status": "launched", "mtime_before": mtime_before}


@router.get("/{name}/editor-status")
def editor_status(name: str, mtime_before: float | None = None):
    """Poll whether the annotate.py subprocess has exited.

    Pass mtime_before (from launch-editor response) to detect if the file was saved.

    Returns:
      - {"status": "running"} — still open
      - {"status": "saved"} — exited and annotation file was modified
      - {"status": "skipped"} — exited without saving (or file unchanged)
      - {"status": "not_started"} — no editor was launched
    """
    status = task_runner.check_editor_status(name)

    if status == "running":
        return {"status": "running"}

    if status == "not_started":
        return {"status": "not_started"}

    # Process exited — check if annotation was saved
    path = _annotation_path(name)
    if not path.exists():
        return {"status": "skipped"}

    mtime_after = path.stat().st_mtime
    if mtime_before is not None and mtime_after > mtime_before:
        # File was modified — mark as annotated
        block_registry.update_block(name, {"stage": "annotated"})
        return {"status": "saved"}

    return {"status": "skipped"}
