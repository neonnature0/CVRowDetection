"""Training endpoints — generate data, train model, SSE progress, invalidate caches."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gui.config import ANNOTATIONS_DIR, PROJECT_ROOT, RECOMMENDED_MINIMUM_TRAINING_BLOCKS
from gui.services import block_registry, detection_cache, task_runner

logger = logging.getLogger(__name__)
router = APIRouter()

# Where train.py writes its progress
CHECKPOINTS_DIR = PROJECT_ROOT / "training" / "checkpoints_fpn"
PROGRESS_FILE = CHECKPOINTS_DIR / "training_progress.json"


def _count_annotated() -> int:
    return _training_data_summary()["annotated_blocks_total"]


def _training_data_summary() -> dict:
    """Compute per-block training data freshness from annotation metadata.

    A block has "fresh" training data if training_data_generated_at exists
    and is newer than the annotation's last modification time. "Stale" means
    training data exists but the annotation was modified afterwards. "Missing"
    means no training data has ever been generated.
    """
    total = 0
    fresh = 0
    stale = 0
    missing = 0

    if not ANNOTATIONS_DIR.exists():
        return {
            "annotated_blocks_total": 0,
            "blocks_with_fresh_training_data": 0,
            "blocks_with_stale_training_data": 0,
            "blocks_without_training_data": 0,
        }

    for f in ANNOTATIONS_DIR.glob("*.json"):
        if f.name == "manifest.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("metadata", {}).get("status") != "complete":
                continue
            total += 1

            metadata = data.get("metadata", {})
            gen_at = metadata.get("training_data_generated_at")
            if not gen_at:
                missing += 1
                continue

            # Compare generation time against annotation modification time.
            # Use modified_at if set, otherwise fall back to file mtime.
            mod_at = metadata.get("modified_at")
            if mod_at:
                annotation_changed = mod_at
            else:
                annotation_changed = datetime.fromtimestamp(
                    f.stat().st_mtime, tz=timezone.utc
                ).isoformat()

            if gen_at >= annotation_changed:
                fresh += 1
            else:
                stale += 1
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "annotated_blocks_total": total,
        "blocks_with_fresh_training_data": fresh,
        "blocks_with_stale_training_data": stale,
        "blocks_without_training_data": missing,
    }


def _model_info() -> dict | None:
    model_path = PROJECT_ROOT / "models" / "best_model_fpn.pth"
    if not model_path.exists():
        # Check old location too
        model_path = CHECKPOINTS_DIR / "best_model.pth"
    if not model_path.exists():
        return None
    stat = model_path.stat()
    return {
        "path": str(model_path.relative_to(PROJECT_ROOT)),
        "size_mb": round(stat.st_size / 1e6, 1),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


@router.get("/stats")
def training_stats():
    """Return annotation count, training data state, and model info."""
    summary = _training_data_summary()
    return {
        "annotated_blocks": summary["annotated_blocks_total"],
        "model": _model_info(),
        "training_status": task_runner.check_training_status(),
        "generate_status": task_runner.check_generate_status(),
        **summary,
        "recommended_minimum_blocks": RECOMMENDED_MINIMUM_TRAINING_BLOCKS,
    }


@router.post("/generate")
async def generate_training_data():
    """Run generate_training_data.py (synchronous — waits for completion)."""
    launched = task_runner.launch_generate(["--status", "complete"])
    if not launched:
        raise HTTPException(409, "Data generation already running")

    # Wait in a thread to avoid blocking
    exit_code = await asyncio.to_thread(task_runner.wait_generate, 600)
    if exit_code != 0:
        raise HTTPException(500, f"generate_training_data.py exited with code {exit_code}")

    return {"status": "complete"}


@router.post("/start")
def start_training():
    """Launch training as a background subprocess."""
    if _count_annotated() == 0:
        raise HTTPException(400, "No completed annotations. Annotate some blocks first.")

    # Use FPN decoder by default
    launched = task_runner.launch_training(["--decoder", "fpn"])
    if not launched:
        raise HTTPException(409, "Training already running")
    return {"status": "started"}


@router.post("/stop")
def stop_training():
    """Kill the training subprocess."""
    stopped = task_runner.stop_training()
    if not stopped:
        raise HTTPException(404, "No training running")

    # Write failed status to progress file
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            data["status"] = "stopped"
            PROGRESS_FILE.write_text(json.dumps(data), encoding="utf-8")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to update progress file on stop: %s", e)

    return {"status": "stopped"}


@router.get("/progress")
async def training_progress():
    """SSE stream that tails the training progress JSON file."""
    async def event_stream():
        last_epoch = -1
        while True:
            status = task_runner.check_training_status()

            if PROGRESS_FILE.exists():
                try:
                    data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
                    epoch = data.get("epoch", -1)
                    if epoch != last_epoch or data.get("status") in ("complete", "stopped", "failed"):
                        last_epoch = epoch
                        yield f"data: {json.dumps(data)}\n\n"
                    if data.get("status") in ("complete", "stopped", "failed"):
                        return
                except (json.JSONDecodeError, OSError):
                    pass  # SSE loop — skip one tick on read error

            if status == "exited" or status == "not_started":
                # Training ended but no progress file update — send final event
                yield f"data: {json.dumps({'status': 'complete'})}\n\n"
                return

            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/invalidate-all")
def invalidate_all_detections():
    """Clear all cached detection results (after model retrain)."""
    detection_cache.invalidate_all()

    # Reset all detected/verified blocks back to draft
    for block in block_registry.list_blocks():
        stage = block.get("stage", "draft")
        if stage in ("detected", "verified"):
            block_registry.update_block(block["name"], {"stage": "draft", "last_detection_at": None})

    return {"status": "invalidated"}
