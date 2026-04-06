"""Batch verification endpoints — pick N blocks, run detection, serve results."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gui.services import block_registry, detection_cache
from gui.services.detection_runner import detect_block, generate_overlay, generate_thumbnail

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared state for the current verify run
_verify_lock = threading.Lock()
_verify_results: list[dict] = []
_verify_progress: dict = {"status": "idle", "done": 0, "total": 0, "current": ""}
_verify_running = False


def _run_verify_batch(blocks: list[dict]):
    """Run detection on a batch of blocks (called in background thread)."""
    global _verify_results, _verify_progress, _verify_running

    total = len(blocks)

    for i, block in enumerate(blocks):
        name = block["name"]
        with _verify_lock:
            _verify_progress = {"status": "running", "done": i, "total": total, "current": name}

        # Use cached result if available, otherwise run detection
        cached = detection_cache.load_cached_result(name)
        if cached:
            with _verify_lock:
                _verify_results.append({
                    "name": name,
                    "row_count": cached.get("row_count", 0),
                    "confidence": cached.get("overall_confidence", 0),
                    "spacing_m": cached.get("mean_spacing_m", 0),
                    "overlay_url": f"/api/detection/{name}/overlay",
                    "cached": True,
                })
        else:
            result_tuple = detect_block(block)
            if result_tuple:
                image_bgr, mask, result, mpp = result_tuple
                overlay = generate_overlay(image_bgr, mask, result, mpp, block_name=name)
                thumbnail = generate_thumbnail(overlay)
                detection_cache.save_result(name, result, image_bgr, overlay, thumbnail)

                # Update block stage
                block_registry.update_block(name, {"stage": "detected"})

                with _verify_lock:
                    _verify_results.append({
                        "name": name,
                        "row_count": result.row_count,
                        "confidence": round(result.overall_confidence, 3),
                        "spacing_m": round(result.mean_spacing_m, 3),
                        "overlay_url": f"/api/detection/{name}/overlay",
                        "cached": False,
                    })
            else:
                with _verify_lock:
                    _verify_results.append({
                        "name": name,
                        "row_count": 0,
                        "confidence": 0,
                        "spacing_m": 0,
                        "overlay_url": "",
                        "error": "Detection failed",
                    })

    with _verify_lock:
        _verify_progress = {"status": "complete", "done": total, "total": total, "current": ""}
        _verify_running = False


@router.post("/run")
async def run_verify(n: int = 10):
    """Start batch verification on N random blocks."""
    global _verify_running

    blocks = block_registry.list_blocks()
    if n >= len(blocks):
        selected = blocks[:]
    else:
        selected = random.sample(blocks, n)

    with _verify_lock:
        if _verify_running:
            return {"status": "already_running"}
        _verify_running = True
        _verify_results = []
        _verify_progress = {"status": "running", "done": 0, "total": len(selected), "current": ""}
        thread = threading.Thread(target=_run_verify_batch, args=(selected,), daemon=True)
        thread.start()

    return {"status": "started", "total": len(selected)}


@router.get("/progress")
async def verify_progress():
    """SSE stream of batch verification progress."""
    async def event_stream():
        while True:
            with _verify_lock:
                data = _verify_progress.copy()
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("status") in ("complete", "idle"):
                return
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/results")
def verify_results():
    """Return the latest verification results."""
    with _verify_lock:
        return [result.copy() for result in _verify_results]
