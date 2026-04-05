"""Detection endpoints — run pipeline, serve results and images."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from gui.services import block_registry, detection_cache
from gui.services.detection_runner import detect_block, generate_overlay, generate_thumbnail

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{name}/run")
async def run_detection(name: str, force: bool = False):
    """Run the pipeline on a block. Returns cached result if available unless force=True."""
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    if not force and detection_cache.has_cached_result(name):
        return detection_cache.load_cached_result(name)

    # Clear old cache if re-running
    detection_cache.invalidate_detection(name)

    # Run pipeline in a thread to keep the event loop alive
    result_tuple = await asyncio.to_thread(detect_block, block)
    if result_tuple is None:
        raise HTTPException(500, "Detection failed")

    image_bgr, mask, result, mpp = result_tuple

    # Generate overlay and thumbnail
    overlay = generate_overlay(image_bgr, mask, result, mpp, block_name=name)
    thumbnail = generate_thumbnail(overlay)

    # Cache everything
    detection_cache.save_result(name, result, image_bgr, overlay, thumbnail)

    # Update block stage
    updates = {
        "last_detection_at": datetime.now(timezone.utc).isoformat(),
    }
    if block.get("stage") == "draft":
        updates["stage"] = "detected"
    block_registry.update_block(name, updates)

    return detection_cache.load_cached_result(name)


@router.get("/{name}")
def get_detection(name: str):
    """Get cached detection result."""
    data = detection_cache.load_cached_result(name)
    if data is None:
        raise HTTPException(404, "No detection result cached. Run detection first.")
    return data


@router.get("/{name}/overlay")
def get_overlay(name: str):
    """Serve the overlay image (aerial + detected rows)."""
    path = detection_cache.get_image_path(name, "overlay")
    if path is None:
        raise HTTPException(404, "No overlay image. Run detection first.")
    return FileResponse(path, media_type="image/png")


@router.get("/{name}/image")
def get_image(name: str):
    """Serve the raw aerial image."""
    path = detection_cache.get_image_path(name, "image")
    if path is None:
        raise HTTPException(404, "No aerial image. Run detection first.")
    return FileResponse(path, media_type="image/png")


@router.get("/{name}/thumbnail")
def get_thumbnail(name: str):
    """Serve the thumbnail image."""
    path = detection_cache.get_image_path(name, "thumbnail")
    if path is None:
        raise HTTPException(404, "No thumbnail. Run detection first.")
    return FileResponse(path, media_type="image/png")
