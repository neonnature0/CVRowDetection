"""Detection endpoints — run pipeline, serve results and images."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gui.config import DETECTIONS_DIR
from gui.services import block_registry, detection_cache
from gui.services.detection_runner import detect_block, generate_overlay, generate_thumbnail
from vinerow.config import PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter()

# The 6 key tunable parameters exposed to the GUI
TUNABLE_PARAMS = {
    "ridge_mode": {
        "type": "select",
        "options": ["ml", "gabor", "ml_ensemble", "ensemble", "hessian", "exg_only", "luminance"],
        "default": "ml",
        "label": "Ridge Mode",
        "description": "Which method detects row ridges. ML uses trained model, gabor uses classical filter.",
        "stage": 3,
    },
    "ridge_scale_factor": {
        "type": "slider", "min": 0.05, "max": 0.50, "step": 0.01,
        "default": 0.2,
        "label": "Gabor Scale",
        "description": "Gabor filter bandwidth as fraction of row spacing. Narrower = more selective. Only affects gabor/ensemble modes.",
        "stage": 3,
    },
    "peak_min_prominence": {
        "type": "slider", "min": 0.02, "max": 0.40, "step": 0.01,
        "default": 0.10,
        "label": "Peak Sensitivity",
        "description": "Minimum peak prominence to count as a candidate row. Lower = detect weaker rows (more false positives). Higher = only strong rows.",
        "stage": 4,
    },
    "position_weight": {
        "type": "slider", "min": 0.3, "max": 2.0, "step": 0.1,
        "default": 1.0,
        "label": "Position Weight",
        "description": "How strongly the tracker favours geometric continuity. Higher = stricter straight-line following. Lower = allows more curve deviation.",
        "stage": 5,
    },
    "spline_smoothing_m": {
        "type": "slider", "min": 0.05, "max": 1.0, "step": 0.05,
        "default": 0.2,
        "label": "Spline Smoothing",
        "description": "Allowed deviation from tracked points (metres). Higher = smoother/straighter rows. Lower = follows curves more closely.",
        "stage": 6,
    },
    "min_row_confidence": {
        "type": "slider", "min": 0.0, "max": 0.50, "step": 0.01,
        "default": 0.15,
        "label": "Min Confidence",
        "description": "Rows below this confidence are discarded. Lower = keep weak rows. Higher = only keep confident detections.",
        "stage": 7,
    },
}


class TuneRequest(BaseModel):
    params: dict[str, Any]


@router.get("/tunable-params")
def get_tunable_params():
    """Return the tunable parameter definitions for the GUI."""
    return TUNABLE_PARAMS


@router.post("/{name}/run")
async def run_detection(name: str, force: bool = False):
    """Run the pipeline on a block with default config."""
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    if not force and detection_cache.has_cached_result(name):
        return detection_cache.load_cached_result(name)

    detection_cache.invalidate_detection(name)

    result_tuple = await asyncio.to_thread(detect_block, block)
    if result_tuple is None:
        raise HTTPException(500, "Detection failed")

    image_bgr, mask, result, mpp = result_tuple
    overlay = generate_overlay(image_bgr, mask, result, mpp, block_name=name)
    thumbnail = generate_thumbnail(overlay)
    detection_cache.save_result(name, result, image_bgr, overlay, thumbnail)

    updates = {"last_detection_at": datetime.now(timezone.utc).isoformat()}
    current_stage = block.get("stage") or "draft"
    if current_stage in ("draft", None, ""):
        updates["stage"] = "detected"
    block_registry.update_block(name, updates)

    return detection_cache.load_cached_result(name)


@router.post("/{name}/tune")
async def run_tuned_detection(name: str, req: TuneRequest):
    """Run detection with custom parameters. Saves result as a 'tuned' variant
    alongside the default result for before/after comparison.

    Does NOT overwrite the default detection cache.
    """
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    # Build config with custom params
    config_kwargs = {}
    for key, value in req.params.items():
        if key in TUNABLE_PARAMS:
            config_kwargs[key] = value
    config = PipelineConfig(**config_kwargs)

    result_tuple = await asyncio.to_thread(detect_block, block, config)
    if result_tuple is None:
        raise HTTPException(500, "Detection failed with tuned params")

    image_bgr, mask, result, mpp = result_tuple
    overlay = generate_overlay(image_bgr, mask, result, mpp, block_name=name)

    # Save tuned overlay separately (don't overwrite default)
    import cv2
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    tuned_overlay_path = DETECTIONS_DIR / f"{name}_tuned_overlay.png"
    cv2.imwrite(str(tuned_overlay_path), overlay)

    # Save tuned params for reference
    tuned_config_path = DETECTIONS_DIR / f"{name}_tuned_config.json"
    tuned_config_path.write_text(json.dumps(req.params, indent=2), encoding="utf-8")

    # Return metrics for comparison
    return {
        "block_name": name,
        "params": req.params,
        "row_count": result.row_count,
        "mean_spacing_m": round(result.mean_spacing_m, 3),
        "dominant_angle_deg": round(result.dominant_angle_deg, 2),
        "overall_confidence": round(result.overall_confidence, 3),
        "total_time_s": round(result.timings.total, 2) if result.timings else None,
    }


@router.post("/{name}/apply-tuned")
def apply_tuned_config(name: str):
    """Promote the tuned result to be the default (overwrite default cache)."""
    tuned_overlay = DETECTIONS_DIR / f"{name}_tuned_overlay.png"
    tuned_config = DETECTIONS_DIR / f"{name}_tuned_config.json"

    if not tuned_overlay.exists():
        raise HTTPException(404, "No tuned result to apply. Run /tune first.")

    import shutil
    default_overlay = DETECTIONS_DIR / f"{name}_overlay.png"
    shutil.copy2(str(tuned_overlay), str(default_overlay))

    # Regenerate thumbnail from new overlay
    import cv2
    from gui.config import THUMBNAILS_DIR
    overlay_img = cv2.imread(str(default_overlay))
    if overlay_img is not None:
        thumb = generate_thumbnail(overlay_img)
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(THUMBNAILS_DIR / f"{name}.png"), thumb)

    return {"status": "applied"}


@router.get("/{name}/tuned-overlay")
def get_tuned_overlay(name: str):
    """Serve the tuned overlay image (for side-by-side comparison)."""
    path = DETECTIONS_DIR / f"{name}_tuned_overlay.png"
    if not path.exists():
        raise HTTPException(404, "No tuned overlay. Run /tune first.")
    return FileResponse(path, media_type="image/png")


@router.get("/{name}/tuned-config")
def get_tuned_config(name: str):
    """Get the saved tuned params for a block."""
    path = DETECTIONS_DIR / f"{name}_tuned_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
