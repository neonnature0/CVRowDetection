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

from gui.services import block_registry, detection_cache
from gui.services.detection_runner import (
    detect_block, generate_overlay, generate_lines_only_overlay, generate_thumbnail,
)
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
    "endpoint_trim_likelihood_ratio": {
        "type": "slider", "min": 0.1, "max": 0.7, "step": 0.05,
        "default": 0.3,
        "label": "Endpoint Trim Ratio",
        "description": "Threshold for trimming weak row tails. Higher = more aggressive trimming, rows end closer to vine canopy. Lower = rows extend further past vines.",
        "stage": 6,
    },
    "endpoint_trim_min_run": {
        "type": "slider", "min": 1, "max": 6, "step": 1,
        "default": 3,
        "label": "Endpoint Trim Run",
        "description": "Consecutive weak strips needed before trimming kicks in. Lower = trim faster at row ends. Higher = more tolerant of noisy tails.",
        "stage": 6,
    },
    "compute_ensemble_confidence": {
        "type": "checkbox",
        "default": False,
        "label": "Ensemble Confidence",
        "description": "Run both ML and Gabor, colour rows by agreement. Doubles detection time but shows which rows both methods agree on.",
        "stage": 0,
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

    # Build config with custom params (separate ensemble flag from PipelineConfig params)
    config_kwargs = {}
    run_ensemble = False
    for key, value in req.params.items():
        if key == "compute_ensemble_confidence":
            run_ensemble = bool(value)
        elif key in TUNABLE_PARAMS:
            config_kwargs[key] = value
    config = PipelineConfig(**config_kwargs)

    result_tuple = await asyncio.to_thread(detect_block, block, config)
    if result_tuple is None:
        raise HTTPException(500, "Detection failed with tuned params")

    image_bgr, mask, result, mpp = result_tuple

    # Run ensemble confidence if requested
    if run_ensemble:
        from gui.services.detection_runner import compute_ensemble_confidence
        await asyncio.to_thread(
            compute_ensemble_confidence, block, result, config, image_bgr, mask, mpp,
        )

    overlay = generate_overlay(image_bgr, mask, result, mpp, block_name=name)

    # Save tuned overlay and lines-only diff separately (don't overwrite default)
    import cv2
    tuned_overlay_path = detection_cache.get_tuned_path(name, "overlay")
    cv2.imwrite(str(tuned_overlay_path), overlay)

    # Lines-only transparent PNG for onion-skin diff
    lines_only = generate_lines_only_overlay(result, image_bgr.shape)
    tuned_lines_path = detection_cache.get_tuned_path(name, "lines")
    cv2.imwrite(str(tuned_lines_path), lines_only)

    # Save tuned params for reference
    tuned_config_path = detection_cache.get_tuned_path(name, "config")
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
    tuned_overlay = detection_cache.get_tuned_path(name, "overlay")

    if not tuned_overlay.exists():
        raise HTTPException(404, "No tuned result to apply. Run /tune first.")

    import shutil
    default_overlay = detection_cache.get_image_path(name, "overlay")
    if default_overlay is None:
        default_overlay = detection_cache._block_dir(name) / "overlay.png"
    shutil.copy2(str(tuned_overlay), str(default_overlay))

    # Regenerate thumbnail from new overlay
    import cv2
    overlay_img = cv2.imread(str(default_overlay))
    if overlay_img is not None:
        thumb = generate_thumbnail(overlay_img)
        thumb_path = detection_cache._block_dir(name) / "thumbnail.png"
        cv2.imwrite(str(thumb_path), thumb)

    return {"status": "applied"}


@router.get("/{name}/tuned-overlay")
def get_tuned_overlay(name: str):
    """Serve the tuned overlay image (for side-by-side comparison)."""
    path = detection_cache.get_tuned_path(name, "overlay")
    if not path.exists():
        raise HTTPException(404, "No tuned overlay. Run /tune first.")
    return FileResponse(path, media_type="image/png")


@router.get("/{name}/tuned-lines")
def get_tuned_lines(name: str):
    """Serve the tuned lines-only transparent PNG (for onion-skin diff)."""
    path = detection_cache.get_tuned_path(name, "lines")
    if not path.exists():
        raise HTTPException(404, "No tuned lines image. Run /tune first.")
    return FileResponse(path, media_type="image/png")


@router.get("/{name}/tuned-config")
def get_tuned_config(name: str):
    """Get the saved tuned params for a block."""
    path = detection_cache.get_tuned_path(name, "config")
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
