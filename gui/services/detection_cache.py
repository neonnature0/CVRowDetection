"""Detection result cache with per-block subfolders.

Cache structure:
  output/detections/{block_name}/
    result.json       — pipeline metrics + row data
    image.png         — stitched aerial image
    overlay.png       — image with detected rows drawn
    thumbnail.png     — downscaled thumbnail
    tuned_overlay.png — overlay with tuned parameters
    tuned_config.json — per-block tuned parameter values
    tuned_lines.png   — transparent lines-only diff image

Invalidation rules (all logic lives here):
- Block boundary edited → invalidate everything (delete subfolder)
- Detection re-run → regenerate overlay + thumbnail
- Annotation saved → nothing (annotations are independent)
- ML model retrained → caller uses invalidate_all()
- Block deleted → delete subfolder + annotation + images
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from gui.config import DETECTIONS_DIR, IMAGES_DIR, ANNOTATIONS_DIR
from gui.services.name_validation import resolve_under_or_400, validate_block_name_or_400
from vinerow.types import BlockRowDetectionResult

logger = logging.getLogger(__name__)


def _block_dir(name: str) -> Path:
    validate_block_name_or_400(name)
    return resolve_under_or_400(DETECTIONS_DIR, name)


def has_cached_result(name: str) -> bool:
    return (_block_dir(name) / "result.json").exists()


def load_cached_result(name: str) -> dict | None:
    path = _block_dir(name) / "result.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(
    name: str,
    result: BlockRowDetectionResult,
    image_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    thumbnail_bgr: np.ndarray,
):
    """Save all detection artifacts for a block."""
    d = _block_dir(name)
    d.mkdir(parents=True, exist_ok=True)

    result_data = {
        "block_name": name,
        "row_count": result.row_count,
        "mean_spacing_m": round(result.mean_spacing_m, 3),
        "spacing_std_m": round(result.spacing_std_m, 3),
        "dominant_angle_deg": round(result.dominant_angle_deg, 2),
        "dominant_angle_bearing": round(result.dominant_angle_bearing, 2),
        "overall_confidence": round(result.overall_confidence, 3),
        "total_time_s": round(result.timings.total, 2) if result.timings else None,
        "rows": [
            {
                "row_index": r.row_index,
                "centerline_px": r.centerline_px,
                "confidence": round(r.confidence, 3),
                "length_m": round(r.length_m, 2),
            }
            for r in result.rows
        ],
    }
    with open(d / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    cv2.imwrite(str(d / "image.png"), image_bgr)
    cv2.imwrite(str(d / "overlay.png"), overlay_bgr)
    cv2.imwrite(str(d / "thumbnail.png"), thumbnail_bgr)

    logger.info("Cached detection for %s (%d rows)", name, result.row_count)


def get_image_path(name: str, kind: str = "image") -> Path | None:
    """Get path to a cached image. kind: 'image', 'overlay', 'thumbnail'."""
    d = _block_dir(name)
    if kind == "thumbnail":
        p = d / "thumbnail.png"
    elif kind == "overlay":
        p = d / "overlay.png"
    else:
        p = d / "image.png"
    return p if p.exists() else None


def get_tuned_path(name: str, kind: str) -> Path:
    """Get path for tuned artifacts. kind: 'overlay', 'config', 'lines'."""
    d = _block_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    if kind == "config":
        return d / "tuned_config.json"
    elif kind == "lines":
        return d / "tuned_lines.png"
    return d / "tuned_overlay.png"


# ── Invalidation ──

def invalidate_block(name: str):
    """Remove all cached artifacts for a block (boundary changed or block deleted)."""
    validate_block_name_or_400(name)
    d = _block_dir(name)
    if d.exists():
        shutil.rmtree(d)
        logger.info("Invalidated block dir: %s", d)

    for dd, p in [(ANNOTATIONS_DIR, f"{name}.json"), (IMAGES_DIR, f"{name}.png"),
                  (IMAGES_DIR, f"{name}_mask.png")]:
        path = resolve_under_or_400(dd, p)
        if path.exists():
            path.unlink()
            logger.info("Invalidated: %s", path)


def invalidate_detection(name: str):
    """Remove detection cache only (for re-run). Keeps annotations and tuned config."""
    validate_block_name_or_400(name)
    d = _block_dir(name)
    for fname in ["result.json", "image.png", "overlay.png", "thumbnail.png",
                  "tuned_overlay.png", "tuned_lines.png"]:
        p = d / fname
        if p.exists():
            p.unlink()
    logger.info("Invalidated detection cache for %s", name)


def invalidate_all():
    """Remove all detection caches (e.g., after model retrain)."""
    count = 0
    if DETECTIONS_DIR.exists():
        for block_dir in DETECTIONS_DIR.iterdir():
            if block_dir.is_dir():
                shutil.rmtree(block_dir)
                count += 1
    logger.info("Invalidated all detection caches (%d blocks)", count)
