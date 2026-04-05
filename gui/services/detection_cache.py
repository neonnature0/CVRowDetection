"""Detection result cache with explicit invalidation rules.

Cache artifacts per block:
- output/detections/{name}.json  — pipeline result metrics + row data
- output/detections/{name}_image.png — stitched aerial image
- output/detections/{name}_overlay.png — image with detected rows drawn
- output/thumbnails/{name}.png — downscaled thumbnail

Invalidation rules (all logic lives here):
- Block boundary edited → invalidate everything
- Detection re-run → regenerate overlay + thumbnail (image re-fetched)
- Annotation saved → nothing (annotations are independent)
- ML model retrained → caller uses invalidate_all() to clear all caches
- Block deleted → delete all associated files
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from gui.config import DETECTIONS_DIR, THUMBNAILS_DIR, IMAGES_DIR, ANNOTATIONS_DIR
from vinerow.types import BlockRowDetectionResult

logger = logging.getLogger(__name__)


def _ensure_dirs():
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


def has_cached_result(name: str) -> bool:
    return (DETECTIONS_DIR / f"{name}.json").exists()


def load_cached_result(name: str) -> dict | None:
    path = DETECTIONS_DIR / f"{name}.json"
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
    _ensure_dirs()

    # Result JSON
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
    with open(DETECTIONS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Images
    cv2.imwrite(str(DETECTIONS_DIR / f"{name}_image.png"), image_bgr)
    cv2.imwrite(str(DETECTIONS_DIR / f"{name}_overlay.png"), overlay_bgr)
    cv2.imwrite(str(THUMBNAILS_DIR / f"{name}.png"), thumbnail_bgr)

    logger.info("Cached detection for %s (%d rows)", name, result.row_count)


def get_image_path(name: str, kind: str = "image") -> Path | None:
    """Get path to a cached image. kind: 'image', 'overlay', or 'thumbnail'."""
    if kind == "thumbnail":
        p = THUMBNAILS_DIR / f"{name}.png"
    elif kind == "overlay":
        p = DETECTIONS_DIR / f"{name}_overlay.png"
    else:
        p = DETECTIONS_DIR / f"{name}_image.png"
    return p if p.exists() else None


# ── Invalidation ──

def invalidate_block(name: str):
    """Remove all cached artifacts for a block (boundary changed or block deleted)."""
    for pattern_dir, patterns in [
        (DETECTIONS_DIR, [f"{name}.json", f"{name}_image.png", f"{name}_overlay.png"]),
        (THUMBNAILS_DIR, [f"{name}.png"]),
    ]:
        for p in patterns:
            path = pattern_dir / p
            if path.exists():
                path.unlink()
                logger.info("Invalidated: %s", path)

    # Also remove annotation and images if they exist
    for d, p in [(ANNOTATIONS_DIR, f"{name}.json"), (IMAGES_DIR, f"{name}.png"),
                 (IMAGES_DIR, f"{name}_mask.png")]:
        path = d / p
        if path.exists():
            path.unlink()
            logger.info("Invalidated: %s", path)


def invalidate_detection(name: str):
    """Remove detection cache only (for re-run). Keeps annotations."""
    for p in [f"{name}.json", f"{name}_image.png", f"{name}_overlay.png"]:
        path = DETECTIONS_DIR / p
        if path.exists():
            path.unlink()
    thumb = THUMBNAILS_DIR / f"{name}.png"
    if thumb.exists():
        thumb.unlink()
    logger.info("Invalidated detection cache for %s", name)


def invalidate_all():
    """Remove all detection caches (e.g., after model retrain)."""
    count = 0
    for d in [DETECTIONS_DIR, THUMBNAILS_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
                    count += 1
    logger.info("Invalidated all detection caches (%d files)", count)
