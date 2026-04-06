"""Annotation endpoints — queue, load/save, launch editor subprocess."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2

from fastapi import APIRouter, HTTPException

from gui.config import ANNOTATIONS_DIR, IMAGES_DIR, DETECTIONS_DIR
from gui.services import block_registry, detection_cache, task_runner
from gui.services.name_validation import resolve_under_or_400, validate_block_name_or_400

logger = logging.getLogger(__name__)
router = APIRouter()


def _annotation_path(name: str) -> Path:
    validate_block_name_or_400(name)
    return resolve_under_or_400(ANNOTATIONS_DIR, f"{name}.json")


def _ensure_annotation(name: str):
    """Create the annotation JSON + image files if they don't exist yet.

    Uses the cached detection result to build the annotation file in the
    format that annotate.py expects. Also copies the aerial image + mask
    into dataset/images/ since annotate.py reads from there.
    """
    ann_path = _annotation_path(name)
    if ann_path.exists():
        return  # already prepared

    # Need cached detection result
    cached = detection_cache.load_cached_result(name)
    if cached is None:
        raise HTTPException(400, "Run detection first before annotating")

    # Need the aerial image and mask (these were saved by detection_cache)
    src_image = resolve_under_or_400(DETECTIONS_DIR, name, "image.png")
    if not src_image.exists():
        raise HTTPException(400, "Aerial image not cached. Re-run detection.")

    # Copy image + generate mask into dataset/images/
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    dst_image = resolve_under_or_400(IMAGES_DIR, f"{name}.png")
    dst_mask = resolve_under_or_400(IMAGES_DIR, f"{name}_mask.png")

    if not dst_image.exists():
        img = cv2.imread(str(src_image))
        if img is not None:
            cv2.imwrite(str(dst_image), img)
            # Generate mask from the block boundary
            # For now, re-read from the detection image (it's masked already)
            # The mask is the non-black region
            import numpy as np
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = (gray > 0).astype(np.uint8) * 255
            cv2.imwrite(str(dst_mask), mask)

    h, w = 0, 0
    img_check = cv2.imread(str(dst_image))
    if img_check is not None:
        h, w = img_check.shape[:2]

    # Build annotation JSON from cached detection rows
    rows_data = []
    for i, row in enumerate(cached.get("rows", [])):
        rows_data.append({
            "id": i,
            "centerline_px": row.get("centerline_px", []),
            "confidence": row.get("confidence", 0.0),
            "origin": "pipeline",
            "modified": False,
        })

    # Compute meters_per_pixel from block centroid + zoom
    block = block_registry.get_block(name) or {}
    mpp = 0.0
    try:
        from vinerow.acquisition.geo_utils import meters_per_pixel, polygon_bbox
        from vinerow.acquisition.tile_fetcher import auto_select_source, default_zoom, TILE_SOURCES
        coords = block["boundary"]["coordinates"][0]
        bbox = polygon_bbox(coords)
        lat = (bbox[1] + bbox[3]) / 2.0
        lng = (bbox[0] + bbox[2]) / 2.0
        source_name = auto_select_source(lng)
        zoom = default_zoom(source_name)
        mpp = meters_per_pixel(lat, zoom, TILE_SOURCES[source_name].tile_size)
    except Exception:
        pass

    annotation = {
        "block_name": name,
        "vineyard_name": block.get("vineyard_name", ""),
        "image_file": f"images/{name}.png",
        "mask_file": f"images/{name}_mask.png",
        "image_size": [w, h],
        "meters_per_pixel": round(mpp, 6),
        "angle_deg": cached.get("dominant_angle_deg", 0.0),
        "angle_source": "fft2d",
        "angle_modified": False,
        "rows": rows_data,
        "metadata": {
            "status": "pending",
            "annotator": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": None,
            "annotation_time_seconds": None,
            "notes": "",
        },
        "ground_truth": {
            "gt_spacing_m": block.get("row_spacing_m"),
            "gt_row_count": block.get("row_count"),
            "gt_row_angle_bearing": block.get("row_angle"),
        },
    }

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
    logger.info("Created annotation file for %s (%d rows)", name, len(rows_data))


@router.get("/queue")
def get_annotation_queue():
    """List blocks that need annotation (stage is 'detected' — detection done but not annotated)."""
    blocks = block_registry.list_blocks()
    queue = [b for b in blocks if b.get("stage") == "detected"]
    return queue


@router.get("/{name}")
def get_annotation(name: str):
    """Get the annotation JSON for a block."""
    validate_block_name_or_400(name)
    path = _annotation_path(name)
    if not path.exists():
        raise HTTPException(404, "No annotation file for this block")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/{name}")
def save_annotation(name: str, data: dict):
    """Save annotation JSON for a block."""
    validate_block_name_or_400(name)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _annotation_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    block_registry.update_block(name, {"stage": "annotated"})
    logger.info("Saved annotation for %s", name)
    return {"status": "saved"}


@router.post("/{name}/prepare-blind")
def prepare_blind_annotation(name: str):
    """Create an annotation file with zero rows (for blind annotation).

    The user will draw all rows from scratch in annotate.py without
    seeing any pipeline output. This produces unbiased ground truth.
    """
    validate_block_name_or_400(name)
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    ann_path = _annotation_path(name)

    # Need the aerial image — either from detection cache or fetch fresh
    src_image = resolve_under_or_400(DETECTIONS_DIR, name, "image.png")
    if not src_image.exists():
        raise HTTPException(400, "Run detection first to cache the aerial image")

    # Copy image + generate mask
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    dst_image = resolve_under_or_400(IMAGES_DIR, f"{name}.png")
    dst_mask = resolve_under_or_400(IMAGES_DIR, f"{name}_mask.png")

    if not dst_image.exists():
        import shutil
        shutil.copy2(str(src_image), str(dst_image))
        img = cv2.imread(str(src_image))
        if img is not None:
            import numpy as np
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = (gray > 0).astype(np.uint8) * 255
            cv2.imwrite(str(dst_mask), mask)

    h, w = 0, 0
    img_check = cv2.imread(str(dst_image))
    if img_check is not None:
        h, w = img_check.shape[:2]

    # Pull angle and mpp from the cached detection result — needed for
    # evaluate_gt.py to correctly project rows onto the perpendicular axis.
    # Without angle_deg, evaluation matches y-coords against x-coords = 0% F1.
    cached = detection_cache.load_cached_result(name)
    detected_angle = cached.get("dominant_angle_deg", 0.0) if cached else 0.0

    # Compute meters_per_pixel from block centroid + zoom
    mpp = 0.0
    try:
        from vinerow.acquisition.geo_utils import meters_per_pixel, polygon_bbox
        from vinerow.acquisition.tile_fetcher import auto_select_source, default_zoom, TILE_SOURCES
        coords = block["boundary"]["coordinates"][0]
        bbox = polygon_bbox(coords)
        lat = (bbox[1] + bbox[3]) / 2.0
        lng = (bbox[0] + bbox[2]) / 2.0
        source_name = auto_select_source(lng)
        zoom = default_zoom(source_name)
        mpp = meters_per_pixel(lat, zoom, TILE_SOURCES[source_name].tile_size)
    except Exception as e:
        logger.warning("Could not compute mpp for blind annotation: %s", e)

    # Create annotation with ZERO rows (blind)
    annotation = {
        "block_name": name,
        "vineyard_name": block.get("vineyard_name", ""),
        "image_file": f"images/{name}.png",
        "mask_file": f"images/{name}_mask.png",
        "image_size": [w, h],
        "meters_per_pixel": round(mpp, 6),
        "angle_deg": round(detected_angle, 2),
        "angle_source": "fft2d",
        "angle_modified": False,
        "rows": [],  # Empty — user draws from scratch
        "metadata": {
            "status": "pending",
            "annotator": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": None,
            "annotation_time_seconds": None,
            "notes": "blind annotation — rows drawn from scratch without pipeline output",
            "blind": True,
        },
        "ground_truth": {
            "gt_spacing_m": block.get("row_spacing_m"),
            "gt_row_count": block.get("row_count"),
            "gt_row_angle_bearing": block.get("row_angle"),
        },
    }

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
    logger.info("Created blind annotation file for %s (0 rows)", name)

    return {"status": "created", "blind": True, "rows": 0}


@router.post("/{name}/launch-editor")
def launch_editor(name: str):
    """Spawn annotate.py as a subprocess for this block.

    Creates the annotation file first if it doesn't exist, using cached
    detection results. The matplotlib window opens in its own OS window.
    Poll /editor-status to check when it exits.
    """
    validate_block_name_or_400(name)
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    # Ensure annotation file + images exist
    _ensure_annotation(name)

    path = _annotation_path(name)
    mtime_before = path.stat().st_mtime if path.exists() else None

    launched = task_runner.launch_editor(name)
    if not launched:
        raise HTTPException(409, "Editor already running for this block")

    return {"status": "launched", "mtime_before": mtime_before}


@router.get("/{name}/editor-status")
def editor_status(name: str, mtime_before: float | None = None):
    """Poll whether the annotate.py subprocess has exited.

    Returns:
      - {"status": "running"} — still open
      - {"status": "saved"} — exited and annotation file was modified
      - {"status": "skipped"} — exited without saving
      - {"status": "not_started"} — no editor was launched
    """
    validate_block_name_or_400(name)
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
        block_registry.update_block(name, {"stage": "annotated"})
        return {"status": "saved"}

    return {"status": "skipped"}
