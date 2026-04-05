#!/usr/bin/env python3
"""
Prepare annotation dataset from the vinerow pipeline.

Runs each test block through the pipeline, caches the stitched image and mask,
and creates a pre-filled annotation JSON with detected row perpendicular
positions. These JSONs serve as starting points for manual correction in
annotate.py.

Usage:
    python prepare_dataset.py --all
    python prepare_dataset.py --block "Block C"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from vinerow.acquisition.geo_utils import meters_per_pixel, polygon_bbox
from vinerow.acquisition.tile_fetcher import (
    TILE_SOURCES,
    auto_select_source,
    default_zoom,
    fetch_and_stitch,
)
from vinerow.config import PipelineConfig
from vinerow.loaders.json_loader import load_test_blocks
from vinerow.pipeline import run_pipeline
from vinerow.types import BlockRowDetectionResult, FittedRow

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s >> %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("dataset")
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"





def safe_name(vineyard: str, block: str) -> str:
    """Generate a filesystem-safe name from vineyard + block."""
    return f"{vineyard.replace(' ', '_')}_{block.replace(' ', '_')}"


def mean_perp_position(
    centerline_px: list[tuple[float, float]], angle_deg: float, cx: float, cy: float,
) -> float:
    """Compute mean perpendicular position of a centerline polyline."""
    angle_rad = math.radians(angle_deg)
    perp_dx = -math.sin(angle_rad)
    perp_dy = math.cos(angle_rad)
    perps = [(x - cx) * perp_dx + (y - cy) * perp_dy for x, y in centerline_px]
    return float(np.mean(perps))


def prepare_block(block: dict, config: PipelineConfig) -> dict | None:
    """Run pipeline on one block and save image, mask, and annotation JSON.

    Returns a manifest entry dict, or None on failure.
    """
    name = block["name"]
    vineyard = block.get("vineyard_name", "Unknown")
    coords = block["boundary"]["coordinates"][0]
    sname = safe_name(vineyard, name)

    image_path = IMAGES_DIR / f"{sname}.png"
    mask_path = IMAGES_DIR / f"{sname}_mask.png"
    annotation_path = ANNOTATIONS_DIR / f"{sname}.json"

    # Tile fetch
    bbox = polygon_bbox(coords)
    cx_lng = (bbox[0] + bbox[2]) / 2.0
    cx_lat = (bbox[1] + bbox[3]) / 2.0
    source_name = auto_select_source(cx_lng)
    source = TILE_SOURCES[source_name]
    zoom = default_zoom(source_name)

    image_bgr, mask, tile_origin = fetch_and_stitch(
        source, coords, zoom, source_name, cache_dir=config.tile_cache_dir,
    )
    mpp = meters_per_pixel(cx_lat, zoom, source.tile_size)

    # Save image and mask
    cv2.imwrite(str(image_path), image_bgr)
    cv2.imwrite(str(mask_path), mask)
    h, w = image_bgr.shape[:2]
    logger.info("Saved image %dx%d: %s", w, h, image_path.name)

    # Run pipeline
    result = run_pipeline(
        image_bgr=image_bgr, mask=mask, mpp=mpp, lat=cx_lat,
        zoom=zoom, tile_size=source.tile_size, tile_origin=tile_origin,
        tile_source=source_name, config=config,
    )

    if result is None:
        logger.error("Pipeline failed for %s — saving image/mask only", name)
        rows_data = []
        angle_deg = 0.0
    else:
        angle_deg = result.dominant_angle_deg
        cx_img, cy_img = w / 2.0, h / 2.0

        # Store full polyline centerlines from pipeline, sorted by mean perp position
        row_items = []
        for row in result.rows:
            cl = list(row.centerline_px)  # list of (x, y) tuples
            perp = mean_perp_position(cl, angle_deg, cx_img, cy_img)
            row_items.append((perp, cl, row.confidence))
        row_items.sort(key=lambda x: x[0])

        rows_data = [
            {
                "id": i,
                "centerline_px": [[round(x, 1), round(y, 1)] for x, y in cl],
                "confidence": round(conf, 4),
                "origin": "pipeline",
                "modified": False,
            }
            for i, (_, cl, conf) in enumerate(row_items)
        ]

    # Ground truth from test_blocks.json
    gt_spacing = block.get("row_spacing_m")
    gt_count = block.get("row_count")
    gt_bearing = block.get("row_angle")

    # Build annotation JSON
    annotation = {
        "block_name": name,
        "vineyard_name": vineyard,
        "image_file": f"images/{sname}.png",
        "mask_file": f"images/{sname}_mask.png",
        "image_size": [w, h],
        "meters_per_pixel": round(mpp, 6),
        "source": source_name,
        "zoom": zoom,
        "angle_deg": round(angle_deg, 2),
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
            "gt_spacing_m": gt_spacing,
            "gt_row_count": gt_count,
            "gt_row_angle_bearing": gt_bearing,
        },
    }

    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)
    logger.info(
        "Saved annotation: %s (%d rows, angle=%.1f deg)",
        annotation_path.name, len(rows_data), angle_deg,
    )

    return {
        "block": name,
        "vineyard": vineyard,
        "file": str(annotation_path),
        "rows": len(rows_data),
        "status": "pending",
    }


def update_manifest(entries: list[dict]) -> None:
    """Write manifest.json tracking all annotation files."""
    manifest_path = ANNOTATIONS_DIR / "manifest.json"

    # Merge with existing manifest if present
    existing = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for e in data.get("blocks", []):
                key = f"{e['vineyard']}_{e['block']}"
                existing[key] = e

    for e in entries:
        key = f"{e['vineyard']}_{e['block']}"
        existing[key] = e

    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_blocks": len(existing),
        "blocks": list(existing.values()),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Updated manifest: %d blocks", len(existing))


def main():
    parser = argparse.ArgumentParser(description="Prepare annotation dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Prepare all test blocks")
    group.add_argument("--block", type=str, help="Prepare a specific block by name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    blocks = load_test_blocks()
    if args.block:
        blocks = [b for b in blocks if b["name"] == args.block]
        if not blocks:
            print(f"Block '{args.block}' not found in test_blocks.json")
            sys.exit(1)

    config = PipelineConfig(save_debug_artifacts=False)

    print(f"\nPreparing dataset for {len(blocks)} blocks...")
    print(f"Images:      {IMAGES_DIR.resolve()}")
    print(f"Annotations: {ANNOTATIONS_DIR.resolve()}\n")

    manifest_entries = []
    for i, block in enumerate(blocks):
        name = block["name"]
        vineyard = block.get("vineyard_name", "Unknown")
        print(f"  [{i+1}/{len(blocks)}] {name} ({vineyard})...", end="", flush=True)

        try:
            entry = prepare_block(block, config)
            if entry:
                print(f" {entry['rows']} rows")
                manifest_entries.append(entry)
            else:
                print(" FAILED")
        except Exception as e:
            print(f" ERROR: {e}")
            logger.exception("Failed on %s", name)

        gc.collect()

    if manifest_entries:
        update_manifest(manifest_entries)

    print(f"\nDone. {len(manifest_entries)} blocks prepared.")
    print(f"Next: python annotate.py --block \"B10\"")


if __name__ == "__main__":
    main()
