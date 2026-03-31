#!/usr/bin/env python3
"""
CLI entry point for vineyard row detection pipeline.

Usage:
    python cli.py --block "Block C" --source linz --zoom 20
    python cli.py --all
    python cli.py --geojson boundary.geojson --source arcgis --zoom 19
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

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
from vinerow.pipeline import run_pipeline

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compass-to-angle mapping (for ground truth comparison)
# ---------------------------------------------------------------------------

COMPASS_TO_DEGREES: dict[str, float] = {
    "N-S": 90.0, "S-N": 90.0,
    "E-W": 0.0, "W-E": 0.0,
    "NE-SW": 45.0, "SW-NE": 45.0,
    "NW-SE": 135.0, "SE-NW": 135.0,
    "N": 90.0, "S": 90.0,
    "E": 0.0, "W": 0.0,
    "NE": 45.0, "SW": 45.0,
    "NW": 135.0, "SE": 135.0,
}


def bearing_to_image_angle(bearing_deg: float) -> float:
    """Convert geographic bearing to image-space angle (0-180)."""
    rad = math.radians(bearing_deg)
    return math.degrees(math.atan2(-math.cos(rad), math.sin(rad))) % 180.0


def angular_distance(a: float, b: float) -> float:
    """Angular distance accounting for 180-degree ambiguity."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


# ---------------------------------------------------------------------------
# Block loading
# ---------------------------------------------------------------------------


def load_test_blocks(path: str = "test_blocks.json") -> list[dict]:
    """Load test block data from JSON file."""
    resolved = Path(path)
    if not resolved.exists():
        logger.error("Test blocks file not found: %s", resolved)
        return []
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = data.get("blocks", [])
    logger.info("Loaded %d test blocks from %s", len(blocks), resolved)
    return blocks


def load_geojson_block(path: str) -> dict:
    """Load a block from a GeoJSON file."""
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    if geojson.get("type") == "FeatureCollection":
        feature = geojson["features"][0]
        geometry = feature["geometry"]
    elif geojson.get("type") == "Feature":
        geometry = geojson["geometry"]
    else:
        geometry = geojson
    return {
        "name": Path(path).stem,
        "vineyard_name": "GeoJSON",
        "boundary": geometry,
    }


# ---------------------------------------------------------------------------
# Single block processing
# ---------------------------------------------------------------------------


def process_block(block: dict, config: PipelineConfig) -> dict | None:
    """Run the full pipeline on one block and return results."""
    name = block["name"]
    vineyard = block.get("vineyard_name", "Unknown")
    boundary = block["boundary"]
    coords = boundary["coordinates"][0]

    logger.info("=" * 60)
    logger.info("Processing: %s (%s)", name, vineyard)
    logger.info("=" * 60)

    # Determine tile source
    bbox = polygon_bbox(coords)
    centroid_lng = (bbox[0] + bbox[2]) / 2.0
    centroid_lat = (bbox[1] + bbox[3]) / 2.0
    source_name = auto_select_source(centroid_lng)
    source = TILE_SOURCES[source_name]
    zoom = default_zoom(source_name)

    # Fetch and stitch
    t0 = time.perf_counter()
    image_bgr, mask, tile_origin = fetch_and_stitch(
        source, coords, zoom, source_name, cache_dir=config.tile_cache_dir,
    )
    fetch_time = time.perf_counter() - t0

    mpp = meters_per_pixel(centroid_lat, zoom, source.tile_size)

    # Run pipeline
    result = run_pipeline(
        image_bgr=image_bgr,
        mask=mask,
        mpp=mpp,
        lat=centroid_lat,
        zoom=zoom,
        tile_size=source.tile_size,
        tile_origin=tile_origin,
        tile_source=source_name,
        config=config,
    )

    if result is None:
        logger.error("Detection failed for %s", name)
        return None

    # Ground truth comparison
    gt_spacing = block.get("row_spacing_m")
    gt_orientation = block.get("row_orientation")
    gt_angle_bearing = block.get("row_angle")
    gt_row_count = block.get("row_count")

    gt_angle_img: float | None = None
    if gt_angle_bearing is not None:
        gt_angle_img = bearing_to_image_angle(gt_angle_bearing)
    elif gt_orientation and gt_orientation in COMPASS_TO_DEGREES:
        gt_angle_img = COMPASS_TO_DEGREES[gt_orientation]

    spacing_error_pct = None
    if gt_spacing and gt_spacing > 0:
        spacing_error_pct = abs(result.mean_spacing_m - gt_spacing) / gt_spacing * 100.0

    angle_error = None
    if gt_angle_img is not None:
        angle_error = angular_distance(result.dominant_angle_deg, gt_angle_img)

    row_count_error = None
    if gt_row_count:
        row_count_error = abs(result.row_count - gt_row_count)

    # Save debug artifacts
    if config.save_debug_artifacts:
        from vinerow.debug.artifacts import save_all_artifacts
        safe_name = name.replace(" ", "_").replace("/", "_")
        output_dir = Path(config.debug_output_dir) / safe_name
        save_all_artifacts(output_dir, result, block_name=name)

    # Summary
    summary = {
        "block": name,
        "vineyard": vineyard,
        "source": source_name,
        "zoom": zoom,
        "rows_detected": result.row_count,
        "angle_deg": result.dominant_angle_deg,
        "angle_bearing": result.dominant_angle_bearing,
        "spacing_m": result.mean_spacing_m,
        "spacing_std_m": result.spacing_std_m,
        "confidence": result.overall_confidence,
        "flags": str(result.quality_flags),
        "total_time_s": round(result.timings.total + fetch_time, 2),
        "gt_spacing_m": gt_spacing,
        "gt_orientation": gt_orientation,
        "gt_row_count": gt_row_count,
        "spacing_error_pct": round(spacing_error_pct, 1) if spacing_error_pct is not None else None,
        "angle_error_deg": round(angle_error, 1) if angle_error is not None else None,
        "row_count_error": row_count_error,
    }

    logger.info(
        "Result: %d rows, angle=%.1f deg (err=%s), spacing=%.2fm (err=%s), conf=%.2f",
        result.row_count,
        result.dominant_angle_deg,
        f"{angle_error:.1f}°" if angle_error is not None else "N/A",
        result.mean_spacing_m,
        f"{spacing_error_pct:.1f}%" if spacing_error_pct is not None else "N/A",
        result.overall_confidence,
    )

    return summary


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    header = (
        f"{'Block':<20} {'Rows':>5} {'Angle':>7} {'Err°':>5} "
        f"{'Space':>6} {'Err%':>5} {'Conf':>5} {'Time':>6} {'Flags'}"
    )
    print("\n" + "=" * 90)
    print(header)
    print("-" * 90)

    for r in results:
        angle_err = f"{r['angle_error_deg']:.1f}" if r.get("angle_error_deg") is not None else "-"
        space_err = f"{r['spacing_error_pct']:.1f}" if r.get("spacing_error_pct") is not None else "-"
        flags = r.get("flags", "")
        if flags == "QualityFlag.NONE":
            flags = ""

        print(
            f"{r['block']:<20} {r['rows_detected']:>5} "
            f"{r['angle_deg']:>7.1f} {angle_err:>5} "
            f"{r['spacing_m']:>6.2f} {space_err:>5} "
            f"{r['confidence']:>5.2f} {r['total_time_s']:>6.1f} {flags}"
        )

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Vineyard Row Detection Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--block", type=str, help="Process a single block by name")
    group.add_argument("--all", action="store_true", help="Process all test blocks")
    group.add_argument("--geojson", type=str, help="Process a GeoJSON file")

    parser.add_argument("--source", choices=["linz", "arcgis", "kelowna"], help="Tile source")
    parser.add_argument("--zoom", type=int, help="Zoom level override")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--no-debug", action="store_true", help="Skip debug artifacts")
    parser.add_argument(
        "--ridge-mode",
        choices=["hessian", "luminance", "exg_only", "gabor", "ensemble",
                 "hessian_small", "hessian_large"],
        default=None,
        help="Ridge detection strategy (default: hessian)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_kwargs: dict = {
        "save_debug_artifacts": not args.no_debug,
        "debug_output_dir": args.output,
    }
    if args.ridge_mode:
        config_kwargs["ridge_mode"] = args.ridge_mode
    config = PipelineConfig(**config_kwargs)

    # Load blocks
    if args.geojson:
        blocks = [load_geojson_block(args.geojson)]
    elif args.block:
        all_blocks = load_test_blocks()
        blocks = [b for b in all_blocks if b["name"] == args.block]
        if not blocks:
            logger.error("Block '%s' not found in test_blocks.json", args.block)
            sys.exit(1)
    else:
        blocks = load_test_blocks()

    if not blocks:
        logger.error("No blocks to process")
        sys.exit(1)

    # Process
    results = []
    for block in blocks:
        summary = process_block(block, config)
        if summary:
            results.append(summary)

    if results:
        print_results_table(results)

        # Save JSON results
        output_path = Path(args.output) / "results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
