#!/usr/bin/env python3
"""
Regression benchmark for vineyard row detection pipeline.

Runs all test blocks, computes per-block metrics (recall, spacing error, angle
error), and prints a summary table. Optionally compares old vs new pipeline and
saves results to CSV for tracking over time.

Usage:
    python benchmark.py                          # Run all blocks
    python benchmark.py --blocks "Block C,B10"   # Run subset
    python benchmark.py --compare                # Side-by-side old vs new
    python benchmark.py --save results.csv       # Append to CSV
    python benchmark.py --ridge-mode gabor       # Test alternative ridge mode
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
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
# Constants
# ---------------------------------------------------------------------------

BLOCK_CATEGORIES: dict[str, str] = {
    # Brooklands — grassed NZ inter-rows (LINZ)
    "Block A": "CLEAN_NZ",
    "Block B": "CLEAN_NZ",
    "Block C": "CLEAN_NZ",
    "Block D": "CLEAN_NZ",
    "Block D2": "CLEAN_NZ",
    # Opawa — large NZ blocks (LINZ)
    "North Block": "LARGE_NZ",
    "South Block": "LARGE_NZ",
    # The View — bare soil CA inter-rows (ArcGIS/Kelowna)
    "B10": "CLEAN_CA",
    "B12": "CLEAN_CA",
    # Other
    "Main Block": "OTHER",
}

COMPASS_TO_DEGREES: dict[str, float] = {
    "N-S": 90.0, "S-N": 90.0,
    "E-W": 0.0, "W-E": 0.0,
    "NE-SW": 45.0, "SW-NE": 45.0,
    "NW-SE": 135.0, "SE-NW": 135.0,
    "N": 90.0, "S": 90.0, "E": 0.0, "W": 0.0,
    "NE": 45.0, "SW": 45.0, "NW": 135.0, "SE": 135.0,
}


def bearing_to_image_angle(bearing_deg: float) -> float:
    """Convert geographic bearing to image-space angle (0-180)."""
    rad = math.radians(bearing_deg)
    return math.degrees(math.atan2(-math.cos(rad), math.sin(rad))) % 180.0


def angular_distance(a: float, b: float) -> float:
    """Angular distance accounting for 180-degree ambiguity."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _category(block: dict) -> str:
    """Determine block category from name or vineyard."""
    name = block["name"]
    if name in BLOCK_CATEGORIES:
        return BLOCK_CATEGORIES[name]
    vineyard = block.get("vineyard_name", "")
    if "Other" in vineyard:
        return "OTHER"
    return "OTHER"


# ---------------------------------------------------------------------------
# Block processing
# ---------------------------------------------------------------------------


def load_test_blocks(path: str = "test_blocks.json") -> list[dict]:
    resolved = Path(path)
    if not resolved.exists():
        logger.error("Test blocks file not found: %s", resolved)
        return []
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("blocks", [])


def run_block(block: dict, config: PipelineConfig) -> dict:
    """Run pipeline on one block and return structured metrics."""
    name = block["name"]
    vineyard = block.get("vineyard_name", "Unknown")
    coords = block["boundary"]["coordinates"][0]

    bbox = polygon_bbox(coords)
    centroid_lng = (bbox[0] + bbox[2]) / 2.0
    centroid_lat = (bbox[1] + bbox[3]) / 2.0
    source_name = auto_select_source(centroid_lng)
    source = TILE_SOURCES[source_name]
    zoom = default_zoom(source_name)

    # Fetch tiles
    t0 = time.perf_counter()
    image_bgr, mask, tile_origin = fetch_and_stitch(
        source, coords, zoom, source_name, cache_dir=config.tile_cache_dir,
    )
    fetch_time = time.perf_counter() - t0
    mpp = meters_per_pixel(centroid_lat, zoom, source.tile_size)

    # Run pipeline
    result = run_pipeline(
        image_bgr=image_bgr, mask=mask, mpp=mpp, lat=centroid_lat,
        zoom=zoom, tile_size=source.tile_size, tile_origin=tile_origin,
        tile_source=source_name, config=config,
    )

    # Ground truth
    gt_spacing = block.get("row_spacing_m")
    gt_count = block.get("row_count")
    gt_bearing = block.get("row_angle")
    gt_orientation = block.get("row_orientation")

    gt_angle_img: float | None = None
    if gt_bearing is not None:
        gt_angle_img = bearing_to_image_angle(gt_bearing)
    elif gt_orientation and gt_orientation in COMPASS_TO_DEGREES:
        gt_angle_img = COMPASS_TO_DEGREES[gt_orientation]

    # Compute metrics
    if result is None:
        return {
            "block": name, "vineyard": vineyard, "category": _category(block),
            "rows_det": 0, "rows_gt": gt_count, "recall": 0.0,
            "angle": None, "angle_gt": gt_angle_img, "angle_err": None,
            "spacing": None, "spacing_gt": gt_spacing, "spacing_err": None,
            "confidence": 0.0, "flags": "FAILED", "time_s": round(fetch_time, 1),
        }

    total_time = result.timings.total + fetch_time

    recall = None
    if gt_count and gt_count > 0:
        recall = min(result.row_count, gt_count) / gt_count

    spacing_err = None
    if gt_spacing and gt_spacing > 0:
        spacing_err = abs(result.mean_spacing_m - gt_spacing) / gt_spacing * 100.0

    angle_err = None
    if gt_angle_img is not None:
        angle_err = angular_distance(result.dominant_angle_deg, gt_angle_img)

    flags_str = str(result.quality_flags)
    if flags_str == "QualityFlag.NONE":
        flags_str = "NONE"
    else:
        # Compact flag names: "QualityFlag.SPACING_IRREGULAR|MISSING_ROWS" -> "IRREG,MISS"
        flags_str = flags_str.replace("QualityFlag.", "")
        abbrev = {
            "SPACING_IRREGULAR": "IRREG", "MISSING_ROWS": "MISS",
            "WEAK_SIGNAL": "WEAK", "LOW_CONFIDENCE": "LOW",
            "ORIENTATION_UNCERTAIN": "ORIENT", "FEW_ROWS": "FEW",
            "HEADLAND_DISTORTION": "HEAD", "HARMONIC_SPACING": "HARM",
        }
        parts = [abbrev.get(f.strip(), f.strip()) for f in flags_str.split("|")]
        flags_str = ",".join(parts)

    return {
        "block": name, "vineyard": vineyard, "category": _category(block),
        "rows_det": result.row_count, "rows_gt": gt_count,
        "recall": round(recall, 4) if recall is not None else None,
        "angle": round(result.dominant_angle_deg, 1),
        "angle_gt": round(gt_angle_img, 1) if gt_angle_img is not None else None,
        "angle_err": round(angle_err, 1) if angle_err is not None else None,
        "spacing": round(result.mean_spacing_m, 2),
        "spacing_gt": gt_spacing,
        "spacing_err": round(spacing_err, 1) if spacing_err is not None else None,
        "confidence": round(result.overall_confidence, 2),
        "flags": flags_str,
        "time_s": round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt(val, fmt_str, width, na="-"):
    """Format a value or return a right-aligned placeholder."""
    if val is None:
        return na.rjust(width)
    return f"{val:{fmt_str}}".rjust(width)


def print_table(results: list[dict]) -> None:
    """Print formatted benchmark results table."""
    header = (
        f"{'Block':<25} {'Rows':>4} {'GT':>4} {'Recall':>7} "
        f"{'Angle':>6} {'GT°':>6} {'Err°':>5} "
        f"{'Space':>6} {'GT_m':>5} {'Err%':>6} "
        f"{'Conf':>5} {'Flags':<16} {'Time':>5}"
    )
    sep = "-" * 115

    print(f"\n{'BENCHMARK RESULTS':=^115}")
    print(header)
    print(sep)

    for r in results:
        recall_str = f"{r['recall']*100:.1f}%" if r['recall'] is not None else "-"
        print(
            f"{r['block'] + ' (' + r['vineyard'][:10] + ')':<25} "
            f"{r['rows_det']:>4} "
            f"{_fmt(r['rows_gt'], 'd', 4)} "
            f"{recall_str:>7} "
            f"{_fmt(r['angle'], '.1f', 6)} "
            f"{_fmt(r['angle_gt'], '.1f', 6)} "
            f"{_fmt(r['angle_err'], '.1f', 5)} "
            f"{_fmt(r['spacing'], '.2f', 6)} "
            f"{_fmt(r['spacing_gt'], '.2f', 5)} "
            f"{_fmt(r['spacing_err'], '.1f', 6)} "
            f"{_fmt(r['confidence'], '.2f', 5)} "
            f"{r['flags']:<16} "
            f"{r['time_s']:>5.1f}"
        )

    print(f"{'':=^115}")


def print_aggregates(results: list[dict]) -> None:
    """Print aggregate summary statistics."""
    # Filter to blocks with ground truth
    with_recall = [r for r in results if r["recall"] is not None]
    with_spacing = [r for r in results if r["spacing_err"] is not None]
    with_angle = [r for r in results if r["angle_err"] is not None]

    if not with_recall:
        print("\nNo blocks with ground truth to aggregate.")
        return

    mean_recall = np.mean([r["recall"] for r in with_recall]) * 100
    mean_spacing_err = np.mean([r["spacing_err"] for r in with_spacing]) if with_spacing else 0
    mean_angle_err = np.mean([r["angle_err"] for r in with_angle]) if with_angle else 0
    blocks_80 = sum(1 for r in with_recall if r["recall"] >= 0.80)
    total_time = sum(r["time_s"] for r in results)

    print(f"\n{'AGGREGATE':=^60}")
    print(f"  Mean recall:      {mean_recall:5.1f}%  (target: >85%)")
    print(f"  Mean angle err:   {mean_angle_err:5.1f}°  (target: <2°)")
    print(f"  Mean spacing err: {mean_spacing_err:5.1f}%  (target: <5%)")
    target_80 = max(0, len(with_recall) - 2)
    print(f"  Blocks >=80% recall: {blocks_80}/{len(with_recall)} (target: {target_80}/{len(with_recall)})")
    print(f"  Total time:       {total_time:5.1f}s")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    if len(categories) > 1:
        print(f"\n  {'Category':<12} {'Recall':>7} {'Spac Err':>9} {'Ang Err':>8}")
        print(f"  {'-'*40}")
        for cat in categories:
            cat_recall = [r for r in with_recall if r["category"] == cat]
            cat_spacing = [r for r in with_spacing if r["category"] == cat]
            cat_angle = [r for r in with_angle if r["category"] == cat]
            cr = np.mean([r["recall"] for r in cat_recall]) * 100 if cat_recall else 0
            cs = np.mean([r["spacing_err"] for r in cat_spacing]) if cat_spacing else 0
            ca = np.mean([r["angle_err"] for r in cat_angle]) if cat_angle else 0
            print(f"  {cat:<12} {cr:6.1f}% {cs:8.1f}% {ca:7.1f}°")

    print(f"{'':=^60}\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp", "ridge_mode", "block", "vineyard", "category",
    "rows_det", "rows_gt", "recall", "angle", "angle_gt", "angle_err",
    "spacing", "spacing_gt", "spacing_err", "confidence", "flags", "time_s",
]


def save_csv(results: list[dict], path: str, ridge_mode: str = "hessian") -> None:
    """Append results to CSV with timestamp."""
    csv_path = Path(path)
    write_header = not csv_path.exists()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in results:
            row = {**r, "timestamp": ts, "ridge_mode": ridge_mode}
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    print(f"Results appended to {csv_path} ({len(results)} rows)")


# ---------------------------------------------------------------------------
# Old pipeline comparison
# ---------------------------------------------------------------------------

def run_old_pipeline(block: dict) -> dict | None:
    """Run the old fft2d + row_locator pipeline for comparison.

    Returns a metrics dict compatible with the benchmark table, or None if
    the old pipeline isn't available or fails.
    """
    try:
        from detect_rows import process_block as old_process_block
        from vinerow.config import PipelineConfig as _PC

        config = _PC(save_debug_artifacts=False)
        summary = old_process_block(block, config)
        if summary is None:
            return None

        gt_count = block.get("row_count")
        recall = None
        if gt_count and gt_count > 0 and summary.get("rows_detected"):
            recall = min(summary["rows_detected"], gt_count) / gt_count

        return {
            "block": block["name"],
            "vineyard": block.get("vineyard_name", "Unknown"),
            "category": _category(block),
            "rows_det": summary.get("rows_detected", 0),
            "rows_gt": gt_count,
            "recall": round(recall, 4) if recall is not None else None,
            "angle": summary.get("angle_deg"),
            "angle_gt": None,
            "angle_err": summary.get("angle_error_deg"),
            "spacing": summary.get("spacing_m"),
            "spacing_gt": block.get("row_spacing_m"),
            "spacing_err": summary.get("spacing_error_pct"),
            "confidence": summary.get("confidence", 0),
            "flags": "",
            "time_s": summary.get("total_time_s", 0),
        }
    except Exception as e:
        logger.warning("Old pipeline failed for %s: %s", block["name"], e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Regression benchmark for vineyard row detection"
    )
    parser.add_argument(
        "--blocks", type=str, default=None,
        help="Comma-separated block names to run (default: all)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["CLEAN_NZ", "LARGE_NZ", "CLEAN_CA", "OTHER"],
        help="Run only blocks in this category"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run old pipeline for side-by-side comparison"
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="FILE",
        help="Append results to CSV file"
    )
    parser.add_argument(
        "--ridge-mode", type=str, default=None,
        help="Override ridge mode in pipeline config"
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Skip saving debug artifacts (faster)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug logging"
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load blocks
    all_blocks = load_test_blocks()
    if not all_blocks:
        print("No test blocks found.")
        sys.exit(1)

    # Filter
    blocks = all_blocks
    if args.blocks:
        names = [n.strip() for n in args.blocks.split(",")]
        blocks = [b for b in all_blocks if b["name"] in names]
        if not blocks:
            print(f"No blocks matched: {names}")
            sys.exit(1)
    if args.category:
        blocks = [b for b in blocks if _category(b) == args.category]
        if not blocks:
            print(f"No blocks in category: {args.category}")
            sys.exit(1)

    # Config
    config = PipelineConfig(
        save_debug_artifacts=not args.no_debug,
    )
    ridge_mode = args.ridge_mode or config.ridge_mode
    if args.ridge_mode:
        if hasattr(config, "ridge_mode"):
            config.ridge_mode = args.ridge_mode
        else:
            logger.warning("PipelineConfig has no ridge_mode field yet (Step 3)")

    # Run new pipeline
    print(f"\nRunning benchmark on {len(blocks)} blocks (ridge_mode={ridge_mode})...")
    results: list[dict] = []
    for i, block in enumerate(blocks):
        name = block["name"]
        vineyard = block.get("vineyard_name", "")
        print(f"  [{i+1}/{len(blocks)}] {name} ({vineyard})...", end="", flush=True)
        t0 = time.perf_counter()
        r = run_block(block, config)
        elapsed = time.perf_counter() - t0
        recall_str = f"{r['recall']*100:.0f}%" if r['recall'] is not None else "N/A"
        print(f" {r['rows_det']} rows, recall={recall_str} ({elapsed:.1f}s)")
        results.append(r)

    print_table(results)
    print_aggregates(results)

    # CSV export
    if args.save:
        save_csv(results, args.save, ridge_mode)

    # Old pipeline comparison
    if args.compare:
        print("\n--- OLD PIPELINE (fft2d + row_locator) ---")
        old_results: list[dict] = []
        for block in blocks:
            print(f"  [old] {block['name']}...", end="", flush=True)
            old_r = run_old_pipeline(block)
            if old_r:
                recall_str = f"{old_r['recall']*100:.0f}%" if old_r['recall'] is not None else "N/A"
                print(f" {old_r['rows_det']} rows, recall={recall_str}")
                old_results.append(old_r)
            else:
                print(" FAILED")

        if old_results:
            print_table(old_results)
            print_aggregates(old_results)

            # Side-by-side comparison
            print(f"\n{'COMPARISON (New vs Old)':=^70}")
            print(f"  {'Block':<20} {'New Recall':>10} {'Old Recall':>10} {'Delta':>7}")
            print(f"  {'-'*50}")
            for nr in results:
                old_match = next((o for o in old_results if o["block"] == nr["block"]), None)
                nr_str = f"{nr['recall']*100:.1f}%" if nr['recall'] is not None else "N/A"
                if old_match and old_match["recall"] is not None and nr["recall"] is not None:
                    or_str = f"{old_match['recall']*100:.1f}%"
                    delta = (nr["recall"] - old_match["recall"]) * 100
                    d_str = f"{delta:+.1f}pp"
                else:
                    or_str = "N/A"
                    d_str = "-"
                print(f"  {nr['block']:<20} {nr_str:>10} {or_str:>10} {d_str:>7}")
            print(f"{'':=^70}\n")

    # Exit code: 0 if mean recall >= 70%
    with_recall = [r for r in results if r["recall"] is not None]
    if with_recall:
        mean_recall = np.mean([r["recall"] for r in with_recall])
        sys.exit(0 if mean_recall >= 0.70 else 1)


if __name__ == "__main__":
    main()
