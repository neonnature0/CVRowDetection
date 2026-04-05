#!/usr/bin/env python3
"""
Evaluate pipeline detection against annotated ground truth.

Uses bipartite matching (Hungarian algorithm) to pair detected rows with
GT rows, then computes precision, recall, F1, and localization error.

Usage:
    python evaluate_gt.py                     # All annotated blocks
    python evaluate_gt.py --block "B10"       # Single block
    python evaluate_gt.py --report            # Generate markdown report
    python evaluate_gt.py --status complete   # Only evaluate complete annotations
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment

from vinerow.acquisition.geo_utils import meters_per_pixel, polygon_bbox
from vinerow.acquisition.tile_fetcher import (
    TILE_SOURCES,
    auto_select_source,
    default_zoom,
    fetch_and_stitch,
)
from vinerow.config import PipelineConfig
from vinerow.loaders.json_loader import load_test_blocks as _load_test_blocks
from vinerow.pipeline import run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s >> %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("dataset")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
EVALUATION_DIR = DATASET_DIR / "evaluation"

COMPASS_TO_DEGREES: dict[str, float] = {
    "N-S": 90.0, "S-N": 90.0, "E-W": 0.0, "W-E": 0.0,
    "NE-SW": 45.0, "SW-NE": 45.0, "NW-SE": 135.0, "SE-NW": 135.0,
}


def bearing_to_image_angle(bearing_deg: float) -> float:
    rad = math.radians(bearing_deg)
    return math.degrees(math.atan2(-math.cos(rad), math.sin(rad))) % 180.0


def angular_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    block: str
    vineyard: str
    n_gt: int
    n_det: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    localization_error_m: float
    localization_error_px: float
    spacing_error_pct: float | None
    angle_error_deg: float | None
    false_positives: int
    false_negatives: int
    time_s: float


def match_rows(
    gt_perps: list[float],
    det_perps: list[float],
    mpp: float,
    match_threshold_factor: float = 0.4,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[float]]:
    """Match detected rows to GT rows using bipartite assignment.

    Args:
        gt_perps: Ground truth perpendicular positions (sorted).
        det_perps: Detected perpendicular positions (sorted).
        mpp: Meters per pixel.
        match_threshold_factor: Max distance as fraction of median GT spacing.

    Returns:
        (matched_pairs, unmatched_gt, unmatched_det, matched_distances)
    """
    n_gt = len(gt_perps)
    n_det = len(det_perps)

    if n_gt == 0 or n_det == 0:
        return [], list(range(n_gt)), list(range(n_det)), []

    # Compute match threshold from GT spacings
    if n_gt >= 2:
        gt_spacings = [gt_perps[i+1] - gt_perps[i] for i in range(n_gt - 1)]
        median_spacing = float(np.median(gt_spacings))
    else:
        median_spacing = 25.0  # fallback ~2.5m at 0.1 mpp

    threshold = match_threshold_factor * median_spacing

    # Cost matrix: absolute perpendicular distance
    cost = np.zeros((n_gt, n_det), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_det):
            cost[i, j] = abs(gt_perps[i] - det_perps[j])

    # Hungarian assignment
    gt_indices, det_indices = linear_sum_assignment(cost)

    matched_pairs = []
    matched_distances = []
    matched_gt_set = set()
    matched_det_set = set()

    for gi, di in zip(gt_indices, det_indices):
        dist = cost[gi, di]
        if dist <= threshold:
            matched_pairs.append((gi, di))
            matched_distances.append(dist)
            matched_gt_set.add(gi)
            matched_det_set.add(di)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt_set]
    unmatched_det = [j for j in range(n_det) if j not in matched_det_set]

    return matched_pairs, unmatched_gt, unmatched_det, matched_distances


def evaluate_block(annotation_path: Path, config: PipelineConfig) -> EvalResult | None:
    """Run pipeline and compare against annotation ground truth."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    block_name = ann["block_name"]
    vineyard = ann["vineyard_name"]
    mpp = ann["meters_per_pixel"]
    gt_angle = ann["angle_deg"]
    w, h = ann["image_size"]

    # GT row positions from annotation (extract mean perp from polylines)
    cx, cy = w / 2.0, h / 2.0
    gt_angle_rad = math.radians(gt_angle)
    gt_pdx, gt_pdy = -math.sin(gt_angle_rad), math.cos(gt_angle_rad)

    gt_perps = []
    for r in ann["rows"]:
        if "centerline_px" in r and r["centerline_px"]:
            perps = [(p[0] - cx) * gt_pdx + (p[1] - cy) * gt_pdy for p in r["centerline_px"]]
            gt_perps.append(float(np.mean(perps)))
        elif "perp_position_px" in r:
            gt_perps.append(r["perp_position_px"])
    gt_perps.sort()
    n_gt = len(gt_perps)

    if n_gt == 0:
        logger.warning("No GT rows in %s", annotation_path.name)
        return None

    # Load block definition from test_blocks.json to get boundary
    test_blocks = _load_test_blocks()
    block_def = next(
        (b for b in test_blocks
         if b["name"] == block_name and b.get("vineyard_name") == vineyard),
        None,
    )
    if block_def is None:
        logger.error("Block %s/%s not found in test_blocks.json", vineyard, block_name)
        return None

    # Run pipeline
    coords = block_def["boundary"]["coordinates"][0]
    bbox = polygon_bbox(coords)
    cx_lng = (bbox[0] + bbox[2]) / 2.0
    cx_lat = (bbox[1] + bbox[3]) / 2.0
    source_name = auto_select_source(cx_lng)
    source = TILE_SOURCES[source_name]
    zoom = default_zoom(source_name)

    t0 = time.perf_counter()
    image_bgr, mask, tile_origin = fetch_and_stitch(
        source, coords, zoom, source_name, cache_dir=config.tile_cache_dir,
    )
    run_mpp = meters_per_pixel(cx_lat, zoom, source.tile_size)

    result = run_pipeline(
        image_bgr=image_bgr, mask=mask, mpp=run_mpp, lat=cx_lat,
        zoom=zoom, tile_size=source.tile_size, tile_origin=tile_origin,
        tile_source=source_name, config=config,
        block_name=block_name, vineyard_name=vineyard,
    )
    elapsed = time.perf_counter() - t0

    if result is None:
        logger.error("Pipeline failed for %s", block_name)
        return EvalResult(
            block=block_name, vineyard=vineyard,
            n_gt=n_gt, n_det=0, n_matched=0,
            precision=0.0, recall=0.0, f1=0.0,
            localization_error_m=0.0, localization_error_px=0.0,
            spacing_error_pct=None, angle_error_deg=None,
            false_positives=0, false_negatives=n_gt, time_s=elapsed,
        )

    # Extract detected perpendicular positions
    det_angle = result.dominant_angle_deg
    cx_img, cy_img = w / 2.0, h / 2.0
    angle_rad = math.radians(det_angle)
    pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)

    det_perps = []
    for row in result.rows:
        perps = [(x - cx_img) * pdx + (y - cy_img) * pdy for x, y in row.centerline_px]
        det_perps.append(float(np.mean(perps)))
    det_perps.sort()
    n_det = len(det_perps)

    # Match
    matched, unmatched_gt, unmatched_det, distances = match_rows(gt_perps, det_perps, mpp)
    n_matched = len(matched)

    precision = n_matched / n_det if n_det > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    loc_error_px = float(np.mean(distances)) if distances else 0.0
    loc_error_m = loc_error_px * mpp

    # Spacing error
    gt_sp = ann.get("ground_truth", {}).get("gt_spacing_m")
    spacing_err = None
    if gt_sp and gt_sp > 0:
        spacing_err = abs(result.mean_spacing_m - gt_sp) / gt_sp * 100.0

    # Angle error
    angle_err = angular_distance(det_angle, gt_angle) if gt_angle else None

    # Save overlay image
    _save_overlay(
        annotation_path, image_bgr, mask, gt_perps, det_perps,
        matched, unmatched_gt, unmatched_det,
        gt_angle, det_angle, cx_img, cy_img, w, h,
    )

    return EvalResult(
        block=block_name, vineyard=vineyard,
        n_gt=n_gt, n_det=n_det, n_matched=n_matched,
        precision=round(precision, 4), recall=round(recall, 4),
        f1=round(f1, 4),
        localization_error_m=round(loc_error_m, 4),
        localization_error_px=round(loc_error_px, 2),
        spacing_error_pct=round(spacing_err, 1) if spacing_err is not None else None,
        angle_error_deg=round(angle_err, 2) if angle_err is not None else None,
        false_positives=len(unmatched_det),
        false_negatives=len(unmatched_gt),
        time_s=round(elapsed, 1),
    )


def _save_overlay(
    annotation_path: Path,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    gt_perps: list[float],
    det_perps: list[float],
    matched: list[tuple[int, int]],
    unmatched_gt: list[int],
    unmatched_det: list[int],
    gt_angle: float,
    det_angle: float,
    cx: float, cy: float,
    w: int, h: int,
):
    """Save an overlay image showing GT vs detected rows with matches."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    canvas = image_bgr.copy()
    diag = math.sqrt(w**2 + h**2)

    def _draw_line(perp, angle, color, thickness=1):
        rad = math.radians(angle)
        rdx, rdy = math.cos(rad), math.sin(rad)
        pdx, pdy = -math.sin(rad), math.cos(rad)
        lx, ly = cx + perp * pdx, cy + perp * pdy
        x1, y1 = int(lx - diag * rdx), int(ly - diag * rdy)
        x2, y2 = int(lx + diag * rdx), int(ly + diag * rdy)
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Draw matched GT (green) and detected (magenta)
    matched_gt_set = {g for g, d in matched}
    matched_det_set = {d for g, d in matched}

    for i, perp in enumerate(gt_perps):
        if i in matched_gt_set:
            _draw_line(perp, gt_angle, (0, 200, 0), 2)  # green
        else:
            _draw_line(perp, gt_angle, (255, 100, 0), 2)  # blue = false negative

    for j, perp in enumerate(det_perps):
        if j in matched_det_set:
            _draw_line(perp, det_angle, (255, 0, 255), 1)  # magenta
        else:
            _draw_line(perp, det_angle, (0, 0, 255), 2)  # red = false positive

    # Draw match connections (thin white lines between matched row centers)
    for gi, di in matched:
        gt_rad = math.radians(gt_angle)
        det_rad = math.radians(det_angle)
        gt_x = int(cx + gt_perps[gi] * (-math.sin(gt_rad)))
        gt_y = int(cy + gt_perps[gi] * math.cos(gt_rad))
        det_x = int(cx + det_perps[di] * (-math.sin(det_rad)))
        det_y = int(cy + det_perps[di] * math.cos(det_rad))
        cv2.line(canvas, (gt_x, gt_y), (det_x, det_y), (255, 255, 255), 1, cv2.LINE_AA)

    # Mask boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)

    stem = annotation_path.stem
    out_path = EVALUATION_DIR / f"{stem}_eval.png"
    cv2.imwrite(str(out_path), canvas)
    logger.info("Saved overlay: %s", out_path.name)




# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results: list[EvalResult]):
    """Print formatted results table."""
    header = (
        f"{'Block':<25} {'GT':>4} {'Det':>4} {'Match':>5} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'Loc(m)':>7} {'SpErr%':>7} {'AngErr':>7} "
        f"{'FP':>3} {'FN':>3} {'Time':>5}"
    )
    sep = "-" * 105

    print(f"\n{'GROUND TRUTH EVALUATION':=^105}")
    print(header)
    print(sep)

    for r in results:
        sp_err = f"{r.spacing_error_pct:.1f}" if r.spacing_error_pct is not None else "-"
        a_err = f"{r.angle_error_deg:.1f}" if r.angle_error_deg is not None else "-"
        print(
            f"{r.block + ' (' + r.vineyard[:10] + ')':<25} "
            f"{r.n_gt:>4} {r.n_det:>4} {r.n_matched:>5} "
            f"{r.precision*100:>5.1f}% {r.recall*100:>5.1f}% {r.f1*100:>5.1f}% "
            f"{r.localization_error_m:>7.3f} {sp_err:>7} {a_err:>7} "
            f"{r.false_positives:>3} {r.false_negatives:>3} {r.time_s:>5.1f}"
        )

    print(f"{'':=^105}")

    # Aggregates
    if results:
        mean_p = np.mean([r.precision for r in results])
        mean_r = np.mean([r.recall for r in results])
        mean_f1 = np.mean([r.f1 for r in results])
        mean_loc = np.mean([r.localization_error_m for r in results])
        print(f"\n  Mean precision:  {mean_p*100:.1f}%")
        print(f"  Mean recall:     {mean_r*100:.1f}%")
        print(f"  Mean F1:         {mean_f1*100:.1f}%")
        print(f"  Mean loc error:  {mean_loc:.3f}m")
        print()


def generate_report(results: list[EvalResult], path: Path):
    """Generate a markdown evaluation report."""
    lines = ["# Ground Truth Evaluation Report\n"]
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    lines.append("## Per-Block Results\n")
    lines.append("| Block | GT | Det | Match | Prec | Rec | F1 | Loc(m) | FP | FN |")
    lines.append("|-------|---:|----:|------:|-----:|----:|---:|-------:|---:|---:|")

    for r in results:
        lines.append(
            f"| {r.block} ({r.vineyard}) | {r.n_gt} | {r.n_det} | {r.n_matched} | "
            f"{r.precision*100:.1f}% | {r.recall*100:.1f}% | {r.f1*100:.1f}% | "
            f"{r.localization_error_m:.3f} | {r.false_positives} | {r.false_negatives} |"
        )

    if results:
        lines.append("\n## Aggregates\n")
        lines.append(f"- Mean precision: {np.mean([r.precision for r in results])*100:.1f}%")
        lines.append(f"- Mean recall: {np.mean([r.recall for r in results])*100:.1f}%")
        lines.append(f"- Mean F1: {np.mean([r.f1 for r in results])*100:.1f}%")
        lines.append(f"- Mean localization error: {np.mean([r.localization_error_m for r in results]):.3f}m")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved: %s", path)


# Need this import for the report
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against ground truth")
    parser.add_argument("--block", type=str, help="Evaluate a specific block")
    parser.add_argument("--status", type=str, default=None,
                        choices=["pending", "modified", "complete"],
                        help="Only evaluate blocks with this status")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find annotation files
    ann_files = sorted(
        p for p in ANNOTATIONS_DIR.glob("*.json")
        if p.name != "manifest.json"
    )

    if not ann_files:
        print("No annotation files found. Run prepare_dataset.py first.")
        sys.exit(1)

    # Filter
    if args.block:
        filtered = []
        for f in ann_files:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if d["block_name"] == args.block:
                filtered.append(f)
        ann_files = filtered

    if args.status:
        filtered = []
        for f in ann_files:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("metadata", {}).get("status") == args.status:
                filtered.append(f)
        ann_files = filtered

    if not ann_files:
        print("No matching annotation files.")
        sys.exit(1)

    config = PipelineConfig(save_debug_artifacts=False)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {len(ann_files)} blocks...")
    results = []
    for i, f in enumerate(ann_files):
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        name = d["block_name"]
        vineyard = d["vineyard_name"]
        print(f"  [{i+1}/{len(ann_files)}] {name} ({vineyard})...", end="", flush=True)

        try:
            r = evaluate_block(f, config)
            if r:
                print(f" P={r.precision*100:.0f}% R={r.recall*100:.0f}% F1={r.f1*100:.0f}% loc={r.localization_error_m:.3f}m")
                results.append(r)
            else:
                print(" SKIPPED")
        except Exception as e:
            print(f" ERROR: {e}")
            logger.exception("Failed on %s", name)

        gc.collect()

    if results:
        print_results(results)

        if args.report:
            report_path = EVALUATION_DIR / "report.md"
            generate_report(results, report_path)

    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
