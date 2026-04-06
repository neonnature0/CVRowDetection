#!/usr/bin/env python3
"""
Evaluate pipeline detection against annotated ground truth.

Uses bipartite matching (Hungarian algorithm) to pair detected rows with
GT rows, then computes precision, recall, F1, localization error, and
shape-based distance.

Reports F1 at three match thresholds (loose/medium/strict) to show how
accuracy degrades with tighter tolerances. Shows both unweighted and
row-weighted means.

Usage:
    python evaluate_gt.py                     # All annotated blocks
    python evaluate_gt.py --block "a3f2c1"    # Single block
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

# Three match threshold levels (as fraction of median GT spacing)
THRESHOLD_LOOSE = 0.4   # ~1.0m at 2.5m spacing — current default
THRESHOLD_MEDIUM = 0.2  # ~0.5m
THRESHOLD_STRICT = 0.1  # ~0.25m


def bearing_to_image_angle(bearing_deg: float) -> float:
    rad = math.radians(bearing_deg)
    return math.degrees(math.atan2(-math.cos(rad), math.sin(rad))) % 180.0


def angular_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _interpolate_dense(points: list, step: float = 5.0) -> list[tuple[float, float]]:
    """Interpolate a polyline to dense points at given pixel step.

    Handles both list-of-tuples and list-of-lists input.
    """
    if len(points) < 2:
        return [(float(p[0]), float(p[1])) for p in points]

    result = []
    for i in range(len(points) - 1):
        x0, y0 = float(points[i][0]), float(points[i][1])
        x1, y1 = float(points[i + 1][0]), float(points[i + 1][1])
        seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        n_steps = max(1, int(seg_len / step))
        for j in range(n_steps):
            t = j / n_steps
            result.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    result.append((float(points[-1][0]), float(points[-1][1])))
    return result


def _point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _point_to_polyline_dist(px: float, py: float, polyline: list[tuple[float, float]]) -> float:
    """Minimum distance from a point to any segment of a polyline."""
    min_d = float("inf")
    for i in range(len(polyline) - 1):
        d = _point_to_segment_dist(px, py, polyline[i][0], polyline[i][1],
                                   polyline[i + 1][0], polyline[i + 1][1])
        if d < min_d:
            min_d = d
    return min_d


def polyline_shape_distance(
    gt_points: list,
    det_points: list,
    dense_step: float = 5.0,
) -> float:
    """Mean bidirectional point-to-polyline distance between two polylines.

    Both polylines are first dense-interpolated to `dense_step` pixel
    resolution to avoid sparse control-point sampling artifacts.
    """
    gt_dense = _interpolate_dense(gt_points, dense_step)
    det_dense = _interpolate_dense(det_points, dense_step)

    if len(gt_dense) < 2 or len(det_dense) < 2:
        return float("inf")

    # GT → detected
    gt_to_det = [_point_to_polyline_dist(p[0], p[1], det_dense) for p in gt_dense]
    # Detected → GT
    det_to_gt = [_point_to_polyline_dist(p[0], p[1], gt_dense) for p in det_dense]

    return float(np.mean(gt_to_det + det_to_gt))


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
    f1_medium: float        # F1 at 0.2x threshold
    f1_strict: float        # F1 at 0.1x threshold
    localization_error_m: float
    localization_error_px: float
    shape_error_px: float   # mean bidirectional point-to-polyline distance
    shape_error_m: float
    spacing_error_pct: float | None
    angle_error_deg: float | None
    false_positives: int
    false_negatives: int
    time_s: float
    is_blind: bool = False


def match_rows(
    gt_perps: list[float],
    det_perps: list[float],
    mpp: float,
    match_threshold_factor: float = 0.4,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[float]]:
    """Match detected rows to GT rows using bipartite assignment."""
    n_gt = len(gt_perps)
    n_det = len(det_perps)

    if n_gt == 0 or n_det == 0:
        return [], list(range(n_gt)), list(range(n_det)), []

    if n_gt >= 2:
        gt_spacings = [gt_perps[i + 1] - gt_perps[i] for i in range(n_gt - 1)]
        median_spacing = float(np.median(gt_spacings))
    else:
        median_spacing = 25.0

    threshold = match_threshold_factor * median_spacing

    cost = np.zeros((n_gt, n_det), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_det):
            cost[i, j] = abs(gt_perps[i] - det_perps[j])

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


def _compute_f1(gt_perps, det_perps, mpp, threshold_factor):
    """Quick F1 at a given threshold."""
    matched, _, _, _ = match_rows(gt_perps, det_perps, mpp, threshold_factor)
    n_matched = len(matched)
    n_gt, n_det = len(gt_perps), len(det_perps)
    p = n_matched / n_det if n_det > 0 else 0.0
    r = n_matched / n_gt if n_gt > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_block(annotation_path: Path, config: PipelineConfig) -> EvalResult | None:
    """Run pipeline and compare against annotation ground truth."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    block_name = ann["block_name"]
    vineyard = ann["vineyard_name"]
    mpp = ann.get("meters_per_pixel", 0.0)
    gt_angle = ann["angle_deg"]
    w, h = ann["image_size"]
    is_blind = ann.get("metadata", {}).get("blind", False)

    # GT row polylines and perpendicular positions
    cx, cy = w / 2.0, h / 2.0
    gt_angle_rad = math.radians(gt_angle)
    gt_pdx, gt_pdy = -math.sin(gt_angle_rad), math.cos(gt_angle_rad)

    gt_perps = []
    gt_polylines = []  # keep polylines for shape distance
    for r in ann["rows"]:
        pts = r.get("centerline_px") or r.get("control_points")
        if pts and len(pts) >= 2:
            perps = [(p[0] - cx) * gt_pdx + (p[1] - cy) * gt_pdy for p in pts]
            gt_perps.append(float(np.mean(perps)))
            gt_polylines.append(pts)
        elif "perp_position_px" in r:
            gt_perps.append(r["perp_position_px"])
            gt_polylines.append([])

    # Sort GT by perp position (keep polylines aligned)
    sorted_indices = sorted(range(len(gt_perps)), key=lambda i: gt_perps[i])
    gt_perps = [gt_perps[i] for i in sorted_indices]
    gt_polylines = [gt_polylines[i] for i in sorted_indices]
    n_gt = len(gt_perps)

    if n_gt == 0:
        logger.warning("No GT rows in %s", annotation_path.name)
        return None

    # Load block definition from test_blocks.json to get boundary
    test_blocks = _load_test_blocks()
    # Match by name only (vineyard_name may be empty for anonymized blocks)
    block_def = next((b for b in test_blocks if b["name"] == block_name), None)
    if block_def is None:
        # Fallback: try matching with vineyard too
        block_def = next(
            (b for b in test_blocks
             if b["name"] == block_name and b.get("vineyard_name") == vineyard),
            None,
        )
    if block_def is None:
        logger.error("Block %s not found in test_blocks.json", block_name)
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
    if mpp == 0.0:
        mpp = run_mpp  # use computed mpp if annotation didn't have it

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
            f1_medium=0.0, f1_strict=0.0,
            localization_error_m=0.0, localization_error_px=0.0,
            shape_error_px=0.0, shape_error_m=0.0,
            spacing_error_pct=None, angle_error_deg=None,
            false_positives=0, false_negatives=n_gt, time_s=elapsed,
            is_blind=is_blind,
        )

    # Extract detected perpendicular positions and polylines
    det_angle = result.dominant_angle_deg
    cx_img, cy_img = w / 2.0, h / 2.0
    angle_rad = math.radians(det_angle)
    pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)

    det_perps = []
    det_polylines = []
    for row in result.rows:
        perps = [(x - cx_img) * pdx + (y - cy_img) * pdy for x, y in row.centerline_px]
        det_perps.append(float(np.mean(perps)))
        det_polylines.append(row.centerline_px)

    sorted_det = sorted(range(len(det_perps)), key=lambda i: det_perps[i])
    det_perps = [det_perps[i] for i in sorted_det]
    det_polylines = [det_polylines[i] for i in sorted_det]
    n_det = len(det_perps)

    # Match at loose threshold (primary)
    matched, unmatched_gt, unmatched_det, distances = match_rows(
        gt_perps, det_perps, mpp, THRESHOLD_LOOSE,
    )
    n_matched = len(matched)

    precision = n_matched / n_det if n_det > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # F1 at medium and strict thresholds
    f1_medium = _compute_f1(gt_perps, det_perps, mpp, THRESHOLD_MEDIUM)
    f1_strict = _compute_f1(gt_perps, det_perps, mpp, THRESHOLD_STRICT)

    loc_error_px = float(np.mean(distances)) if distances else 0.0
    loc_error_m = loc_error_px * mpp

    # Shape distance on matched pairs (dense-interpolated polylines)
    shape_dists = []
    for gi, di in matched:
        gt_pl = gt_polylines[gi]
        det_pl = det_polylines[di]
        if gt_pl and len(gt_pl) >= 2 and det_pl and len(det_pl) >= 2:
            shape_dists.append(polyline_shape_distance(gt_pl, det_pl, dense_step=5.0))
    shape_error_px = float(np.mean(shape_dists)) if shape_dists else 0.0
    shape_error_m = shape_error_px * mpp

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
        f1_medium=round(f1_medium, 4),
        f1_strict=round(f1_strict, 4),
        localization_error_m=round(loc_error_m, 4),
        localization_error_px=round(loc_error_px, 2),
        shape_error_px=round(shape_error_px, 2),
        shape_error_m=round(shape_error_m, 4),
        spacing_error_pct=round(spacing_err, 1) if spacing_err is not None else None,
        angle_error_deg=round(angle_err, 2) if angle_err is not None else None,
        false_positives=len(unmatched_det),
        false_negatives=len(unmatched_gt),
        time_s=round(elapsed, 1),
        is_blind=is_blind,
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
    diag = math.sqrt(w ** 2 + h ** 2)

    def _draw_line(perp, angle, color, thickness=1):
        rad = math.radians(angle)
        rdx, rdy = math.cos(rad), math.sin(rad)
        pdx, pdy = -math.sin(rad), math.cos(rad)
        lx, ly = cx + perp * pdx, cy + perp * pdy
        x1, y1 = int(lx - diag * rdx), int(ly - diag * rdy)
        x2, y2 = int(lx + diag * rdx), int(ly + diag * rdy)
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    matched_gt_set = {g for g, d in matched}
    matched_det_set = {d for g, d in matched}

    for i, perp in enumerate(gt_perps):
        if i in matched_gt_set:
            _draw_line(perp, gt_angle, (0, 200, 0), 2)
        else:
            _draw_line(perp, gt_angle, (255, 100, 0), 2)

    for j, perp in enumerate(det_perps):
        if j in matched_det_set:
            _draw_line(perp, det_angle, (255, 0, 255), 1)
        else:
            _draw_line(perp, det_angle, (0, 0, 255), 2)

    for gi, di in matched:
        gt_rad = math.radians(gt_angle)
        det_rad = math.radians(det_angle)
        gt_x = int(cx + gt_perps[gi] * (-math.sin(gt_rad)))
        gt_y = int(cy + gt_perps[gi] * math.cos(gt_rad))
        det_x = int(cx + det_perps[di] * (-math.sin(det_rad)))
        det_y = int(cy + det_perps[di] * math.cos(det_rad))
        cv2.line(canvas, (gt_x, gt_y), (det_x, det_y), (255, 255, 255), 1, cv2.LINE_AA)

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
    """Print formatted results table sorted by F1 ascending (worst first)."""
    results_sorted = sorted(results, key=lambda r: r.f1)

    header = (
        f"{'Block':<12} {'Rows':>9} "
        f"{'F1(.4x)':>8} {'F1(.2x)':>8} {'F1(.1x)':>8} "
        f"{'Loc(m)':>7} {'Shape(m)':>8} "
        f"{'SpErr%':>7} {'AngErr':>7} "
        f"{'FP':>3} {'FN':>3}"
    )
    sep = "-" * 100

    print(f"\n{'GROUND TRUTH EVALUATION':=^100}")
    print(header)
    print(sep)

    for r in results_sorted:
        blind_tag = " [B]" if r.is_blind else ""
        sp_err = f"{r.spacing_error_pct:.1f}" if r.spacing_error_pct is not None else "-"
        a_err = f"{r.angle_error_deg:.1f}" if r.angle_error_deg is not None else "-"
        rows_str = f"{r.n_det}/{r.n_gt}"
        pct = f"({(r.n_det - r.n_gt) / r.n_gt * 100:+.0f}%)" if r.n_gt > 0 else ""
        print(
            f"{r.block + blind_tag:<12} {rows_str + pct:>9} "
            f"{r.f1 * 100:>7.1f}% {r.f1_medium * 100:>7.1f}% {r.f1_strict * 100:>7.1f}% "
            f"{r.localization_error_m:>7.3f} {r.shape_error_m:>8.3f} "
            f"{sp_err:>7} {a_err:>7} "
            f"{r.false_positives:>3} {r.false_negatives:>3}"
        )

    print(f"{'':=^100}")

    if not results:
        return

    # --- Summary ---

    # Unweighted means
    mean_f1 = float(np.mean([r.f1 for r in results]))
    mean_f1_m = float(np.mean([r.f1_medium for r in results]))
    mean_f1_s = float(np.mean([r.f1_strict for r in results]))
    mean_loc = float(np.mean([r.localization_error_m for r in results]))
    mean_shape = float(np.mean([r.shape_error_m for r in results]))

    # Row-weighted means
    total_gt = sum(r.n_gt for r in results)
    if total_gt > 0:
        w_f1 = sum(r.f1 * r.n_gt for r in results) / total_gt
        w_f1_m = sum(r.f1_medium * r.n_gt for r in results) / total_gt
        w_f1_s = sum(r.f1_strict * r.n_gt for r in results) / total_gt
        w_loc = sum(r.localization_error_m * r.n_gt for r in results) / total_gt
        w_shape = sum(r.shape_error_m * r.n_gt for r in results) / total_gt
    else:
        w_f1 = w_f1_m = w_f1_s = w_loc = w_shape = 0.0

    print(f"\n  {'Metric':<22} {'Unweighted':>12} {'Row-weighted':>12}")
    print(f"  {'─' * 48}")
    print(f"  {'F1 (loose 0.4x)':<22} {mean_f1 * 100:>11.1f}% {w_f1 * 100:>11.1f}%")
    print(f"  {'F1 (medium 0.2x)':<22} {mean_f1_m * 100:>11.1f}% {w_f1_m * 100:>11.1f}%")
    print(f"  {'F1 (strict 0.1x)':<22} {mean_f1_s * 100:>11.1f}% {w_f1_s * 100:>11.1f}%")
    print(f"  {'Localization error':<22} {mean_loc:>11.3f}m {w_loc:>11.3f}m")
    print(f"  {'Shape error':<22} {mean_shape:>11.3f}m {w_shape:>11.3f}m")

    # Blind vs pipeline-seeded breakdown
    blind = [r for r in results if r.is_blind]
    seeded = [r for r in results if not r.is_blind]
    if blind and seeded:
        print(f"\n  Blind annotations ({len(blind)} blocks):  F1 = {np.mean([r.f1 for r in blind]) * 100:.1f}%")
        print(f"  Pipeline-seeded ({len(seeded)} blocks):   F1 = {np.mean([r.f1 for r in seeded]) * 100:.1f}%")

    # Worst 3
    worst = results_sorted[:min(3, len(results_sorted))]
    print(f"\n  Worst blocks: {', '.join(f'{r.block} (F1={r.f1 * 100:.0f}%)' for r in worst)}")
    print()


def generate_report(results: list[EvalResult], path: Path):
    """Generate a markdown evaluation report."""
    results_sorted = sorted(results, key=lambda r: r.f1)

    lines = ["# Ground Truth Evaluation Report\n"]
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    lines.append("## Per-Block Results (sorted by F1, worst first)\n")
    lines.append(
        "| Block | Rows | F1 (0.4x) | F1 (0.2x) | F1 (0.1x) | Loc(m) | Shape(m) | FP | FN |"
    )
    lines.append("|-------|-----:|----------:|----------:|----------:|-------:|---------:|---:|---:|")

    for r in results_sorted:
        blind_tag = " [B]" if r.is_blind else ""
        lines.append(
            f"| {r.block}{blind_tag} | {r.n_det}/{r.n_gt} | "
            f"{r.f1 * 100:.1f}% | {r.f1_medium * 100:.1f}% | {r.f1_strict * 100:.1f}% | "
            f"{r.localization_error_m:.3f} | {r.shape_error_m:.3f} | "
            f"{r.false_positives} | {r.false_negatives} |"
        )

    if results:
        total_gt = sum(r.n_gt for r in results)
        lines.append("\n## Summary\n")
        lines.append(f"- Blocks evaluated: {len(results)}")
        lines.append(f"- Total GT rows: {total_gt}")
        lines.append(f"- Mean F1 (loose): {np.mean([r.f1 for r in results]) * 100:.1f}%")
        lines.append(f"- Mean F1 (strict): {np.mean([r.f1_strict for r in results]) * 100:.1f}%")
        lines.append(f"- Mean shape error: {np.mean([r.shape_error_m for r in results]):.3f}m")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved: %s", path)


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

    ann_files = sorted(
        p for p in ANNOTATIONS_DIR.glob("*.json")
        if p.name != "manifest.json"
    )

    if not ann_files:
        print("No annotation files found. Run prepare_dataset.py first.")
        sys.exit(1)

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
        blind = " [blind]" if d.get("metadata", {}).get("blind") else ""
        print(f"  [{i + 1}/{len(ann_files)}] {name}{blind}...", end="", flush=True)

        try:
            r = evaluate_block(f, config)
            if r:
                print(
                    f" F1={r.f1 * 100:.0f}%/{r.f1_medium * 100:.0f}%/{r.f1_strict * 100:.0f}% "
                    f"shape={r.shape_error_m:.3f}m"
                )
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

        # Record to progress tracking
        try:
            from tracking.hooks import build_run_record, build_block_records
            from tracking.storage import append_run, append_block_results

            # Build difficulty and region maps from block registry if available
            difficulty_map = {}
            region_map = {}
            try:
                blocks = _load_test_blocks()
                difficulty_map = {b["name"]: b.get("difficulty_rating") for b in blocks}
                region_map = {b["name"]: b.get("region") for b in blocks}
            except Exception:
                pass

            record = build_run_record(
                run_type="evaluation",
                eval_results=results,
                block_region_map=region_map,
            )
            append_run(record)

            block_records = build_block_records(
                run_id=record["run_id"],
                eval_results=results,
                block_difficulty_map=difficulty_map,
                block_region_map=region_map,
            )
            append_block_results(block_records)
            print(f"  Tracking: recorded run {record['run_id']} ({len(results)} blocks)")
        except Exception as e:
            logger.warning("Failed to record tracking data: %s", e)

    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
