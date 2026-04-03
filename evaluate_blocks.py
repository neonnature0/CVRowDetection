#!/usr/bin/env python3
"""
Evaluate block detection predictions against ground-truth annotations.

Uses IoU (Intersection over Union) between predicted and ground-truth
block polygons with Hungarian assignment for optimal matching.

Usage:
    python evaluate_blocks.py --predictions output/predictions.geojson --ground-truth dataset/standalone/site.geojson
    python evaluate_blocks.py --checkpoint-dir detection/checkpoints --annotations dataset/standalone/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def polygon_iou(poly_a: ShapelyPolygon, poly_b: ShapelyPolygon) -> float:
    """Compute IoU between two shapely polygons."""
    if not poly_a.is_valid:
        poly_a = make_valid(poly_a)
    if not poly_b.is_valid:
        poly_b = make_valid(poly_b)
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union < 1e-10:
        return 0.0
    return intersection / union


def hausdorff_distance(poly_a: ShapelyPolygon, poly_b: ShapelyPolygon) -> float:
    """Compute Hausdorff distance between polygon boundaries."""
    if not poly_a.is_valid:
        poly_a = make_valid(poly_a)
    if not poly_b.is_valid:
        poly_b = make_valid(poly_b)
    return poly_a.hausdorff_distance(poly_b)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_polygons(
    predicted: list[ShapelyPolygon],
    ground_truth: list[ShapelyPolygon],
    iou_threshold: float = 0.5,
) -> dict:
    """Match predicted to ground-truth polygons using Hungarian algorithm.

    Returns:
        {
            'matches': [(pred_idx, gt_idx, iou), ...],
            'false_positives': [pred_idx, ...],
            'false_negatives': [gt_idx, ...],
        }
    """
    n_pred = len(predicted)
    n_gt = len(ground_truth)

    if n_pred == 0 and n_gt == 0:
        return {"matches": [], "false_positives": [], "false_negatives": []}

    if n_pred == 0:
        return {"matches": [], "false_positives": [], "false_negatives": list(range(n_gt))}

    if n_gt == 0:
        return {"matches": [], "false_positives": list(range(n_pred)), "false_negatives": []}

    # Build IoU cost matrix (we minimize cost, so use 1 - IoU)
    cost_matrix = np.ones((n_pred, n_gt), dtype=np.float64)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

    for i, pred in enumerate(predicted):
        for j, gt in enumerate(ground_truth):
            iou = polygon_iou(pred, gt)
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = 1.0 - iou

    # Hungarian assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_pred = set()
    matched_gt = set()

    for pred_idx, gt_idx in zip(row_indices, col_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            matches.append((int(pred_idx), int(gt_idx), float(iou)))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

    false_positives = [i for i in range(n_pred) if i not in matched_pred]
    false_negatives = [j for j in range(n_gt) if j not in matched_gt]

    return {
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    matching: dict,
    predicted: list[ShapelyPolygon],
    ground_truth: list[ShapelyPolygon],
) -> dict:
    """Compute aggregate metrics from matching results."""
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    matches = matching["matches"]
    n_matches = len(matches)

    precision = n_matches / n_pred if n_pred > 0 else 0.0
    recall = n_matches / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Mean IoU of matched pairs
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

    # Hausdorff distances of matched pairs
    hausdorff_dists = []
    for pred_idx, gt_idx, _ in matches:
        hd = hausdorff_distance(predicted[pred_idx], ground_truth[gt_idx])
        hausdorff_dists.append(hd)
    mean_hausdorff = float(np.mean(hausdorff_dists)) if hausdorff_dists else 0.0

    # Area ratios of matched pairs
    area_ratios = []
    for pred_idx, gt_idx, _ in matches:
        pred_area = predicted[pred_idx].area
        gt_area = ground_truth[gt_idx].area
        if gt_area > 0:
            area_ratios.append(pred_area / gt_area)
    mean_area_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0

    return {
        "n_predicted": n_pred,
        "n_ground_truth": n_gt,
        "n_matched": n_matches,
        "n_false_positives": len(matching["false_positives"]),
        "n_false_negatives": len(matching["false_negatives"]),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_hausdorff_px": round(mean_hausdorff, 2),
        "mean_area_ratio": round(mean_area_ratio, 4),
        "polygon_count_error": abs(n_pred - n_gt),
        "per_match_iou": [round(m[2], 4) for m in matches],
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_overlay(
    image_bgr: np.ndarray,
    predicted: list[ShapelyPolygon],
    ground_truth: list[ShapelyPolygon],
    matching: dict,
    output_path: Path,
):
    """Generate overlay visualization of predictions vs ground truth."""
    overlay = image_bgr.copy()

    # Draw ground truth in green
    for j, gt in enumerate(ground_truth):
        coords = np.array(gt.exterior.coords, dtype=np.int32)
        color = (0, 255, 0)  # green
        if j in matching["false_negatives"]:
            color = (255, 100, 0)  # blue — false negative (missed)
        cv2.polylines(overlay, [coords], True, color, 2)

    # Draw predictions in magenta
    for i, pred in enumerate(predicted):
        coords = np.array(pred.exterior.coords, dtype=np.int32)
        color = (255, 0, 255)  # magenta
        if i in matching["false_positives"]:
            color = (0, 0, 255)  # red — false positive
        cv2.polylines(overlay, [coords], True, color, 2)

    # Draw matched pairs — yellow overlap
    for pred_idx, gt_idx, iou in matching["matches"]:
        pred = predicted[pred_idx]
        gt = ground_truth[gt_idx]
        intersection = pred.intersection(gt)
        if not intersection.is_empty and intersection.geom_type == "Polygon":
            coords = np.array(intersection.exterior.coords, dtype=np.int32)
            cv2.fillPoly(overlay, [coords], (0, 255, 255))  # yellow fill

    # Blend overlay
    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)

    cv2.imwrite(str(output_path), blended)
    logger.info("Saved overlay to %s", output_path)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_polygons_from_geojson(path: Path) -> list[ShapelyPolygon]:
    """Load block polygons from a GeoJSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})

        # Accept both block annotations and prediction outputs
        is_block = props.get("feature_type") == "block" or "block_id" in props
        if geom.get("type") == "Polygon" and is_block:
            ring = geom["coordinates"][0]
            if len(ring) >= 3:
                poly = ShapelyPolygon([(c[0], c[1]) for c in ring])
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.area > 0:
                    polygons.append(poly)

    return polygons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate block detection")
    parser.add_argument("--predictions", type=str, required=True, help="Predicted blocks GeoJSON")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ground truth blocks GeoJSON")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--output-dir", type=str, default="output/block_evaluation", help="Output directory")
    parser.add_argument("--image", type=str, default=None, help="Property image for overlay visualization")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load polygons
    predicted = load_polygons_from_geojson(Path(args.predictions))
    ground_truth = load_polygons_from_geojson(Path(args.ground_truth))

    print(f"Predicted blocks: {len(predicted)}")
    print(f"Ground truth blocks: {len(ground_truth)}")

    # Match and compute metrics
    matching = match_polygons(predicted, ground_truth, args.iou_threshold)
    metrics = compute_metrics(matching, predicted, ground_truth)

    # Print results
    print(f"\n{'='*50}")
    print(f"Block Detection Evaluation (IoU threshold: {args.iou_threshold})")
    print(f"{'='*50}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1 Score:            {metrics['f1']:.4f}")
    print(f"  Mean IoU:            {metrics['mean_iou']:.4f}")
    print(f"  Mean Hausdorff (px): {metrics['mean_hausdorff_px']:.2f}")
    print(f"  Mean Area Ratio:     {metrics['mean_area_ratio']:.4f}")
    print(f"  Polygon Count Error: {metrics['polygon_count_error']}")
    print(f"  Matched:             {metrics['n_matched']}")
    print(f"  False Positives:     {metrics['n_false_positives']}")
    print(f"  False Negatives:     {metrics['n_false_negatives']}")

    if metrics["per_match_iou"]:
        print(f"\n  Per-match IoU: {metrics['per_match_iou']}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate overlay if image provided
    if args.image:
        image = cv2.imread(args.image)
        if image is not None:
            overlay_path = output_dir / "overlay.png"
            generate_overlay(image, predicted, ground_truth, matching, overlay_path)


if __name__ == "__main__":
    main()
