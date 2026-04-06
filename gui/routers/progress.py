"""Progress tracking endpoints — runs, comparisons, learning curves."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from tracking import storage
from tracking.metrics import (
    bootstrap_confidence_interval,
    calibration_bins,
    fit_learning_curve,
    paired_bootstrap_test,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/runs")
def list_runs():
    """Return all runs, newest first."""
    return storage.load_runs()


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """Return a single run with full details."""
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run


@router.get("/block-trajectory/{block_id}")
def block_trajectory(block_id: str):
    """Return all per-block results for one block across all runs, chronologically."""
    all_results = storage.load_block_results()
    trajectory = [r for r in all_results if r.get("block_id") == block_id]
    trajectory.sort(key=lambda r: r.get("timestamp", ""))
    return trajectory


@router.get("/learning-curve")
def learning_curve():
    """Compute learning curve from historical training runs.

    Returns raw points (train_set_size, mean_f1_04) plus fitted curve
    parameters if at least 4 training runs exist.
    """
    runs = storage.load_runs()
    training_runs = [
        r for r in runs
        if r.get("run_type") == "training"
        and r.get("aggregate_metrics")
        and r.get("train_set_size")
    ]

    # Build data points: one per training run
    points = []
    for r in training_runs:
        metrics = r["aggregate_metrics"]
        points.append({
            "run_id": r["run_id"],
            "train_set_size": r["train_set_size"],
            "mean_f1_04": metrics.get("mean_f1_04"),
            "mean_f1_02": metrics.get("mean_f1_02"),
            "mean_f1_01": metrics.get("mean_f1_01"),
            "timestamp": r.get("timestamp"),
        })

    # Sort by training set size
    points.sort(key=lambda p: p["train_set_size"])

    # Fit power-law curve if enough data
    fit = None
    if len(points) >= 4:
        sizes = [p["train_set_size"] for p in points]
        scores = [p["mean_f1_04"] for p in points if p["mean_f1_04"] is not None]
        if len(scores) >= 4:
            fit = fit_learning_curve(sizes[:len(scores)], scores)

    return {
        "points": points,
        "n_training_runs": len(points),
        "fit": fit,
    }


@router.get("/compare/{old_run_id}/{new_run_id}")
def compare_runs(old_run_id: str, new_run_id: str):
    """Paired bootstrap comparison between two runs.

    Uses intersection of block IDs evaluated in both runs.
    """
    all_results = storage.load_block_results()

    old_blocks = {r["block_id"]: r for r in all_results if r.get("run_id") == old_run_id}
    new_blocks = {r["block_id"]: r for r in all_results if r.get("run_id") == new_run_id}

    if not old_blocks:
        raise HTTPException(404, f"No block results found for run '{old_run_id}'")
    if not new_blocks:
        raise HTTPException(404, f"No block results found for run '{new_run_id}'")

    # Intersection of block IDs
    common_ids = sorted(set(old_blocks.keys()) & set(new_blocks.keys()))
    if not common_ids:
        raise HTTPException(
            400,
            f"No common blocks between runs. "
            f"Old has {len(old_blocks)} blocks, new has {len(new_blocks)} blocks, "
            f"but their intersection is empty.",
        )

    # Aligned arrays
    old_f1 = [old_blocks[bid]["f1_04"] for bid in common_ids]
    new_f1 = [new_blocks[bid]["f1_04"] for bid in common_ids]

    paired_result = paired_bootstrap_test(old_f1, new_f1)

    # Per-block deltas
    per_block_deltas = []
    for bid in common_ids:
        old_r = old_blocks[bid]
        new_r = new_blocks[bid]
        delta = (new_r.get("f1_04", 0) or 0) - (old_r.get("f1_04", 0) or 0)
        per_block_deltas.append({
            "block_id": bid,
            "old_f1_04": old_r.get("f1_04"),
            "new_f1_04": new_r.get("f1_04"),
            "delta": round(delta, 4),
            "difficulty_rating": new_r.get("difficulty_rating"),
        })

    # Sort by absolute delta (biggest changes first)
    per_block_deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)

    return {
        "old_run_id": old_run_id,
        "new_run_id": new_run_id,
        "n_blocks_compared": len(common_ids),
        "paired_test": paired_result,
        "per_block_deltas": per_block_deltas,
    }


@router.get("/calibration/{run_id}")
def calibration_data(run_id: str):
    """Return binned calibration data for a reliability diagram.

    If per-row confidence data is not available for this run, returns
    {"available": false, "reason": "..."}.
    """
    # Per-row confidence data is not stored in the tracking files —
    # it would need to be computed live by re-running evaluation.
    # For now, check if the run has ECE computed (which implies confidence data existed).
    run = storage.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")

    metrics = run.get("aggregate_metrics") or {}
    ece = metrics.get("ece")

    if ece is None:
        return {
            "available": False,
            "reason": "Per-row confidence scores were not available when this run was recorded.",
        }

    # Calibration bins are not stored per-run — return a placeholder
    # that indicates ECE was computed but bins aren't available for historical runs.
    # Live calibration would require re-running the pipeline.
    return {
        "available": False,
        "reason": "Calibration bin data is not stored for historical runs. ECE was computed as aggregate only.",
        "ece": ece,
    }
