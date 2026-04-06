"""Integration hooks: bridge evaluate_gt.py results into tracking records.

Used by training/train.py, evaluate_gt.py, and gui/routers/detection.py
to create properly structured tracking records from evaluation results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tracking import storage
from tracking.metrics import (
    bootstrap_confidence_interval,
    calibration_bins,
    expected_calibration_error,
)

logger = logging.getLogger(__name__)


def build_run_record(
    run_type: str,
    eval_results: list | None = None,
    config_diff: dict | None = None,
    train_set_size: int | None = None,
    train_block_ids: list[str] | None = None,
    train_blind_count: int | None = None,
    train_pipeline_seeded_count: int | None = None,
    training_time_seconds: float | None = None,
    notes: str | None = None,
    per_row_confidences: list[float] | None = None,
    per_row_correctness: list[bool] | None = None,
    block_region_map: dict[str, str | None] | None = None,
) -> dict:
    """Build a complete run record from evaluation results.

    Args:
        run_type: "training", "evaluation", or "tuning"
        eval_results: list of EvalResult dataclass instances from evaluate_gt.py
        config_diff: dict of changed config keys, each {"old": ..., "new": ...}
        train_set_size: number of training blocks (training runs only)
        train_block_ids: list of block IDs in training set
        train_blind_count: how many training blocks used blind annotation
        train_pipeline_seeded_count: how many used pipeline-seeded annotation
        training_time_seconds: wall clock seconds for training
        notes: optional free-text notes
        per_row_confidences: flat list of all detected row confidences (for ECE)
        per_row_correctness: flat list of whether each row matched GT (for ECE)
    """
    run_id = storage.generate_run_id()
    git_commit, git_dirty = storage.get_git_info()
    now = datetime.now(timezone.utc)

    # Compute aggregate metrics from eval results
    aggregate = None
    bootstrap_ci = None

    if eval_results:
        f1_04 = [r.f1 for r in eval_results]
        f1_02 = [r.f1_medium for r in eval_results]
        f1_01 = [r.f1_strict for r in eval_results]
        loc_errors = [r.localization_error_m for r in eval_results]
        shape_errors = [r.shape_error_m for r in eval_results]

        # Failure mode counts (aggregate across blocks)
        total_fp = sum(r.false_positives for r in eval_results)
        total_fn = sum(r.false_negatives for r in eval_results)
        # off_center_matches and endpoint_overshoots require match-level data
        # which EvalResult doesn't carry. Set to 0 — these are computed
        # per-block in the detailed tracking path.
        failure_counts = {
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "off_center_matches": 0,
            "endpoint_overshoots": 0,
            "phantom_rows": 0,
        }

        worst_idx = int(np.argmin(f1_04))

        ece = expected_calibration_error(per_row_confidences, per_row_correctness)
        bins = calibration_bins(per_row_confidences, per_row_correctness)

        aggregate = {
            "mean_f1_04": round(float(np.mean(f1_04)), 4),
            "mean_f1_02": round(float(np.mean(f1_02)), 4),
            "mean_f1_01": round(float(np.mean(f1_01)), 4),
            "median_f1_04": round(float(np.median(f1_04)), 4),
            "std_f1_04": round(float(np.std(f1_04)), 4),
            "worst_block_f1_04": round(float(f1_04[worst_idx]), 4),
            "worst_block_id": eval_results[worst_idx].block,
            "mean_localization_error_m": round(float(np.mean(loc_errors)), 4),
            "mean_shape_distance_m": round(float(np.mean(shape_errors)), 4),
            "ece": ece,
            "calibration_bins": bins,
            "total_blocks_evaluated": len(eval_results),
            "failure_mode_counts": failure_counts,
        }

        # Bootstrap CIs (10k iterations)
        bootstrap_ci = {
            "mean_f1_04": _ci_to_list(bootstrap_confidence_interval(f1_04)),
            "mean_f1_02": _ci_to_list(bootstrap_confidence_interval(f1_02)),
            "mean_f1_01": _ci_to_list(bootstrap_confidence_interval(f1_01)),
        }

    # Compute per-region metrics
    per_region = None
    if eval_results and block_region_map:
        per_region = _compute_per_region_metrics(eval_results, block_region_map)

    record = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "run_type": run_type,
        "config_diff": config_diff or {},
        "train_set_size": train_set_size,
        "train_block_ids": train_block_ids,
        "train_blind_count": train_blind_count,
        "train_pipeline_seeded_count": train_pipeline_seeded_count,
        "training_time_seconds": round(training_time_seconds, 1) if training_time_seconds else None,
        "notes": notes,
        "aggregate_metrics": aggregate,
        "bootstrap_ci_95": bootstrap_ci,
        "per_region_metrics": per_region,
    }

    return record


def build_block_records(
    run_id: str,
    eval_results: list,
    block_difficulty_map: dict[str, int | None] | None = None,
    block_blind_map: dict[str, bool] | None = None,
    block_region_map: dict[str, str | None] | None = None,
) -> list[dict]:
    """Build per-block result records from EvalResult list.

    Args:
        run_id: the parent run ID
        eval_results: list of EvalResult dataclass instances
        block_difficulty_map: {block_name: difficulty_rating} from block registry
        block_blind_map: {block_name: is_blind} (usually from EvalResult.is_blind)
    """
    if block_difficulty_map is None:
        block_difficulty_map = {}
    if block_blind_map is None:
        block_blind_map = {}
    if block_region_map is None:
        block_region_map = {}

    records = []
    now = datetime.now(timezone.utc)

    for r in eval_results:
        records.append({
            "run_id": run_id,
            "block_id": r.block,
            "timestamp": now.isoformat(),
            "f1_04": r.f1,
            "f1_02": r.f1_medium,
            "f1_01": r.f1_strict,
            "precision": r.precision,
            "recall": r.recall,
            "n_gt_rows": r.n_gt,
            "n_detected_rows": r.n_det,
            "row_count_error": r.n_det - r.n_gt,
            "mean_localization_error_m": r.localization_error_m,
            "shape_distance_m": r.shape_error_m,
            "false_positives": r.false_positives,
            "false_negatives": r.false_negatives,
            "endpoint_overshoots": 0,  # requires match-level data
            "ece": None,  # per-block ECE not computed here
            "is_blind_annotation": r.is_blind or block_blind_map.get(r.block, False),
            "difficulty_rating": block_difficulty_map.get(r.block),
            "region": block_region_map.get(r.block),
        })

    return records


def _compute_per_region_metrics(
    eval_results: list,
    block_region_map: dict[str, str | None],
) -> dict:
    """Group eval results by region and compute per-region aggregate metrics.

    Regions with zero blocks are omitted. No bootstrap CIs — per-region
    sample sizes are too small for them to be meaningful.
    """
    from collections import defaultdict

    by_region: dict[str, list] = defaultdict(list)
    for r in eval_results:
        region = block_region_map.get(r.block)
        if region:
            by_region[region].append(r)

    per_region = {}
    for region, results in by_region.items():
        f1_04 = [r.f1 for r in results]
        f1_02 = [r.f1_medium for r in results]
        f1_01 = [r.f1_strict for r in results]
        worst_idx = int(np.argmin(f1_04))

        per_region[region] = {
            "n_blocks": len(results),
            "mean_f1_04": round(float(np.mean(f1_04)), 4),
            "mean_f1_02": round(float(np.mean(f1_02)), 4),
            "mean_f1_01": round(float(np.mean(f1_01)), 4),
            "worst_block_f1_04": round(float(f1_04[worst_idx]), 4),
            "worst_block_id": results[worst_idx].block,
        }

    return per_region


def _ci_to_list(ci: tuple[float, float] | None) -> list[float] | None:
    """Convert CI tuple to JSON-serializable list."""
    if ci is None:
        return None
    return [round(ci[0], 4), round(ci[1], 4)]
