"""Statistical metrics for pipeline progress tracking.

Functions:
- bootstrap_confidence_interval: CI for a single metric
- paired_bootstrap_test: paired comparison between two runs
- expected_calibration_error: ECE from per-row confidences
- compute_failure_mode_counts: categorise detection errors
- fit_learning_curve: power-law fit for training set size → F1
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    values: list[float] | np.ndarray,
    n_iterations: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Bootstrap CI for the mean of `values`.

    Returns (lower, upper) tuple, or None if values is empty.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None

    rng = np.random.default_rng()
    means = np.empty(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()

    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def paired_bootstrap_test(
    old_values: list[float] | np.ndarray,
    new_values: list[float] | np.ndarray,
    n_iterations: int = 10000,
) -> dict[str, Any] | None:
    """Paired bootstrap test: are new_values significantly different from old_values?

    Both arrays must be aligned (same blocks, same order). Caller is responsible
    for alignment (e.g., intersection of block IDs evaluated in both runs).

    Returns dict with mean_diff, ci_lower, ci_upper, significant, n_blocks.
    Differences are new - old: positive means improvement.
    Returns None if arrays are empty.
    """
    old = np.asarray(old_values, dtype=np.float64)
    new = np.asarray(new_values, dtype=np.float64)

    if len(old) == 0 or len(new) == 0 or len(old) != len(new):
        return None

    diffs = new - old
    observed_mean = float(diffs.mean())

    rng = np.random.default_rng()
    boot_means = np.empty(n_iterations)
    n = len(diffs)
    for i in range(n_iterations):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    significant = (ci_lower > 0) or (ci_upper < 0)  # CI excludes zero

    return {
        "mean_diff": round(observed_mean, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "significant": significant,
        "n_blocks": int(n),
    }


def expected_calibration_error(
    confidences: list[float] | np.ndarray | None,
    correctness: list[bool] | np.ndarray | None,
    n_bins: int = 10,
) -> float | None:
    """Expected Calibration Error (ECE).

    Bins confidences into n_bins equal-width bins. For each bin, computes the
    gap between mean confidence and accuracy. Returns the weighted mean of
    absolute gaps.

    Returns None if confidences is empty/None.
    """
    if confidences is None or len(confidences) == 0:
        return None

    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)

    if len(conf) != len(corr):
        return None

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    ece = 0.0

    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = mask | (conf == bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = conf[mask].mean()
        avg_accuracy = corr[mask].mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return round(float(ece), 6)


def calibration_bins(
    confidences: list[float] | np.ndarray | None,
    correctness: list[bool] | np.ndarray | None,
    n_bins: int = 10,
) -> list[dict] | None:
    """Return per-bin calibration data for a reliability diagram.

    Each bin has: bin_center, mean_confidence, accuracy, count.
    Returns None if confidences is empty/None.
    """
    if confidences is None or len(confidences) == 0:
        return None

    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)

    if len(conf) != len(corr):
        return None

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (conf == bin_edges[i + 1])

        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            bins.append({
                "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2),
                "mean_confidence": None,
                "accuracy": None,
                "count": 0,
            })
        else:
            bins.append({
                "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2),
                "mean_confidence": round(float(conf[mask].mean()), 4),
                "accuracy": round(float(corr[mask].mean()), 4),
                "count": n_in_bin,
            })

    return bins


def compute_failure_mode_counts(
    matched_pairs: list[tuple[int, int]],
    unmatched_gt: list[int],
    unmatched_det: list[int],
    gt_rows: list,
    det_rows: list,
    matched_distances_m: list[float],
    mpp: float,
    gt_bounding_region=None,
) -> dict[str, int]:
    """Categorise detection errors into failure modes.

    Args:
        matched_pairs: list of (gt_idx, det_idx) tuples
        unmatched_gt: GT row indices with no match
        unmatched_det: detected row indices with no match
        gt_rows: list of GT row polylines (list of (x,y) tuples each)
        det_rows: list of detected row polylines
        matched_distances_m: localization error in metres for each matched pair
        mpp: metres per pixel
        gt_bounding_region: shapely Polygon (convex hull of GT endpoints + buffer).
            If None, phantom_rows is set to 0.

    Returns dict with: false_positives, false_negatives, off_center_matches,
        endpoint_overshoots, phantom_rows
    """
    false_positives = len(unmatched_det)
    false_negatives = len(unmatched_gt)

    # Off-center matches: matched pairs with localization error > 0.5m
    off_center = 0
    for dist_m in matched_distances_m:
        if dist_m > 0.5:
            off_center += 1

    # Endpoint overshoots: detected row length exceeds GT length by >10%
    endpoint_overshoots = 0
    for gt_idx, det_idx in matched_pairs:
        gt_len = _polyline_length(gt_rows[gt_idx])
        det_len = _polyline_length(det_rows[det_idx])
        if gt_len > 0 and det_len > gt_len * 1.1:
            endpoint_overshoots += 1

    # Phantom rows: false positives entirely outside GT bounding region
    phantom_rows = 0
    if gt_bounding_region is not None:
        try:
            from shapely.geometry import LineString
            for det_idx in unmatched_det:
                pts = det_rows[det_idx]
                if len(pts) >= 2:
                    line = LineString(pts)
                    if not line.intersects(gt_bounding_region):
                        phantom_rows += 1
        except ImportError:
            logger.warning("shapely not available — phantom_rows set to 0")

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "off_center_matches": off_center,
        "endpoint_overshoots": endpoint_overshoots,
        "phantom_rows": phantom_rows,
    }


def build_gt_bounding_region(gt_rows: list, median_spacing_m: float, mpp: float):
    """Build the GT bounding region: convex hull of all GT row endpoints,
    buffered by 1x median GT row spacing.

    Args:
        gt_rows: list of row polylines (each a list of (x,y) pixel coords)
        median_spacing_m: median spacing between GT rows in metres
        mpp: metres per pixel

    Returns a shapely Polygon, or None if shapely unavailable or too few points.
    """
    try:
        from shapely.geometry import MultiPoint
    except ImportError:
        logger.warning("shapely not available — cannot build GT bounding region")
        return None

    endpoints = []
    for row in gt_rows:
        if len(row) >= 2:
            endpoints.append(row[0])
            endpoints.append(row[-1])

    if len(endpoints) < 3:
        return None

    hull = MultiPoint(endpoints).convex_hull
    buffer_px = median_spacing_m / max(mpp, 1e-6)
    return hull.buffer(buffer_px)


def fit_learning_curve(
    x_sizes: list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
) -> dict[str, float] | None:
    """Fit y = a - b * x^(-c) to (training set size, F1) data.

    Returns dict with a (asymptote), b, c, and predicted values,
    or None if fit fails or fewer than 4 points.
    """
    x = np.asarray(x_sizes, dtype=np.float64)
    y = np.asarray(y_scores, dtype=np.float64)

    if len(x) < 4:
        return None

    try:
        from scipy.optimize import curve_fit

        def power_law(x, a, b, c):
            return a - b * np.power(x, -c)

        # Initial guess: asymptote near max y, decay from there
        p0 = [max(y) + 0.02, 0.5, 0.5]
        bounds = ([0, 0, 0.01], [1.0, 10.0, 5.0])

        popt, _ = curve_fit(power_law, x, y, p0=p0, bounds=bounds, maxfev=10000)
        a, b, c = popt

        return {
            "a": round(float(a), 4),
            "b": round(float(b), 4),
            "c": round(float(c), 4),
            "asymptote": round(float(a), 4),
        }
    except Exception as e:
        logger.warning("Learning curve fit failed: %s", e)
        return None


def _polyline_length(pts: list) -> float:
    """Compute total Euclidean length of a polyline in pixel coordinates."""
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    return total
