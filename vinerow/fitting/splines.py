"""
Centerline fitting via smoothing splines.

Converts noisy row trajectories (sequences of candidate points) into
smooth geometric centerlines. Straight rows get straight lines; curved
rows get smooth curves. Smoothing controls noise suppression.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.interpolate import UnivariateSpline

from vinerow.acquisition.geo_utils import pixel_to_lnglat
from vinerow.config import PipelineConfig
from vinerow.types import CoarseOrientation, FittedRow, RowTrajectory

logger = logging.getLogger(__name__)


def _fit_single_row(
    trajectory: RowTrajectory,
    coarse: CoarseOrientation,
    mpp: float,
    mask: np.ndarray,
    config: PipelineConfig,
) -> FittedRow | None:
    """Fit a smooth centerline to a single row trajectory."""
    matched = [(c.strip_index, c) for c in trajectory.candidates if c is not None]
    if not matched:
        return None

    angle_rad = math.radians(coarse.angle_deg)
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)

    # Sort by strip index (along-row order)
    matched.sort(key=lambda m: m[0])

    if len(matched) == 1:
        # Single point: create a short line segment
        c = matched[0][1]
        # Extend in both directions by a few spacing widths
        extent = coarse.spacing_px * 3
        p1 = (c.x - row_dx * extent, c.y - row_dy * extent)
        p2 = (c.x + row_dx * extent, c.y + row_dy * extent)
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[p1, p2],
            confidence=round(c.strength, 3),
            length_m=round(2 * extent * mpp, 2),
        )

    if len(matched) == 2:
        # Two points: straight line between them
        c1, c2 = matched[0][1], matched[1][1]
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[(c1.x, c1.y), (c2.x, c2.y)],
            confidence=round((c1.strength + c2.strength) / 2, 3),
            length_m=round(math.hypot(c2.x - c1.x, c2.y - c1.y) * mpp, 2),
        )

    # 3+ points: fit a smoothing spline to the (along_position, perp_position) track
    # Extract along-row and perpendicular coordinates relative to first point
    ref_x, ref_y = matched[0][1].x, matched[0][1].y

    along_positions = []
    perp_positions = []
    strengths = []
    for _, c in matched:
        dx = c.x - ref_x
        dy = c.y - ref_y
        along = dx * row_dx + dy * row_dy
        perp = dx * perp_x + dy * perp_y
        along_positions.append(along)
        perp_positions.append(perp)
        strengths.append(c.strength)

    along_arr = np.array(along_positions)
    perp_arr = np.array(perp_positions)
    strength_arr = np.array(strengths)

    # Ensure along positions are monotonically increasing (required by UnivariateSpline)
    sort_idx = np.argsort(along_arr)
    along_arr = along_arr[sort_idx]
    perp_arr = perp_arr[sort_idx]
    strength_arr = strength_arr[sort_idx]

    # Remove duplicates in along position
    unique_mask = np.diff(along_arr, prepend=-1e9) > 0.5
    along_arr = along_arr[unique_mask]
    perp_arr = perp_arr[unique_mask]
    strength_arr = strength_arr[unique_mask]

    if len(along_arr) < 3:
        # Degenerated after cleanup — straight line
        p1 = (matched[0][1].x, matched[0][1].y)
        p2 = (matched[-1][1].x, matched[-1][1].y)
        conf = float(np.mean(strength_arr)) if len(strength_arr) > 0 else 0.5
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[p1, p2],
            confidence=round(conf, 3),
            length_m=round(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) * mpp, 2),
        )

    # Fit spline with smoothing
    # s = n * (allowed_deviation_px)^2 — smoothing factor
    allowed_dev_px = config.spline_smoothing_m / mpp
    s_factor = len(along_arr) * (allowed_dev_px ** 2)

    try:
        # Use weights proportional to candidate strength
        weights = np.maximum(strength_arr, 0.1)  # don't let weights go to zero
        spline = UnivariateSpline(along_arr, perp_arr, w=weights, s=s_factor, k=3)
    except Exception:
        # Spline fitting failed — fall back to linear
        logger.debug("Spline fitting failed for track %d, using linear", trajectory.track_id)
        p1 = (matched[0][1].x, matched[0][1].y)
        p2 = (matched[-1][1].x, matched[-1][1].y)
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[p1, p2],
            confidence=round(float(np.mean(strength_arr)), 3),
            length_m=round(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) * mpp, 2),
        )

    # Sample the spline at regular intervals
    sample_interval = config.centerline_sample_interval_px
    along_min, along_max = float(along_arr[0]), float(along_arr[-1])
    n_samples = max(2, int((along_max - along_min) / sample_interval) + 1)
    sample_along = np.linspace(along_min, along_max, n_samples)
    sample_perp = spline(sample_along)

    # Clamp spline output to prevent wild extrapolation
    perp_range = float(np.ptp(perp_arr))
    perp_center = float(np.mean(perp_arr))
    max_deviation = max(perp_range * 3.0, coarse.spacing_px)
    sample_perp = np.clip(sample_perp, perp_center - max_deviation, perp_center + max_deviation)

    # Convert back to image coordinates
    h_img, w_img = mask.shape[:2]
    centerline_px = []
    for a, p in zip(sample_along, sample_perp):
        x = ref_x + row_dx * a + perp_x * p
        y = ref_y + row_dy * a + perp_y * p
        # Clamp to image bounds with generous margin
        x = max(-w_img, min(2 * w_img, float(x)))
        y = max(-h_img, min(2 * h_img, float(y)))
        centerline_px.append((round(x, 1), round(y, 1)))

    # Compute curvature: max angular change per meter
    curvature_max = 0.0
    if len(centerline_px) >= 3:
        for i in range(1, len(centerline_px) - 1):
            dx1 = centerline_px[i][0] - centerline_px[i-1][0]
            dy1 = centerline_px[i][1] - centerline_px[i-1][1]
            dx2 = centerline_px[i+1][0] - centerline_px[i][0]
            dy2 = centerline_px[i+1][1] - centerline_px[i][1]
            a1 = math.atan2(dy1, dx1)
            a2 = math.atan2(dy2, dx2)
            angle_change = abs(a2 - a1)
            if angle_change > math.pi:
                angle_change = 2 * math.pi - angle_change
            segment_len = math.hypot(dx1, dy1) * mpp
            if segment_len > 0.1:
                curvature = math.degrees(angle_change) / segment_len
                curvature_max = max(curvature_max, curvature)

    # Length
    total_length_px = 0.0
    for i in range(1, len(centerline_px)):
        total_length_px += math.hypot(
            centerline_px[i][0] - centerline_px[i-1][0],
            centerline_px[i][1] - centerline_px[i-1][1],
        )
    length_m = total_length_px * mpp

    confidence = float(np.mean(strength_arr))
    # Penalize confidence for short tracks (few matched strips)
    # Use total strip count as denominator — a row doesn't need to span
    # the whole block to be valid (block shape may be narrow at ends)
    total_strips = len([c for c in trajectory.candidates])  # total strip slots
    completeness = trajectory.n_matched / max(total_strips * 0.5, 1)
    confidence *= min(completeness, 1.0)

    return FittedRow(
        row_index=trajectory.track_id,
        centerline_px=centerline_px,
        confidence=round(confidence, 3),
        length_m=round(length_m, 2),
        curvature_max_deg_per_m=round(curvature_max, 4),
    )


def fit_centerlines(
    trajectories: list[RowTrajectory],
    coarse: CoarseOrientation,
    mpp: float,
    mask: np.ndarray,
    tile_origin: tuple[int, int],
    zoom: int,
    tile_size: int,
    config: PipelineConfig,
) -> list[FittedRow]:
    """Fit smooth centerlines to all row trajectories.

    Args:
        trajectories: Row trajectories from tracking stage.
        coarse: Coarse orientation estimate.
        mpp: Meters per pixel.
        mask: Binary polygon mask.
        tile_origin: For geographic coordinate conversion.
        zoom: Tile zoom level.
        tile_size: Tile pixel size.
        config: Pipeline configuration.

    Returns:
        List of FittedRow, sorted by perpendicular position (row order).
    """
    fitted: list[FittedRow] = []

    for traj in trajectories:
        row = _fit_single_row(traj, coarse, mpp, mask, config)
        if row is None:
            continue
        fitted.append(row)

    # Sort by the y-intercept-like metric (mean perpendicular position)
    # Use the midpoint of each centerline projected onto the perpendicular axis
    angle_rad = math.radians(coarse.angle_deg)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)

    def _mean_perp(row: FittedRow) -> float:
        if not row.centerline_px:
            return 0.0
        mid_idx = len(row.centerline_px) // 2
        mx, my = row.centerline_px[mid_idx]
        return mx * perp_x + my * perp_y

    fitted.sort(key=_mean_perp)

    # Re-index and compute spacing to previous
    for i, row in enumerate(fitted):
        row.row_index = i
        if i > 0:
            prev = fitted[i - 1]
            # Spacing: perpendicular distance between centerlines at their midpoints
            perp_curr = _mean_perp(row)
            perp_prev = _mean_perp(prev)
            spacing = abs(perp_curr - perp_prev) * mpp
            row.spacing_to_prev_m = round(spacing, 3)

    # Convert to geographic coordinates
    for row in fitted:
        geo_line = []
        for px, py in row.centerline_px:
            lng, lat = pixel_to_lnglat(px, py, tile_origin, zoom, tile_size)
            geo_line.append((round(lng, 7), round(lat, 7)))
        row.centerline_geo = geo_line

    logger.info("Fitted %d centerlines (from %d trajectories)", len(fitted), len(trajectories))

    return fitted
