"""
Global perpendicular profile row detection.

Replaces the strip-based tracker with a simpler, more robust approach:
1. Project the likelihood map onto the perpendicular axis (one global profile)
2. Detect peaks in the profile (one peak = one row)
3. Create straight lines at the FFT-detected angle, clipped to the block mask

Based on methods from Comba et al. (2015), Ronchetti et al. (2020), and
Primicerio et al. (2015) — perpendicular profile analysis with global
periodicity constraint.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from vinerow.config import PipelineConfig
from vinerow.types import CoarseOrientation, FittedRow

logger = logging.getLogger(__name__)


def detect_rows_global_profile(
    likelihood_map: np.ndarray,
    mask: np.ndarray,
    coarse: CoarseOrientation,
    mpp: float,
    config: PipelineConfig,
) -> list[FittedRow]:
    """Detect vine rows using global perpendicular profile peak detection.

    Projects the likelihood map onto the axis perpendicular to the row
    direction, producing a 1D profile where each peak corresponds to one
    row. Rows are straight lines at the FFT-detected angle, clipped to
    the block mask.

    Args:
        likelihood_map: Ridge likelihood (float32, 0-1).
        mask: Block polygon mask (uint8, 0 or 255).
        coarse: Orientation and spacing from FFT.
        mpp: Meters per pixel.
        config: Pipeline config.

    Returns:
        List of FittedRow sorted by perpendicular position.
    """
    h, w = mask.shape[:2]
    angle_rad = math.radians(coarse.angle_deg)
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)

    # Mask centroid as reference point
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        logger.warning("Empty mask")
        return []
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # Project all masked pixels onto perpendicular axis
    ys, xs = np.where(mask > 0)
    perp_dists = (xs.astype(np.float64) - cx) * perp_x + \
                 (ys.astype(np.float64) - cy) * perp_y
    lk_vals = likelihood_map[ys, xs].astype(np.float64)

    # Build 1D profile: mean likelihood at each perpendicular position
    min_d = float(perp_dists.min())
    max_d = float(perp_dists.max())
    n_bins = int(max_d - min_d) + 1
    if n_bins < 3:
        return []

    bin_indices = np.clip((perp_dists - min_d).astype(int), 0, n_bins - 1)
    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(bin_sums, bin_indices, lk_vals)
    np.add.at(bin_counts, bin_indices, 1.0)

    profile = np.where(bin_counts > 0, bin_sums / bin_counts, 0.0)
    positions = np.arange(n_bins, dtype=np.float64) + min_d

    # Smooth profile
    sigma = max(1.0, coarse.spacing_px * 0.1)
    smooth = gaussian_filter1d(profile, sigma=sigma)

    # Detect peaks
    min_distance = max(3, int(coarse.spacing_px * config.peak_min_distance_factor))
    max_val = float(np.max(smooth))
    if max_val < 1e-6:
        return []
    prominence = config.peak_min_prominence * max_val

    peak_indices, _ = find_peaks(
        smooth,
        distance=min_distance,
        prominence=max(prominence, 1e-6),
    )

    logger.info(
        "Global profile: %d bins, %d peaks, spacing=%.1f px, angle=%.1f deg",
        n_bins, len(peak_indices), coarse.spacing_px, coarse.angle_deg,
    )

    # Create FittedRow for each peak
    rows: list[FittedRow] = []
    sample_step = max(3, int(config.centerline_sample_interval_px))

    for row_idx, pidx in enumerate(peak_indices):
        perp_pos = float(positions[pidx])
        strength = float(smooth[pidx]) / max_val

        # Reference point on the row
        ref_px = cx + perp_x * perp_pos
        ref_py = cy + perp_y * perp_pos

        # Sample along the line, clip to mask
        extent = max(h, w) * 1.5
        n_samples = int(extent * 2 / sample_step)
        centerline_segments: list[list[tuple[float, float]]] = []
        current_seg: list[tuple[float, float]] = []

        for t in np.linspace(-extent, extent, n_samples):
            px = ref_px + row_dx * t
            py = ref_py + row_dy * t
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
                current_seg.append((round(px, 1), round(py, 1)))
            else:
                if len(current_seg) >= 2:
                    centerline_segments.append(current_seg)
                current_seg = []
        if len(current_seg) >= 2:
            centerline_segments.append(current_seg)

        if not centerline_segments:
            continue

        # Use the longest segment as the primary centerline
        # (handles mask notches — the longest segment is the main row)
        primary = max(centerline_segments, key=len)

        # Compute length
        length_px = sum(
            math.hypot(primary[i][0] - primary[i-1][0],
                       primary[i][1] - primary[i-1][1])
            for i in range(1, len(primary))
        )
        length_m = length_px * mpp

        rows.append(FittedRow(
            row_index=row_idx,
            centerline_px=primary,
            confidence=round(strength, 3),
            length_m=round(length_m, 2),
            curvature_max_deg_per_m=0.0,  # straight lines have zero curvature
        ))

    # Re-index and compute spacing
    for i, row in enumerate(rows):
        row.row_index = i
        if i > 0:
            # Spacing = perpendicular distance between row perp positions
            # Peak positions are in pixel units on the perp axis
            prev_perp = float(positions[peak_indices[i - 1]])
            curr_perp = float(positions[peak_indices[i]])
            row.spacing_to_prev_m = round(abs(curr_perp - prev_perp) * mpp, 3)

    logger.info("Global profile complete: %d rows fitted", len(rows))
    return rows
