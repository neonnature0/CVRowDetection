"""
Local row candidate extraction from the ridge-likelihood map.

Divides the block into overlapping strips along the row direction, projects
the likelihood map onto the perpendicular axis within each strip, and finds
peaks — each peak is a candidate row position. No spacing grid is assumed;
candidates are found purely from the data.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from vinerow.config import PipelineConfig
from vinerow.types import CoarseOrientation, RowCandidate

logger = logging.getLogger(__name__)


def _build_perpendicular_profile(
    image: np.ndarray,
    mask: np.ndarray,
    angle_rad: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project masked pixels onto the perpendicular axis.

    Returns (perp_positions, mean_intensity) — one value per 1-pixel bin.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([]), np.array([])

    vals = image[ys, xs].astype(np.float64)

    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    perp_dists = (xs.astype(np.float64) - cx) * perp_x + (ys.astype(np.float64) - cy) * perp_y

    min_d = float(perp_dists.min())
    max_d = float(perp_dists.max())
    n_bins = int(max_d - min_d) + 1
    if n_bins < 3:
        return np.array([]), np.array([])

    bin_indices = (perp_dists - min_d).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(bin_sums, bin_indices, vals)
    np.add.at(bin_counts, bin_indices, 1.0)

    valid = bin_counts > 0
    profile = np.zeros(n_bins, dtype=np.float64)
    profile[valid] = bin_sums[valid] / bin_counts[valid]

    positions = np.arange(n_bins, dtype=np.float64) + min_d
    return positions, profile


def _profile_to_candidates(
    positions: np.ndarray,
    profile: np.ndarray,
    strip_index: int,
    spacing_px: float,
    cx: float,
    cy: float,
    angle_rad: float,
    strip_along_center: float,
    config: PipelineConfig,
    prominence_override: float | None = None,
    likelihood_map: np.ndarray | None = None,
) -> list[RowCandidate]:
    """Find peak candidates in a 1D perpendicular profile."""
    if len(profile) < 5:
        return []

    # Light smoothing
    sigma = max(1.0, spacing_px * 0.1)
    smooth = gaussian_filter1d(profile, sigma=sigma)

    # Adaptive peak finding: no grid, just find all peaks with minimum separation
    min_distance = max(3, int(spacing_px * config.peak_min_distance_factor))
    prom_factor = prominence_override if prominence_override is not None else config.peak_min_prominence
    prominence = prom_factor * float(np.max(smooth))

    peak_indices, properties = find_peaks(
        smooth,
        distance=min_distance,
        prominence=max(prominence, 1e-6),
    )

    if len(peak_indices) == 0:
        return []

    # Convert to candidates
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)

    candidates = []
    max_strength = float(np.max(smooth))
    if max_strength < 1e-6:
        max_strength = 1.0

    for i, idx in enumerate(peak_indices):
        # Sub-pixel refinement: weighted centroid in a small window around the peak
        half_win = max(2, int(spacing_px * 0.15))
        lo = max(0, idx - half_win)
        hi = min(len(smooth), idx + half_win + 1)
        window_vals = smooth[lo:hi]
        window_pos = positions[lo:hi]
        total_weight = float(np.sum(window_vals))
        if total_weight > 1e-6:
            perp_pos = float(np.sum(window_pos * window_vals) / total_weight)
        else:
            perp_pos = float(positions[idx])
        strength = float(smooth[idx]) / max_strength

        # Convert to image coordinates
        x = cx + row_dx * strip_along_center + perp_x * perp_pos
        y = cy + row_dy * strip_along_center + perp_y * perp_pos

        # Peak half-width from prominences
        hw = 0.0
        if "prominences" in properties and i < len(properties["prominences"]):
            hw = float(properties["prominences"][i]) / max_strength

        # Sample raw likelihood at candidate position
        lk = 0.0
        if likelihood_map is not None:
            ix, iy = int(round(x)), int(round(y))
            lh, lw = likelihood_map.shape[:2]
            if 0 <= ix < lw and 0 <= iy < lh:
                lk = float(likelihood_map[iy, ix])

        candidates.append(RowCandidate(
            x=round(x, 1),
            y=round(y, 1),
            strip_index=strip_index,
            perp_position=round(perp_pos, 2),
            strength=round(strength, 4),
            half_width_px=round(hw, 2),
            likelihood=round(lk, 4),
        ))

    return candidates


def _compute_texture_map(luminance: np.ndarray, window_size: int) -> np.ndarray:
    """Local standard deviation map — high values indicate textured regions (vine canopy)."""
    img = luminance.astype(np.float32)
    win = max(3, window_size | 1)  # odd
    mean = cv2.blur(img, (win, win))
    mean_sq = cv2.blur(img * img, (win, win))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance)


def _check_and_correct_phase(
    candidates: list[RowCandidate],
    luminance: np.ndarray,
    mask: np.ndarray,
    angle_rad: float,
    spacing_px: float,
    cx: float,
    cy: float,
    config: PipelineConfig,
) -> tuple[list[RowCandidate], bool]:
    """Check if candidates land on inter-row (smooth) instead of vine canopy (textured).

    Uses per-strip texture voting: if most strips show midpoints are more
    textured than candidate positions, shift all candidates by half-spacing.

    Returns (candidates, was_corrected).
    """
    h, w = mask.shape[:2]
    window = max(5, int(spacing_px / 3))
    texture = _compute_texture_map(luminance, window)

    perp_dx = -math.sin(angle_rad)
    perp_dy = math.cos(angle_rad)
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)

    # Group candidates by strip
    by_strip: dict[int, list[RowCandidate]] = {}
    for c in candidates:
        by_strip.setdefault(c.strip_index, []).append(c)

    # Compute global texture at ALL candidate positions vs ALL midpoints
    all_cand_tex: list[float] = []
    all_mid_tex: list[float] = []

    for strip_idx, strip_cands in by_strip.items():
        if len(strip_cands) < 3:
            continue

        sorted_cands = sorted(strip_cands, key=lambda c: c.perp_position)
        cand_perps = [c.perp_position for c in sorted_cands]

        # Use the strip's average along position
        strip_along = sum(c.x * row_dx + c.y * row_dy for c in sorted_cands) / len(sorted_cands)
        strip_along -= cx * row_dx + cy * row_dy

        for p in cand_perps:
            x = int(cx + row_dx * strip_along + perp_dx * p)
            y = int(cy + row_dy * strip_along + perp_dy * p)
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                all_cand_tex.append(float(texture[y, x]))

        for i in range(len(cand_perps) - 1):
            mp = (cand_perps[i] + cand_perps[i + 1]) / 2.0
            x = int(cx + row_dx * strip_along + perp_dx * mp)
            y = int(cy + row_dy * strip_along + perp_dy * mp)
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                all_mid_tex.append(float(texture[y, x]))

    if len(all_cand_tex) < 10 or len(all_mid_tex) < 10:
        logger.debug("Phase correction: too few texture samples, skipping")
        return candidates, False

    mean_cand_tex = sum(all_cand_tex) / len(all_cand_tex)
    mean_mid_tex = sum(all_mid_tex) / len(all_mid_tex)
    tex_ratio = mean_mid_tex / max(mean_cand_tex, 1e-6)

    # Also check luminance contrast — if rows vs midpoints have HIGH contrast,
    # the Gabor has strong signal and its phase choice is reliable.
    # Only apply phase correction when contrast is LOW (ambiguous imagery).
    all_cand_lum: list[float] = []
    all_mid_lum: list[float] = []
    lum_f32 = luminance.astype(np.float32)

    for strip_idx, strip_cands in by_strip.items():
        if len(strip_cands) < 3:
            continue
        sorted_cands = sorted(strip_cands, key=lambda c: c.perp_position)
        cand_perps = [c.perp_position for c in sorted_cands]
        strip_along = sum(c.x * row_dx + c.y * row_dy for c in sorted_cands) / len(sorted_cands)
        strip_along -= cx * row_dx + cy * row_dy

        for p in cand_perps:
            x = int(cx + row_dx * strip_along + perp_dx * p)
            y = int(cy + row_dy * strip_along + perp_dy * p)
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                all_cand_lum.append(float(lum_f32[y, x]))

        for i in range(len(cand_perps) - 1):
            mp = (cand_perps[i] + cand_perps[i + 1]) / 2.0
            x = int(cx + row_dx * strip_along + perp_dx * mp)
            y = int(cy + row_dy * strip_along + perp_dy * mp)
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                all_mid_lum.append(float(lum_f32[y, x]))

    mean_cand_lum = sum(all_cand_lum) / len(all_cand_lum) if all_cand_lum else 0
    mean_mid_lum = sum(all_mid_lum) / len(all_mid_lum) if all_mid_lum else 0
    lum_mean = (mean_cand_lum + mean_mid_lum) / 2.0
    lum_contrast = abs(mean_cand_lum - mean_mid_lum) / max(lum_mean, 1.0) * 100

    logger.info(
        "Phase correction: tex_ratio=%.3f  lum_contrast=%.1f%%  (need tex>1.20 AND lum<30%%)",
        tex_ratio, lum_contrast,
    )

    if tex_ratio < 1.20 or lum_contrast >= 30.0:
        return candidates, False

    # Apply shift: move all candidates by +half_spacing in perpendicular direction
    half_spacing = spacing_px / 2.0
    shifted = []
    for c in candidates:
        new_perp = c.perp_position + half_spacing
        new_x = cx + row_dx * (c.x * row_dx + c.y * row_dy - cx * row_dx - cy * row_dy) + perp_dx * new_perp
        new_y = cy + row_dy * (c.x * row_dx + c.y * row_dy - cx * row_dx - cy * row_dy) + perp_dy * new_perp
        ix, iy = int(round(new_x)), int(round(new_y))
        if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
            shifted.append(RowCandidate(
                x=round(new_x, 1),
                y=round(new_y, 1),
                strip_index=c.strip_index,
                perp_position=round(new_perp, 2),
                strength=c.strength,
                half_width_px=c.half_width_px,
            ))

    logger.info(
        "Phase correction: shifted %d candidates by %.1f px, %d survived mask check",
        len(candidates), half_spacing, len(shifted),
    )
    return shifted, True


def extract_candidates(
    likelihood_map: np.ndarray,
    mask: np.ndarray,
    coarse: CoarseOrientation,
    mpp: float,
    config: PipelineConfig,
    luminance: np.ndarray | None = None,
) -> tuple[list[RowCandidate], list[float]]:
    """Extract row candidates from the likelihood map in overlapping strips.

    Args:
        likelihood_map: Row-likelihood map (float32, 0-1).
        mask: Binary mask (uint8, 0 or 255).
        coarse: Coarse orientation estimate.
        mpp: Meters per pixel.
        config: Pipeline configuration.

    Returns:
        Tuple of (all_candidates, strip_along_centers) where strip_along_centers
        is the along-row position of each strip center.
    """
    h, w = mask.shape[:2]
    angle_rad = math.radians(coarse.angle_deg)
    spacing_px = coarse.spacing_px

    # Mask centroid as reference point
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        logger.warning("extract_candidates: empty mask")
        return [], []
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # Compute along-row extent of the block
    ys, xs = np.where(mask > 0)
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)
    along_dists = (xs.astype(np.float64) - cx) * row_dx + (ys.astype(np.float64) - cy) * row_dy
    along_min = float(along_dists.min())
    along_max = float(along_dists.max())
    block_length = along_max - along_min

    # Adaptive strip sizing
    strip_width = config.strip_width_factor * spacing_px
    strip_step = strip_width * (1.0 - config.strip_overlap)

    if strip_step < 5:
        strip_step = block_length
        strip_width = block_length

    n_strips = max(1, int(math.ceil((block_length - strip_width) / strip_step)) + 1)

    logger.info(
        "Candidate extraction: %d strips, width=%.0f px (%.1f m), "
        "overlap=%.0f%%, block_length=%.0f px",
        n_strips, strip_width, strip_width * mpp,
        config.strip_overlap * 100, block_length,
    )

    # Pre-compute perpendicular distances for all masked pixels
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)

    all_candidates: list[RowCandidate] = []
    strip_centers: list[float] = []

    # Edge strip detection: first/last N% of strips get reduced prominence
    n_edge = max(1, int(n_strips * config.edge_strip_fraction))
    edge_indices = set(range(n_edge)) | set(range(n_strips - n_edge, n_strips))

    for s in range(n_strips):
        strip_start = along_min + s * strip_step
        strip_end = strip_start + strip_width
        strip_center = (strip_start + strip_end) / 2.0
        strip_centers.append(strip_center)

        # Build strip mask: pixels within the along-row range
        in_strip = (along_dists >= strip_start) & (along_dists < strip_end)
        strip_mask = np.zeros_like(mask)
        strip_mask[ys[in_strip], xs[in_strip]] = 255

        if cv2.countNonZero(strip_mask) < 10:
            continue

        # Project likelihood map onto perpendicular axis within this strip
        positions, profile = _build_perpendicular_profile(
            likelihood_map, strip_mask, angle_rad, cx, cy,
        )

        if len(profile) < 5:
            continue

        # Adaptive prominence: lower threshold at block edges
        is_edge = s in edge_indices
        prom_override = None
        if is_edge:
            prom_override = config.peak_min_prominence * config.edge_prominence_factor

        # Find candidates in this strip
        strip_candidates = _profile_to_candidates(
            positions, profile, s, spacing_px, cx, cy, angle_rad, strip_center,
            config, prominence_override=prom_override,
            likelihood_map=likelihood_map,
        )

        all_candidates.extend(strip_candidates)

        tag = "EDGE" if is_edge else "core"
        logger.info(
            "  Strip %2d/%d [%s]: %d candidates",
            s + 1, n_strips, tag, len(strip_candidates),
        )

    # Summary stats
    candidates_per_strip = {}
    for c in all_candidates:
        candidates_per_strip.setdefault(c.strip_index, 0)
        candidates_per_strip[c.strip_index] += 1

    if candidates_per_strip:
        mean_per_strip = sum(candidates_per_strip.values()) / len(candidates_per_strip)
    else:
        mean_per_strip = 0

    logger.info(
        "Candidate extraction complete: %d total candidates, %.1f mean per strip",
        len(all_candidates), mean_per_strip,
    )

    # Phase correction: check if candidates are on inter-row instead of vine canopy
    if config.phase_correction_enabled and luminance is not None and len(all_candidates) > 0:
        all_candidates, phase_corrected = _check_and_correct_phase(
            all_candidates, luminance, mask, angle_rad, spacing_px, cx, cy, config,
        )
        if phase_corrected:
            logger.info("Phase correction APPLIED: candidates shifted by half-spacing")

    return all_candidates, strip_centers
