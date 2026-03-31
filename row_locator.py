"""
Individual row position detection using perpendicular projection.

Takes the 2D FFT's detected angle and average spacing as a prior, then
finds the actual centre-line of every row in the block by:
  1. Building a 1D intensity profile perpendicular to rows.
  2. Laying a regular grid at the FFT spacing.
  3. Refining each grid position to the nearest local extremum.
  4. Strip analysis to detect per-row curvature.
"""

from dataclasses import dataclass, field
import logging
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from image_preprocessor import PreprocessResult
from fft2d_detector import FFT2DResult
from geo_utils import meters_per_pixel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DetectedRow:
    row_index: int
    center_positions: list[tuple[float, float]]  # [(x, y), ...] along row
    mean_perpendicular: float   # average perp position (pixels)
    spacing_to_previous: float | None  # meters, None for first row
    confidence: float           # 0-1 from local peak clarity
    length_px: float            # row extent along-row direction
    max_lateral_deviation_px: float = 0.0  # max distance from straight-line fit
    is_straight: bool = True              # True if deviation < 1 pixel


@dataclass
class RowLocatorResult:
    rows: list[DetectedRow]
    global_angle_deg: float
    mean_spacing_m: float
    spacing_std_m: float
    spacing_range_m: tuple[float, float]
    total_row_count: int
    # Debug data for visualization
    perpendicular_profile: np.ndarray = field(repr=False)
    profile_perp_positions: np.ndarray = field(repr=False)
    peak_perp_positions: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_perpendicular_profile(
    vegetation: np.ndarray,
    mask: np.ndarray,
    angle_rad: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project masked vegetation pixels onto the perpendicular axis.

    Returns (perp_positions, mean_intensity) — one value per 1-pixel bin.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([]), np.array([])

    vals = vegetation[ys, xs].astype(np.float64)

    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    perp_dists = (xs.astype(np.float64) - cx) * perp_x + (ys.astype(np.float64) - cy) * perp_y

    min_d, max_d = float(perp_dists.min()), float(perp_dists.max())
    n_bins = int(max_d - min_d) + 1
    if n_bins < 3:
        return np.array([]), np.array([])

    bin_indices = ((perp_dists - min_d)).astype(int)
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


def _grid_then_refine(
    profile: np.ndarray,
    positions: np.ndarray,
    spacing_px: float,
    mask_counts: np.ndarray | None = None,
    luminance_profile: np.ndarray | None = None,
    force_inverted: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lay a regular grid at expected spacing, then refine each to local extremum.

    Determines whether rows are peaks or valleys by testing both and seeing
    which yields higher aggregate correlation with the grid.

    When ``luminance_profile`` is provided, the phase decision uses grayscale
    brightness instead of ExG — vine canopy is consistently brighter in
    luminance than inter-row grass, whereas inter-row grass is greener (higher
    ExG).  This avoids placing lines on the grass strips.

    When ``force_inverted`` is set, skip scoring and use the given orientation
    (used by Stage 2 strips to reuse the global phase decision).

    Returns (refined_perp_positions, confidences).
    """
    if len(profile) < 3 or spacing_px < 2:
        return np.array([]), np.array([])

    # Smooth lightly to reduce noise
    sigma = max(1.0, spacing_px * 0.12)
    smooth = gaussian_filter1d(profile, sigma=sigma)

    # Smooth luminance profile if provided
    lum_smooth = None
    if luminance_profile is not None and len(luminance_profile) == len(profile):
        lum_smooth = gaussian_filter1d(luminance_profile.astype(np.float64), sigma=sigma)

    # Confidence window: compute local contrast in this radius around each grid point
    conf_radius = max(2, int(spacing_px * 0.3))

    n = len(smooth)
    use_inverted = False

    # Build grids for both orientations
    grids: dict[bool, list[int]] = {}
    for inverted in [False, True]:
        test_profile = -smooth if inverted else smooth
        max_idx = int(np.argmax(test_profile))

        grid: list[int] = []
        pos = float(max_idx)
        while pos >= 0:
            grid.append(int(round(pos)))
            pos -= spacing_px
        pos = float(max_idx) + spacing_px
        while pos < n:
            grid.append(int(round(pos)))
            pos += spacing_px
        grids[inverted] = sorted(set(g for g in grid if 0 <= g < n))

    if force_inverted is not None:
        # Reuse the global phase decision from Stage 1
        use_inverted = force_inverted
    else:
        # --- Primary: peak width (vine rows are narrower than inter-row gaps) ---
        min_dist = max(3, int(spacing_px * 0.5))
        exg_peaks, _ = find_peaks(smooth, distance=min_dist)
        exg_valleys, _ = find_peaks(-smooth, distance=min_dist)

        phase_method = "fallback"
        if len(exg_peaks) >= 2 and len(exg_valleys) >= 2:
            pw = peak_widths(smooth, exg_peaks, rel_height=0.5)[0]
            vw = peak_widths(-smooth, exg_valleys, rel_height=0.5)[0]
            avg_pw = float(np.mean(pw))
            avg_vw = float(np.mean(vw))
            width_ratio = min(avg_pw, avg_vw) / max(avg_pw, avg_vw, 1e-6)

            if width_ratio < 0.90:
                # Widths differ by >10% — narrower features are vine rows
                use_inverted = avg_vw < avg_pw
                phase_method = "width"
                logger.debug(
                    "  grid refine: WIDTH scoring — peak_w=%.1f, valley_w=%.1f, ratio=%.2f → %s",
                    avg_pw, avg_vw, width_ratio,
                    "INVERTED (valleys=rows)" if use_inverted else "NORMAL (peaks=rows)",
                )
            elif lum_smooth is not None:
                # --- Secondary: luminance (vine canopy is brighter) ---
                avg_norm = float(np.mean([lum_smooth[g] for g in grids[False]])) if grids[False] else 0
                avg_inv = float(np.mean([lum_smooth[g] for g in grids[True]])) if grids[True] else 0
                use_inverted = avg_inv > avg_norm
                phase_method = "luminance"
                logger.debug(
                    "  grid refine: LUMINANCE scoring (widths ambiguous %.2f) — normal=%.1f, inverted=%.1f → %s",
                    width_ratio, avg_norm, avg_inv,
                    "INVERTED (valleys=rows)" if use_inverted else "NORMAL (peaks=rows)",
                )
            else:
                # ExG sum fallback
                score_norm = sum(smooth[g] for g in grids[False]) if grids[False] else 0
                score_inv = sum((-smooth)[g] for g in grids[True]) if grids[True] else 0
                use_inverted = score_inv > score_norm
                phase_method = "exg_sum"
        elif lum_smooth is not None:
            avg_norm = float(np.mean([lum_smooth[g] for g in grids[False]])) if grids[False] else 0
            avg_inv = float(np.mean([lum_smooth[g] for g in grids[True]])) if grids[True] else 0
            use_inverted = avg_inv > avg_norm
            phase_method = "luminance"
        else:
            score_norm = sum(smooth[g] for g in grids[False]) if grids[False] else 0
            score_inv = sum((-smooth)[g] for g in grids[True]) if grids[True] else 0
            use_inverted = score_inv > score_norm
            phase_method = "exg_sum"

        if phase_method not in ("width", "luminance"):
            logger.debug("  grid refine: phase by %s → %s",
                         phase_method, "INVERTED" if use_inverted else "NORMAL")

    if use_inverted:
        work_profile = -smooth
        anchor_grid = grids[True]
        logger.debug("  grid refine: using INVERTED profile (valleys=rows)")
    else:
        work_profile = smooth
        anchor_grid = grids[False]
        logger.debug("  grid refine: using NORMAL profile (peaks=rows)")

    # Use grid positions directly (no position refinement — FFT spacing is accurate)
    # Compute confidence from local contrast around each grid point
    final_indices = []
    confidences = []
    profile_range = float(np.max(work_profile) - np.min(work_profile))
    if profile_range < 1e-6:
        profile_range = 1.0

    for g in anchor_grid:
        if g < 0 or g >= n:
            continue
        # Check mask coverage
        if mask_counts is not None and g < len(mask_counts) and mask_counts[g] < 3:
            continue

        # Confidence: local contrast in a window around this grid point
        lo = max(0, g - conf_radius)
        hi = min(n, g + conf_radius + 1)
        local = work_profile[lo:hi]
        if len(local) == 0:
            continue
        val_at_grid = work_profile[g]
        local_min = float(np.min(local))
        conf = (val_at_grid - local_min) / profile_range

        final_indices.append(g)
        confidences.append(max(0.0, min(conf, 1.0)))

    if not final_indices:
        return np.array([]), np.array([])

    perp_positions = positions[np.array(final_indices)]
    return perp_positions, np.array(confidences)


# ---------------------------------------------------------------------------
# Harmonic spacing resolution
# ---------------------------------------------------------------------------


def _resolve_spacing_harmonic(
    profile: np.ndarray,
    spacing_px: float,
    mpp: float,
) -> float:
    """Detect and correct half-period FFT spacing.

    The 2D FFT can return the half-period (row-to-gap distance) instead of the
    full row-to-row distance when the tile source has non-standard tile size
    (e.g. 512 px), causing the FFT's internal plausibility check to reject the
    correct primary peak and adopt a secondary peak at half the period.

    Uses physical plausibility in meters (requires correct mpp from the caller)
    as the primary decision, with 1D spectral analysis as a backup for the
    ambiguous zone.

    Returns the (possibly doubled) spacing in pixels.
    """
    sp_1x_m = spacing_px * mpp
    sp_2x_m = sp_1x_m * 2.0

    # Vineyard row spacing physical limits (matching FFT plausibility range)
    MIN_ROW_SPACING_M = 1.5
    MAX_ROW_SPACING_M = 4.5

    # --- Plausibility guards ---
    if sp_1x_m < MIN_ROW_SPACING_M:
        if sp_2x_m <= MAX_ROW_SPACING_M:
            logger.info(
                "  harmonic: 1x=%.2fm < %.1fm min → DOUBLING to 2x=%.2fm",
                sp_1x_m, MIN_ROW_SPACING_M, sp_2x_m,
            )
            return spacing_px * 2.0
        logger.info("  harmonic: both implausible (1x=%.2fm, 2x=%.2fm), keeping 1x", sp_1x_m, sp_2x_m)
        return spacing_px

    if sp_2x_m > MAX_ROW_SPACING_M:
        logger.info(
            "  harmonic: 2x=%.2fm > %.1fm max → keeping 1x=%.2fm",
            sp_2x_m, MAX_ROW_SPACING_M, sp_1x_m,
        )
        return spacing_px

    # --- Both plausible: 1D spectral analysis as tiebreaker ---
    n = len(profile)
    if spacing_px < 5 or spacing_px * 2 > n / 3:
        logger.info("  harmonic: both plausible (1x=%.2fm, 2x=%.2fm), keeping 1x (profile too short)", sp_1x_m, sp_2x_m)
        return spacing_px

    sigma = max(1.0, spacing_px * 0.12)
    smooth = gaussian_filter1d(profile, sigma=sigma)
    centered = smooth - np.mean(smooth)

    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(n)

    def _peak_power(target_freq: float, search_radius: int = 3) -> float:
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        lo = max(0, idx - search_radius)
        hi = min(len(spectrum), idx + search_radius + 1)
        return float(np.max(spectrum[lo:hi]))

    power_1x = _peak_power(1.0 / spacing_px)
    power_2x = _peak_power(1.0 / (spacing_px * 2.0))
    ratio = power_2x / max(power_1x, 1.0)

    logger.info(
        "  harmonic: both plausible 1x=%.2fm 2x=%.2fm | spectral power 1x=%.0f 2x=%.0f (ratio=%.2f)",
        sp_1x_m, sp_2x_m, power_1x, power_2x, ratio,
    )

    # Conservative threshold: only double when spectral evidence is overwhelming
    SPECTRAL_THRESHOLD = 5.0
    if ratio >= SPECTRAL_THRESHOLD:
        logger.info("  harmonic: DOUBLING (spectral ratio %.1f >= %.1f)", ratio, SPECTRAL_THRESHOLD)
        return spacing_px * 2.0

    logger.info("  harmonic: keeping 1x (spectral ratio %.1f < %.1f)", ratio, SPECTRAL_THRESHOLD)
    return spacing_px


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def locate_rows(
    preprocessed: PreprocessResult,
    fft_result: FFT2DResult,
    lat: float,
    zoom: int,
    n_strips: int = 10,
    tile_size: int = 256,
) -> RowLocatorResult | None:
    """Locate individual row positions using perpendicular projection.

    Args:
        preprocessed: Preprocessed imagery with vegetation index and mask.
        fft_result: 2D FFT detection result (provides angle and spacing).
        lat: Centroid latitude for meter conversion.
        zoom: Tile zoom level.
        n_strips: Number of strips for curvature analysis.
        tile_size: Pixel size of each tile (256 standard, 512 for hi-res).

    Returns:
        RowLocatorResult or None if detection fails.
    """
    vegetation = preprocessed.vegetation
    mask = preprocessed.mask
    h, w = vegetation.shape[:2]
    mpp = meters_per_pixel(lat, zoom, tile_size)

    angle_deg = fft_result.angle_degrees
    spacing_px = fft_result.spacing_pixels
    spacing_m = fft_result.spacing_meters
    angle_rad = math.radians(angle_deg)

    logger.info(
        "locate_rows: angle=%.1f, spacing=%.1fpx (%.2fm), image=%dx%d",
        angle_deg, spacing_px, spacing_m, w, h,
    )

    # Image center (anchor for projections)
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        logger.warning("locate_rows: empty mask")
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # ------------------------------------------------------------------
    # Stage 1: Global perpendicular profile + grid-based row finding
    # ------------------------------------------------------------------
    positions, raw_profile = _build_perpendicular_profile(
        vegetation, mask, angle_rad, cx, cy,
    )

    # Build a luminance profile for phase determination (vine canopy is
    # brighter in grayscale than inter-row grass, even though grass has
    # higher ExG).
    grayscale = preprocessed.grayscale
    _, luminance_profile = _build_perpendicular_profile(
        grayscale, mask, angle_rad, cx, cy,
    )
    if len(raw_profile) < 3:
        logger.warning("locate_rows: perpendicular profile too short")
        return None

    # Also build a pixel-count profile for mask coverage filtering
    ys_all, xs_all = np.where(mask > 0)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    perp_dists_all = (xs_all.astype(np.float64) - cx) * perp_x + (ys_all.astype(np.float64) - cy) * perp_y
    min_d = float(positions[0])
    count_profile = np.zeros(len(positions), dtype=np.float64)
    count_indices = ((perp_dists_all - min_d)).astype(int)
    count_indices = np.clip(count_indices, 0, len(count_profile) - 1)
    np.add.at(count_profile, count_indices, 1.0)

    # Resolve harmonic: check if FFT spacing is half-period
    resolved_spacing_px = _resolve_spacing_harmonic(raw_profile, spacing_px, mpp)
    if resolved_spacing_px != spacing_px:
        logger.info(
            "locate_rows: spacing adjusted %.1fpx (%.2fm) -> %.1fpx (%.2fm)",
            spacing_px, spacing_px * mpp,
            resolved_spacing_px, resolved_spacing_px * mpp,
        )
        spacing_px = resolved_spacing_px
        spacing_m = spacing_px * mpp

    peak_perp_positions, peak_confidences = _grid_then_refine(
        raw_profile, positions, spacing_px, mask_counts=count_profile,
        luminance_profile=luminance_profile,
    )

    # Capture the global phase decision for Stage 2 strips
    # (avoid re-scoring per strip — phase is consistent along the row)
    global_inverted = False
    if len(peak_perp_positions) > 0:
        sigma_check = max(1.0, spacing_px * 0.12)
        smooth_check = gaussian_filter1d(raw_profile, sigma=sigma_check)
        # The grid landed on valleys if the mean ExG at grid positions is
        # lower than the overall mean — i.e. the inverted profile was used
        grid_indices = []
        for pp in peak_perp_positions:
            idx = int(round(pp - float(positions[0])))
            if 0 <= idx < len(smooth_check):
                grid_indices.append(idx)
        if grid_indices:
            mean_at_grid = np.mean([smooth_check[i] for i in grid_indices])
            mean_overall = np.mean(smooth_check)
            global_inverted = mean_at_grid < mean_overall

    if len(peak_perp_positions) < 2:
        logger.warning("locate_rows: found fewer than 2 rows")
        return None

    n_rows = len(peak_perp_positions)
    logger.info("locate_rows: Stage 1 found %d rows from grid refinement", n_rows)

    # ------------------------------------------------------------------
    # Stage 2: Strip analysis for curvature
    # ------------------------------------------------------------------
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)
    along_dists_all = (
        (xs_all.astype(np.float64) - cx) * row_dx
        + (ys_all.astype(np.float64) - cy) * row_dy
    )

    along_min = float(along_dists_all.min())
    along_max = float(along_dists_all.max())
    strip_width = (along_max - along_min) / n_strips

    if strip_width < 10:
        n_strips = 1
        strip_width = along_max - along_min

    # For each strip, detect peaks then match to global rows with smoothing
    strip_peaks_list: list[np.ndarray] = []
    strip_centers: list[float] = []

    for s in range(n_strips):
        strip_lo = along_min + s * strip_width
        strip_hi = strip_lo + strip_width
        strip_center = (strip_lo + strip_hi) / 2.0
        strip_centers.append(strip_center)

        # Build strip mask
        in_strip = (along_dists_all >= strip_lo) & (along_dists_all < strip_hi)
        strip_mask = np.zeros_like(mask)
        strip_mask[ys_all[in_strip], xs_all[in_strip]] = 255

        s_positions, s_raw_profile = _build_perpendicular_profile(
            vegetation, strip_mask, angle_rad, cx, cy,
        )
        if len(s_raw_profile) < 3:
            strip_peaks_list.append(np.array([]))
            continue

        s_peaks, _ = _grid_then_refine(
            s_raw_profile, s_positions, spacing_px,
            force_inverted=global_inverted,
        )
        strip_peaks_list.append(s_peaks)

    # Smoothed tracking: each row tracked independently with gap coasting
    max_jump_px = spacing_px * 0.3
    tracks: list[list[tuple[float, float] | None]] = [
        [None] * n_strips for _ in range(n_rows)
    ]

    for row_idx in range(n_rows):
        last_perp = float(peak_perp_positions[row_idx])

        for s in range(n_strips):
            peaks = strip_peaks_list[s]
            if len(peaks) == 0:
                continue

            diffs = np.abs(peaks - last_perp)
            best = int(np.argmin(diffs))
            if diffs[best] < max_jump_px:
                matched_perp = float(peaks[best])
                tracks[row_idx][s] = (strip_centers[s], matched_perp)
                # Weighted update: 70% matched, 30% previous (smooth tracking)
                last_perp = last_perp * 0.3 + matched_perp * 0.7

    # ------------------------------------------------------------------
    # Stage 3: Assemble row geometries
    # ------------------------------------------------------------------
    detected_rows: list[DetectedRow] = []

    for row_idx in range(n_rows):
        matched = [t for t in tracks[row_idx] if t is not None]

        if len(matched) < 1:
            # Fallback: straight line from global position
            perp_pos = float(peak_perp_positions[row_idx])
            pt1_x = cx + row_dx * along_min + perp_x * perp_pos
            pt1_y = cy + row_dy * along_min + perp_y * perp_pos
            pt2_x = cx + row_dx * along_max + perp_x * perp_pos
            pt2_y = cy + row_dy * along_max + perp_y * perp_pos
            center_positions = [(pt1_x, pt1_y), (pt2_x, pt2_y)]
            length = math.sqrt((pt2_x - pt1_x)**2 + (pt2_y - pt1_y)**2)
        else:
            center_positions = []
            for along, perp in matched:
                px = cx + row_dx * along + perp_x * perp
                py = cy + row_dy * along + perp_y * perp
                center_positions.append((px, py))
            center_positions.sort(key=lambda p: (p[0] - cx) * row_dx + (p[1] - cy) * row_dy)
            if len(center_positions) >= 2:
                p0, p1 = center_positions[0], center_positions[-1]
                length = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
            else:
                length = 0.0

        conf = float(peak_confidences[row_idx]) if row_idx < len(peak_confidences) else 0.5
        mean_perp = float(peak_perp_positions[row_idx])

        # Curvature metric: max deviation from a straight-line fit
        max_dev = 0.0
        is_straight = True
        matched = [t for t in tracks[row_idx] if t is not None]
        if len(matched) >= 3:
            strip_idxs = np.array([strip_centers.index(t[0]) for t in matched])
            perp_vals = np.array([t[1] for t in matched])
            coeffs = np.polyfit(strip_idxs, perp_vals, deg=1)
            fitted = np.polyval(coeffs, strip_idxs)
            deviations = np.abs(perp_vals - fitted)
            max_dev = float(np.max(deviations))
            is_straight = max_dev < 1.0

        detected_rows.append(DetectedRow(
            row_index=row_idx,
            center_positions=center_positions,
            mean_perpendicular=mean_perp,
            spacing_to_previous=None,
            confidence=round(min(conf, 1.0), 3),
            length_px=round(length, 1),
            max_lateral_deviation_px=round(max_dev, 2),
            is_straight=is_straight,
        ))

    # Sort by perpendicular position and compute spacings
    detected_rows.sort(key=lambda r: r.mean_perpendicular)
    spacings_m: list[float] = []
    for i, row in enumerate(detected_rows):
        row.row_index = i
        if i > 0:
            delta_px = row.mean_perpendicular - detected_rows[i - 1].mean_perpendicular
            sp = delta_px * mpp
            row.spacing_to_previous = round(sp, 3)
            spacings_m.append(sp)

    if len(spacings_m) > 0:
        mean_sp = float(np.mean(spacings_m))
        std_sp = float(np.std(spacings_m))
        min_sp = float(np.min(spacings_m))
        max_sp = float(np.max(spacings_m))
    else:
        mean_sp, std_sp = spacing_m, 0.0
        min_sp = max_sp = spacing_m

    logger.info(
        "locate_rows: %d rows, mean_spacing=%.2fm (std=%.3fm), range=[%.2f, %.2f]m",
        len(detected_rows), mean_sp, std_sp, min_sp, max_sp,
    )

    return RowLocatorResult(
        rows=detected_rows,
        global_angle_deg=angle_deg,
        mean_spacing_m=round(mean_sp, 3),
        spacing_std_m=round(std_sp, 4),
        spacing_range_m=(round(min_sp, 3), round(max_sp, 3)),
        total_row_count=len(detected_rows),
        perpendicular_profile=gaussian_filter1d(raw_profile, sigma=max(1.0, spacing_px * 0.12)),
        profile_perp_positions=positions,
        peak_perp_positions=peak_perp_positions,
    )
