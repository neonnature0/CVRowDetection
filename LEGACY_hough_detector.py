"""
Approach A -- Hough line detection for vineyard row orientation and spacing.

Uses Probabilistic Hough Transform on Canny edges to find line segments,
clusters them by orientation, and derives row spacing from the dominant
cluster's perpendicular projection.
"""

from dataclasses import dataclass
import logging
import math

import cv2
import numpy as np

from image_preprocessor import PreprocessResult
from geo_utils import meters_per_pixel, pixel_spacing_to_meters

logger = logging.getLogger(__name__)


@dataclass
class HoughResult:
    angle_degrees: float  # Dominant row orientation (0-180)
    spacing_meters: float
    spacing_pixels: float
    row_count: int
    confidence: float  # 0-1
    all_lines: np.ndarray | None  # Raw HoughLinesP output
    dominant_lines: list[tuple]  # Lines in dominant cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _line_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    """Compute orientation angle of a line segment, normalised to 0-180 degrees.

    0 = horizontal (E-W), 90 = vertical (N-S).
    """
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    # Normalise into [0, 180)
    angle = angle % 180
    return angle


def _line_length(x1: int, y1: int, x2: int, y2: int) -> float:
    """Euclidean length of a line segment."""
    return math.hypot(x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def detect_lines(
    edges: np.ndarray,
    mask: np.ndarray,
    threshold: int = 50,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> np.ndarray | None:
    """Run HoughLinesP on masked edge image.

    Returns:
        Nx1x4 array of line segments [[x1,y1,x2,y2], ...] or None.
    """
    # Apply mask to edges (should already be masked, but be safe)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    if cv2.countNonZero(masked_edges) == 0:
        logger.warning("detect_lines: no edge pixels after masking")
        return None

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        logger.info("detect_lines: HoughLinesP returned no lines")
        return None

    logger.info("detect_lines: found %d line segments", len(lines))
    return lines


def cluster_by_angle(
    lines: np.ndarray, tolerance_deg: float = 5.0
) -> list[list]:
    """Group lines by orientation angle.

    1. For each line, compute angle (0-180) and length.
    2. Build a 180-bin histogram weighted by line length.
    3. Find the bin with the most total line-length.
    4. Collect all lines within tolerance_deg of that peak.

    Returns:
        List of clusters sorted by total line-length (largest first).
        Each cluster is a list of (x1, y1, x2, y2, angle, length) tuples.
    """
    if lines is None or len(lines) == 0:
        return []

    # Build list of (x1, y1, x2, y2, angle, length)
    line_data: list[tuple] = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle = _line_angle(x1, y1, x2, y2)
        length = _line_length(x1, y1, x2, y2)
        line_data.append((x1, y1, x2, y2, angle, length))

    # Histogram (1-degree bins, weighted by length)
    n_bins = 180
    hist = np.zeros(n_bins, dtype=np.float64)
    for _, _, _, _, angle, length in line_data:
        bin_idx = int(angle) % n_bins
        hist[bin_idx] += length * length  # Squared: strongly favors long lines

    # Find clusters by iteratively extracting the strongest peak
    clusters: list[list] = []
    remaining = list(line_data)
    hist_remaining = hist.copy()

    while len(remaining) > 0:
        peak_bin = int(np.argmax(hist_remaining))
        if hist_remaining[peak_bin] < 1e-6:
            break  # No more significant peaks

        peak_angle = float(peak_bin) + 0.5  # centre of bin

        # Collect lines within tolerance of peak, handling wrap-around at 0/180
        cluster: list[tuple] = []
        still_remaining: list[tuple] = []

        for ld in remaining:
            angle = ld[4]
            # Angular distance handling 0/180 wrap
            diff = abs(angle - peak_angle)
            if diff > 90:
                diff = 180 - diff
            if diff <= tolerance_deg:
                cluster.append(ld)
            else:
                still_remaining.append(ld)

        if cluster:
            clusters.append(cluster)

        # Zero out used bins and update remaining
        for b in range(n_bins):
            b_angle = float(b) + 0.5
            diff = abs(b_angle - peak_angle)
            if diff > 90:
                diff = 180 - diff
            if diff <= tolerance_deg:
                hist_remaining[b] = 0.0

        remaining = still_remaining

    # Sort clusters by total line-length descending
    clusters.sort(key=lambda c: sum(ld[5] ** 2 for ld in c), reverse=True)

    if clusters:
        logger.info(
            "cluster_by_angle: %d clusters, largest has %d lines (total length %.0f px)",
            len(clusters),
            len(clusters[0]),
            sum(ld[5] for ld in clusters[0]),
        )
    else:
        logger.warning("cluster_by_angle: no clusters found")

    return clusters


def compute_spacing(
    lines: list, angle_degrees: float,
    min_spacing_px: float = 5.0,
) -> tuple[float, float]:
    """Compute row spacing from parallel lines.

    1. For each line, compute midpoint.
    2. Project midpoints onto axis perpendicular to dominant angle.
    3. Sort projections, compute consecutive differences.
    4. Filter diffs below min_spacing_px (removes vine canopy edge pairs).
    5. Take median difference as spacing.

    Args:
        lines: List of line tuples (x1, y1, x2, y2, ...).
        angle_degrees: Dominant row angle in degrees.
        min_spacing_px: Minimum valid spacing in pixels. Should be set to
            approximately (min_row_spacing_m / meters_per_pixel * 0.6) to
            exclude within-row canopy edges at high resolution.

    Returns:
        (spacing_pixels, spacing_std_pixels)
    """
    if len(lines) < 2:
        logger.warning("compute_spacing: need at least 2 lines, got %d", len(lines))
        return 0.0, 0.0

    perp_rad = math.radians(angle_degrees + 90.0)
    cos_p = math.cos(perp_rad)
    sin_p = math.sin(perp_rad)

    projections: list[float] = []
    for x1, y1, x2, y2, *_ in lines:
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        proj = mx * cos_p + my * sin_p
        projections.append(proj)

    projections.sort()
    diffs = np.diff(projections)

    if len(diffs) == 0:
        return 0.0, 0.0

    # Filter out small diffs: vine canopy edges at high resolution produce
    # pairs of edges per row (one on each side of the canopy). The gap between
    # these is the canopy width (~1.0-1.5m), not the row spacing (~2.0-3.0m).
    # min_spacing_px should be set to exclude these intra-row diffs.
    diffs_filtered = diffs[diffs > min_spacing_px]
    if len(diffs_filtered) == 0:
        diffs_filtered = diffs  # fall back

    # Second pass: remove outlier-large gaps (> 3x median = missing rows)
    preliminary_median = float(np.median(diffs_filtered))
    if preliminary_median > 0:
        diffs_filtered = diffs_filtered[diffs_filtered < preliminary_median * 3.0]
    if len(diffs_filtered) == 0:
        diffs_filtered = diffs[diffs > min_spacing_px]  # fall back to first filter

    spacing = float(np.median(diffs_filtered))
    spacing_std = float(np.std(diffs_filtered))

    logger.info(
        "compute_spacing: median=%.1f px, std=%.1f px from %d diffs",
        spacing,
        spacing_std,
        len(diffs_filtered),
    )
    return spacing, spacing_std


def estimate_row_count(
    mask: np.ndarray, angle_degrees: float, spacing_pixels: float
) -> int:
    """Estimate row count by measuring polygon width perpendicular to rows.

    Projects all mask-on pixels onto the perpendicular axis, measures the
    range, and divides by spacing.
    """
    if spacing_pixels <= 0:
        return 0

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0

    perp_rad = math.radians(angle_degrees + 90.0)
    cos_p = math.cos(perp_rad)
    sin_p = math.sin(perp_rad)

    projections = xs.astype(np.float64) * cos_p + ys.astype(np.float64) * sin_p
    width = float(projections.max() - projections.min())

    count = max(1, int(round(width / spacing_pixels)) + 1)
    logger.info(
        "estimate_row_count: perp width=%.0f px, spacing=%.1f px -> %d rows",
        width,
        spacing_pixels,
        count,
    )
    return count


def compute_confidence(
    lines_in_cluster: int,
    expected_rows: int,
    angle_std: float,
    spacing_cv: float,
) -> float:
    """Confidence = line_density * angle_consistency * spacing_regularity.

    - line_density = min(lines_in_cluster / expected_rows, 1.0)
    - angle_consistency = max(1 - angle_std / 5.0, 0)
    - spacing_regularity = max(1 - spacing_cv, 0)
    """
    if expected_rows <= 0:
        line_density = 0.0
    else:
        line_density = min(lines_in_cluster / expected_rows, 1.0)

    angle_consistency = max(1.0 - angle_std / 5.0, 0.0)
    spacing_regularity = max(1.0 - spacing_cv, 0.0)

    conf = line_density * angle_consistency * spacing_regularity
    logger.info(
        "compute_confidence: density=%.2f, angle_cons=%.2f, spacing_reg=%.2f -> %.3f",
        line_density,
        angle_consistency,
        spacing_regularity,
        conf,
    )
    return conf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def detect(
    preprocessed: PreprocessResult, lat: float, zoom: int,
    fft_prior: tuple[float, float] | None = None,
    tile_size: int = 256,
) -> HoughResult | None:
    """Full Hough detection pipeline.

    1. detect_lines with default params.
    2. If <5 lines, retry with relaxed params.
    3. cluster_by_angle.
    4. If no dominant cluster, return None.
    5. compute_spacing from dominant cluster.
    6. Convert to meters via pixel_spacing_to_meters.
    7. estimate_row_count.
    8. compute_confidence.

    Returns:
        HoughResult or None if detection fails.
    """
    edges = preprocessed.edges
    mask = preprocessed.mask

    # 1. Scale line detection params with image size (capped to avoid over-filtering)
    h, w = edges.shape[:2]
    min_dim = min(h, w)
    scaled_min_length = max(50, min(100, int(min_dim * 0.05)))
    logger.info("detect: scaled min_line_length=%d (min_dim=%d)", scaled_min_length, min_dim)
    all_lines = detect_lines(edges, mask, threshold=50,
                             min_line_length=scaled_min_length, max_line_gap=10)

    # 2. If too few lines, retry with relaxed params
    if all_lines is None or len(all_lines) < 5:
        logger.info("detect: too few lines (%s), retrying with relaxed params",
                     0 if all_lines is None else len(all_lines))
        all_lines_retry = detect_lines(
            edges, mask, threshold=30,
            min_line_length=max(30, min(70, int(min_dim * 0.035))),
            max_line_gap=15,
        )
        if all_lines_retry is not None:
            all_lines = all_lines_retry

    if all_lines is None or len(all_lines) < 2:
        logger.warning("detect: insufficient lines for analysis")
        return None

    # 3. Cluster by angle
    clusters = cluster_by_angle(all_lines)
    if not clusters:
        logger.warning("detect: no angle clusters found")
        return None

    # If FFT provides a confident angle prior, promote the matching cluster
    if fft_prior is not None and len(clusters) > 1:
        fft_angle, fft_conf = fft_prior
        if fft_conf >= 0.5:
            for ci, cluster in enumerate(clusters):
                c_mean = float(np.mean([ld[4] for ld in cluster]))
                diff = abs(c_mean - fft_angle)
                diff = min(diff, 180.0 - diff)
                if diff <= 15.0 and ci != 0:
                    logger.info(
                        "detect: FFT prior (%.1f deg, conf=%.2f) promotes cluster %d "
                        "(angle diff=%.1f deg)",
                        fft_angle, fft_conf, ci, diff,
                    )
                    clusters[0], clusters[ci] = clusters[ci], clusters[0]
                    break

    dominant = clusters[0]

    # Compute the cluster's mean angle (handle 0/180 wrap with circular mean)
    angles = np.array([ld[4] for ld in dominant])
    # Use circular mean: double angles to handle 0/180 discontinuity
    doubled = np.radians(angles * 2)
    mean_sin = np.mean(np.sin(doubled))
    mean_cos = np.mean(np.cos(doubled))
    mean_angle = math.degrees(math.atan2(mean_sin, mean_cos)) / 2.0
    mean_angle = mean_angle % 180  # normalise
    angle_std = float(np.std(angles))
    # Adjust std for wrap-around: if std is large, the circular std may be smaller
    if angle_std > 45:
        # Recalculate using wrapped angles
        wrapped = np.where(angles > 90, angles - 180, angles)
        angle_std = float(np.std(wrapped))

    logger.info(
        "detect: dominant cluster %d lines, mean angle=%.1f deg, angle_std=%.1f",
        len(dominant),
        mean_angle,
        angle_std,
    )

    # 5. Compute spacing
    # Minimum vine row spacing is ~1.5m. Vine canopy width is ~1.0-1.2m.
    # At high zoom, the Hough detects both edges of each canopy, so the
    # perpendicular diffs include canopy-width diffs (~1.0m) mixed with
    # actual row-spacing diffs (~2.5m). Filter diffs below 1.5m to only
    # keep row-to-row measurements.
    mpp = meters_per_pixel(lat, zoom, tile_size)
    min_spacing_px = max(5.0, 1.5 / mpp)
    logger.info("detect: min_spacing_px=%.1f (mpp=%.3f m/px)", min_spacing_px, mpp)
    spacing_px, spacing_std = compute_spacing(dominant, mean_angle, min_spacing_px=min_spacing_px)
    if spacing_px <= 0:
        logger.warning("detect: could not determine spacing")
        return None

    # 6. Convert to meters
    spacing_m = pixel_spacing_to_meters(spacing_px, lat, zoom)

    # Sanity check: vine row spacing is typically 1.5-4.0 m
    if spacing_m < 0.5:
        logger.warning("detect: spacing %.2f m seems too small", spacing_m)
    elif spacing_m > 10.0:
        logger.warning("detect: spacing %.2f m seems too large", spacing_m)

    # 7. Estimate row count
    row_count = estimate_row_count(mask, mean_angle, spacing_px)

    # 8. Confidence
    spacing_cv = spacing_std / spacing_px if spacing_px > 0 else 1.0
    confidence = compute_confidence(len(dominant), row_count, angle_std, spacing_cv)

    # Format dominant lines as plain tuples
    dominant_tuples = [(int(ld[0]), int(ld[1]), int(ld[2]), int(ld[3])) for ld in dominant]

    result = HoughResult(
        angle_degrees=round(mean_angle, 2),
        spacing_meters=round(spacing_m, 3),
        spacing_pixels=round(spacing_px, 2),
        row_count=row_count,
        confidence=round(confidence, 4),
        all_lines=all_lines,
        dominant_lines=dominant_tuples,
    )
    logger.info(
        "detect: result angle=%.1f deg, spacing=%.2f m (%.1f px), rows=%d, confidence=%.3f",
        result.angle_degrees,
        result.spacing_meters,
        result.spacing_pixels,
        result.row_count,
        result.confidence,
    )
    return result
