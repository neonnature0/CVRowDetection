"""
Approach C -- 2D FFT-based vineyard row detection.

Computes a single 2D FFT on the masked image. Periodic parallel rows
produce a conjugate pair of peaks in the magnitude spectrum whose polar
coordinates directly encode row angle and spacing. Orders of magnitude
faster than the angular-sweep 1D FFT approach.

References:
    Delenne et al. 2006 — vineyard row detection via 2D FFT
    Rabatel et al. 2008 — crop row orientation from frequency domain
"""

from dataclasses import dataclass
import logging
import math

import cv2
import numpy as np
from scipy.signal import find_peaks as _find_peaks

from image_preprocessor import PreprocessResult
from geo_utils import meters_per_pixel, pixel_spacing_to_meters

logger = logging.getLogger(__name__)


@dataclass
class FFT2DResult:
    angle_degrees: float        # Row angle in image coordinates (0-180)
    spacing_meters: float
    spacing_pixels: float
    row_count: int
    confidence: float           # 0-1
    peak_position: tuple        # (x, y) in the shifted magnitude spectrum
    magnitude_spectrum: np.ndarray  # Log-magnitude for debug visualization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subpixel_peak(mag: np.ndarray, py: int, px: int) -> tuple[float, float]:
    """Refine peak location to sub-pixel precision via parabolic fit.

    Fits a 1D parabola along each axis independently using the 3-point
    neighbourhood around (py, px). Falls back to integer coords if the
    neighbourhood is at the array edge.
    """
    h, w = mag.shape

    # X refinement
    if 0 < px < w - 1:
        left = float(mag[py, px - 1])
        center = float(mag[py, px])
        right = float(mag[py, px + 1])
        denom = 2.0 * (2.0 * center - left - right)
        dx = (right - left) / denom if abs(denom) > 1e-12 else 0.0
    else:
        dx = 0.0

    # Y refinement
    if 0 < py < h - 1:
        top = float(mag[py - 1, px])
        center = float(mag[py, px])
        bottom = float(mag[py + 1, px])
        denom = 2.0 * (2.0 * center - top - bottom)
        dy = (bottom - top) / denom if abs(denom) > 1e-12 else 0.0
    else:
        dy = 0.0

    return float(py) + dy, float(px) + dx


# Plausible vine-row spacing range (meters)
PLAUSIBLE_MIN_M = 1.2
PLAUSIBLE_MAX_M = 4.5


def _extract_radial_profile(
    mag: np.ndarray,
    cy: int,
    cx: int,
    dx_unit: float,
    dy_unit: float,
    r_min: int,
    r_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a 1D magnitude profile along a radial direction.

    Samples the magnitude spectrum along the line from center outward in the
    direction (dx_unit, dy_unit) using bilinear interpolation.

    Returns (radii, values) arrays.
    """
    h, w = mag.shape
    radii = np.arange(r_min, r_max + 1, dtype=np.float64)
    values = np.zeros_like(radii)

    for i, r in enumerate(radii):
        sx = cx + r * dx_unit
        sy = cy + r * dy_unit

        x0, y0 = int(math.floor(sx)), int(math.floor(sy))
        x1, y1 = x0 + 1, y0 + 1

        if 0 <= x0 < w - 1 and 0 <= y0 < h - 1:
            fx, fy = sx - x0, sy - y0
            values[i] = (
                mag[y0, x0] * (1 - fx) * (1 - fy)
                + mag[y0, x1] * fx * (1 - fy)
                + mag[y1, x0] * (1 - fx) * fy
                + mag[y1, x1] * fx * fy
            )

    return radii, values


def _best_plausible_spacing(
    mag: np.ndarray,
    cy: int,
    cx: int,
    peak_x: float,
    peak_y: float,
    pad_h: int,
    pad_w: int,
    mpp: float,
    min_freq_radius: int,
    max_freq_radius: int,
) -> tuple[float, float] | None:
    """Find the strongest peak whose spacing falls in the plausible range.

    Extracts a 1D radial profile from the 2D magnitude spectrum along the
    direction of the dominant peak, finds all local maxima, and returns the
    strongest one whose spacing in meters is within [PLAUSIBLE_MIN_M,
    PLAUSIBLE_MAX_M].

    Returns (spacing_px, spacing_m) or None.
    """
    dx = peak_x - cx
    ky = peak_y - cy
    peak_dist = math.sqrt(dx**2 + ky**2)
    if peak_dist < 1e-6:
        return None

    dx_unit = dx / peak_dist
    dy_unit = ky / peak_dist

    radii, profile = _extract_radial_profile(
        mag, cy, cx, dx_unit, dy_unit, min_freq_radius, max_freq_radius,
    )

    if len(profile) == 0 or np.max(profile) < 1e-6:
        return None

    peak_indices, _ = _find_peaks(
        profile, prominence=0.05 * np.max(profile), distance=3,
    )

    if len(peak_indices) == 0:
        return None

    best_spacing_px: float | None = None
    best_spacing_m: float | None = None
    best_mag = -1.0

    for idx in peak_indices:
        r = radii[idx]
        freq_x = (r * dx_unit) / pad_w
        freq_y = (r * dy_unit) / pad_h
        freq_mag = math.sqrt(freq_x**2 + freq_y**2)
        if freq_mag < 1e-12:
            continue
        sp_px = 1.0 / freq_mag
        sp_m = sp_px * mpp

        if PLAUSIBLE_MIN_M <= sp_m <= PLAUSIBLE_MAX_M and profile[idx] > best_mag:
            best_mag = float(profile[idx])
            best_spacing_px = sp_px
            best_spacing_m = sp_m
            logger.debug(
                "  radial candidate: r=%.1f, spacing=%.2fm, mag=%.1f",
                r, sp_m, profile[idx],
            )

    if best_spacing_px is not None and best_spacing_m is not None:
        return best_spacing_px, best_spacing_m
    return None


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


def detect(
    preprocessed: PreprocessResult, lat: float, zoom: int, tile_size: int = 256,
) -> FFT2DResult | None:
    """Full 2D FFT detection pipeline.

    1. Prepare masked image, subtract mean, apply 2D Hanning window.
    2. Compute 2D FFT, take shifted magnitude spectrum.
    3. Band-pass filter (suppress DC/low-freq and high-freq noise).
    4. Find dominant peak in top half of spectrum (conjugate symmetry).
    5. Convert peak polar coordinates to row angle and spacing.
    6. Harmonic correction (check for 2x frequency peak).
    7. Estimate row count from mask width perpendicular to rows.
    8. Compute confidence from peak SNR.

    Returns:
        FFT2DResult or None if no clear signal.
    """
    # 1. Select input channel
    if preprocessed.use_vegetation:
        image = preprocessed.vegetation
        logger.info("detect (2D FFT): using vegetation index")
    else:
        image = preprocessed.enhanced
        logger.info("detect (2D FFT): using enhanced grayscale")

    mask = preprocessed.mask
    if cv2.countNonZero(mask) == 0:
        logger.warning("detect (2D FFT): mask is empty")
        return None

    h, w = image.shape[:2]
    mpp = meters_per_pixel(lat, zoom, tile_size)

    # 2. Mask, subtract mean, window
    masked_img = image.astype(np.float64) * (mask > 0).astype(np.float64)
    valid_pixels = masked_img[mask > 0]
    if len(valid_pixels) == 0:
        return None
    masked_img[mask > 0] -= np.mean(valid_pixels)

    # 2D Hanning window to reduce spectral leakage from image edges
    window_2d = np.outer(np.hanning(h), np.hanning(w))
    windowed = masked_img * window_2d

    # 3. Compute 2D FFT (zero-pad to 2x for finer frequency resolution)
    # Cap padding to avoid OOM on large images (~1.3GB complex128 limit)
    max_padded_pixels = 80_000_000
    pad_factor = 2 if h * w * 4 <= max_padded_pixels else 1
    if pad_factor == 1:
        logger.info("detect (2D FFT): skipping zero-pad (image too large: %dx%d)", w, h)
    pad_h, pad_w = h * pad_factor, w * pad_factor
    fft_result = np.fft.fft2(windowed.astype(np.float32), s=(pad_h, pad_w))
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)

    cy, cx = pad_h // 2, pad_w // 2

    # 4. Band-pass filter
    # Low-frequency suppression: max plausible vine row spacing
    max_spacing_m = 8.0
    max_spacing_px = max_spacing_m / mpp
    # Frequency radius for a given spacing: dim / spacing_px
    # Use min padded dimension for conservative estimate
    min_dim = min(pad_h, pad_w)
    min_freq_radius = max(5, int(min_dim / max_spacing_px))

    # High-frequency suppression: min plausible vine row spacing
    min_spacing_m = 1.0
    min_spacing_px = min_spacing_m / mpp
    max_freq_radius = int(min_dim / min_spacing_px)

    logger.info(
        "detect (2D FFT): image %dx%d, mpp=%.3f, band-pass radii=[%d, %d]",
        w, h, mpp, min_freq_radius, max_freq_radius,
    )

    # Build distance-from-center map (padded dimensions)
    Y, X = np.ogrid[:pad_h, :pad_w]
    dist_from_center = np.sqrt(
        (X.astype(np.float64) - cx) ** 2 + (Y.astype(np.float64) - cy) ** 2
    )

    # Apply band-pass
    magnitude[dist_from_center <= min_freq_radius] = 0.0
    magnitude[dist_from_center >= max_freq_radius] = 0.0

    # Save log-magnitude for debug visualization (before peak search modifies it)
    log_magnitude = np.log1p(magnitude).copy()

    # 5. Find peak in TOP HALF only (conjugate symmetry)
    search_region = magnitude[:cy, :].copy()
    if np.max(search_region) < 1e-6:
        logger.warning("detect (2D FFT): no significant peak in band-pass region")
        return None

    peak_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
    peak_y_int, peak_x_int = int(peak_idx[0]), int(peak_idx[1])
    peak_mag = float(search_region[peak_y_int, peak_x_int])

    # Sub-pixel refinement
    peak_y, peak_x = _subpixel_peak(search_region, peak_y_int, peak_x_int)

    logger.info(
        "detect (2D FFT): peak at (%.1f, %.1f) magnitude=%.1f",
        peak_x, peak_y, peak_mag,
    )

    # 6. Convert peak to angle and spacing
    dx = peak_x - cx           # horizontal offset in FFT array
    ky = peak_y - cy           # vertical offset — NO y-flip (FFT row axis = image row axis)

    # Row angle directly from FFT peak position.
    # For rows at angle θ in pixel space, the FFT peak appears at
    # (kx, ky) = (cos(θ+90°)·w/S, sin(θ+90°)·h/S) where S is the spacing.
    # Due to conjugate symmetry we may find the actual or conjugate peak in
    # the top half.  The formula θ = atan2(-dx/pad_w, ky/pad_h) handles both:
    row_angle = math.degrees(math.atan2(-dx / pad_w, ky / pad_h)) % 180.0

    # Spacing: account for non-square images (use padded dimensions)
    # Frequency in cycles/pixel along each axis (signs don't matter for magnitude)
    freq_x = dx / pad_w  # cycles per pixel in x
    freq_y = ky / pad_h  # cycles per pixel in y
    freq_magnitude = math.sqrt(freq_x ** 2 + freq_y ** 2)

    if freq_magnitude < 1e-12:
        logger.warning("detect (2D FFT): peak at DC — no periodic signal")
        return None

    spacing_px = 1.0 / freq_magnitude
    spacing_m = pixel_spacing_to_meters(spacing_px, lat, zoom)

    logger.info(
        "detect (2D FFT): dx=%.1f, ky=%.1f -> row_angle=%.1f°, "
        "spacing=%.1f px (%.2f m)",
        dx, ky, row_angle, spacing_px, spacing_m,
    )

    # 7. Plausibility-based spacing correction via radial profile
    if spacing_m > PLAUSIBLE_MAX_M or spacing_m < PLAUSIBLE_MIN_M:
        logger.info(
            "detect (2D FFT): spacing %.2fm outside plausible range "
            "[%.1f, %.1f], searching radial profile for better candidate",
            spacing_m, PLAUSIBLE_MIN_M, PLAUSIBLE_MAX_M,
        )
        correction = _best_plausible_spacing(
            magnitude, cy, cx, peak_x, peak_y,
            pad_h, pad_w, mpp, min_freq_radius, max_freq_radius,
        )
        if correction is not None:
            old_spacing_m = spacing_m
            spacing_px, spacing_m = correction
            logger.info(
                "detect (2D FFT): spacing corrected via radial profile: "
                "%.2fm -> %.2fm",
                old_spacing_m, spacing_m,
            )
        else:
            logger.info(
                "detect (2D FFT): no plausible candidate found in radial "
                "profile, keeping %.2fm",
                spacing_m,
            )

    # Sanity checks
    if spacing_m < 0.5:
        logger.warning("detect (2D FFT): spacing %.2f m too small", spacing_m)
    elif spacing_m > 10.0:
        logger.warning("detect (2D FFT): spacing %.2f m too large", spacing_m)

    # 8. Row count
    perp_rad = math.radians(row_angle + 90.0)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        row_count = 0
    else:
        projections = xs.astype(np.float64) * math.cos(perp_rad) + ys.astype(
            np.float64
        ) * math.sin(perp_rad)
        width_px = float(projections.max() - projections.min())
        row_count = max(1, int(round(width_px / spacing_px)) + 1)

    # 9. Confidence: peak SNR within band-pass region
    band_pass_mask = (dist_from_center > min_freq_radius) & (
        dist_from_center < max_freq_radius
    )
    band_pass_vals = np.abs(fft_shifted)[band_pass_mask]
    non_zero = band_pass_vals[band_pass_vals > 0]
    if len(non_zero) > 0:
        snr = peak_mag / float(np.median(non_zero))
        confidence = min(snr / 20.0, 1.0)
    else:
        confidence = 0.0

    result = FFT2DResult(
        angle_degrees=round(row_angle, 2),
        spacing_meters=round(spacing_m, 3),
        spacing_pixels=round(spacing_px, 2),
        row_count=row_count,
        confidence=round(confidence, 4),
        peak_position=(round(peak_x, 1), round(peak_y, 1)),
        magnitude_spectrum=log_magnitude,
    )
    logger.info(
        "detect (2D FFT): angle=%.1f°, spacing=%.2f m (%.1f px), rows=%d, "
        "confidence=%.3f",
        result.angle_degrees,
        result.spacing_meters,
        result.spacing_pixels,
        result.row_count,
        result.confidence,
    )
    return result
