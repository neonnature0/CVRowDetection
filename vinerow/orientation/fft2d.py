"""
2D FFT-based vineyard row orientation and spacing detection.

Computes a single 2D FFT on the masked image. Periodic parallel rows
produce a conjugate pair of peaks in the magnitude spectrum whose polar
coordinates directly encode row angle and spacing.

References:
    Delenne et al. 2006 -- vineyard row detection via 2D FFT
    Rabatel et al. 2008 -- crop row orientation from frequency domain
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from scipy.signal import find_peaks as _find_peaks
from scipy.ndimage import gaussian_filter1d

from vinerow.acquisition.geo_utils import meters_per_pixel, pixel_spacing_to_meters
from vinerow.types import CoarseOrientation

logger = logging.getLogger(__name__)


# Tight physical bounds for harmonic resolution (vine row spacing)
_HARMONIC_MIN_M = 1.2
_HARMONIC_MAX_M = 4.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subpixel_peak(mag: np.ndarray, py: int, px: int) -> tuple[float, float]:
    """Refine peak location to sub-pixel precision via parabolic fit."""
    h, w = mag.shape

    if 0 < px < w - 1:
        left = float(mag[py, px - 1])
        center = float(mag[py, px])
        right = float(mag[py, px + 1])
        denom = 2.0 * (2.0 * center - left - right)
        dx = (right - left) / denom if abs(denom) > 1e-12 else 0.0
    else:
        dx = 0.0

    if 0 < py < h - 1:
        top = float(mag[py - 1, px])
        center = float(mag[py, px])
        bottom = float(mag[py + 1, px])
        denom = 2.0 * (2.0 * center - top - bottom)
        dy = (bottom - top) / denom if abs(denom) > 1e-12 else 0.0
    else:
        dy = 0.0

    return float(py) + dy, float(px) + dx


def _extract_radial_profile(
    mag: np.ndarray,
    cy: int,
    cx: int,
    dx_unit: float,
    dy_unit: float,
    r_min: int,
    r_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a 1D magnitude profile along a radial direction with bilinear interpolation."""
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


def _resolve_harmonic(
    spacing_px: float,
    spacing_m: float,
    mpp: float,
    magnitude: np.ndarray,
    cy: int,
    cx: int,
    peak_x: float,
    peak_y: float,
    pad_h: int,
    pad_w: int,
) -> tuple[float, float]:
    """Check for harmonic confusion and correct spacing if needed.

    The 2D FFT can pick up the half-period (row-to-gap, ~1.25m) or a
    double-period (every-other-row, ~5m) instead of the true row-to-row
    spacing (~2.5m). Uses tight physical bounds [1.2, 4.5]m to detect
    implausible values and a 1D radial spectral tiebreaker when both 1x
    and 2x are plausible.

    Returns (corrected_spacing_px, corrected_spacing_m).
    """
    sp_half_m = spacing_m / 2.0
    sp_double_m = spacing_m * 2.0

    # Case 1: spacing too large — try halving
    if spacing_m > _HARMONIC_MAX_M:
        if sp_half_m >= _HARMONIC_MIN_M:
            logger.info(
                "harmonic: spacing %.2fm > %.1fm max, HALVING to %.2fm",
                spacing_m, _HARMONIC_MAX_M, sp_half_m,
            )
            return spacing_px / 2.0, sp_half_m
        logger.info(
            "harmonic: spacing %.2fm too large, half %.2fm also implausible, keeping",
            spacing_m, sp_half_m,
        )
        return spacing_px, spacing_m

    # Case 2: spacing too small — try doubling
    if spacing_m < _HARMONIC_MIN_M:
        if sp_double_m <= _HARMONIC_MAX_M:
            logger.info(
                "harmonic: spacing %.2fm < %.1fm min, DOUBLING to %.2fm",
                spacing_m, _HARMONIC_MIN_M, sp_double_m,
            )
            return spacing_px * 2.0, sp_double_m
        logger.info(
            "harmonic: spacing %.2fm too small, double %.2fm also implausible, keeping",
            spacing_m, sp_double_m,
        )
        return spacing_px, spacing_m

    # Case 3: both 1x and 2x plausible — use 1D radial spectral tiebreaker
    if _HARMONIC_MIN_M <= sp_double_m <= _HARMONIC_MAX_M:
        # Extract radial profile through the peak direction
        dx = peak_x - cx
        ky = peak_y - cy
        peak_dist = math.sqrt(dx**2 + ky**2)
        if peak_dist < 3:
            return spacing_px, spacing_m

        dx_unit = dx / peak_dist
        dy_unit = ky / peak_dist

        # Sample along the radial direction to get 1D spectral profile
        r_max = int(peak_dist * 3)  # look up to 3× the peak radius
        r_min = max(3, int(peak_dist * 0.3))
        radii, profile = _extract_radial_profile(
            magnitude, cy, cx, dx_unit, dy_unit, r_min, r_max,
        )

        if len(profile) < 10:
            return spacing_px, spacing_m

        # Check power at the sub-harmonic (half peak radius = double spacing).
        # If the FFT picked the half-period (row-to-gap), the true row-to-row
        # signal lives at half the peak radius (lower frequency).
        target_1x = peak_dist
        target_half = peak_dist / 2.0  # sub-harmonic = double spacing

        def _peak_power(target_r: float, search_radius: int = 3) -> float:
            if len(radii) == 0:
                return 0.0
            idx = int(np.argmin(np.abs(radii - target_r)))
            lo = max(0, idx - search_radius)
            hi = min(len(profile), idx + search_radius + 1)
            return float(np.max(profile[lo:hi]))

        power_1x = _peak_power(target_1x)
        power_half = _peak_power(target_half) if target_half >= r_min else 0.0

        ratio = power_half / max(power_1x, 1.0)

        logger.info(
            "harmonic: both plausible 1x=%.2fm 2x=%.2fm | "
            "radial power peak=%.0f sub-harmonic=%.0f (ratio=%.2f)",
            spacing_m, sp_double_m, power_1x, power_half, ratio,
        )

        # If the sub-harmonic has substantial power, the FFT likely picked
        # the half-period; true spacing is double.  Threshold 0.3 means the
        # row-to-row signal is at least 30% as strong as the row-to-gap signal.
        if ratio >= 0.3:
            logger.info(
                "harmonic: DOUBLING spacing (sub-harmonic ratio %.2f >= 0.30 -> half-period detected)",
                ratio,
            )
            return spacing_px * 2.0, sp_double_m

    # Spacing is plausible as-is
    logger.info("harmonic: spacing %.2fm within [%.1f, %.1f], no correction",
                spacing_m, _HARMONIC_MIN_M, _HARMONIC_MAX_M)
    return spacing_px, spacing_m


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
    plausible_min_m: float,
    plausible_max_m: float,
) -> tuple[float, float] | None:
    """Find the strongest peak whose spacing falls in the plausible range."""
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

        if plausible_min_m <= sp_m <= plausible_max_m and profile[idx] > best_mag:
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
    image: np.ndarray,
    mask: np.ndarray,
    mpp: float,
    lat: float,
    zoom: int,
    tile_size: int = 256,
    plausible_min_m: float = 0.8,
    plausible_max_m: float = 8.0,
) -> CoarseOrientation | None:
    """Full 2D FFT detection pipeline.

    Accepts a single-channel image (vegetation index, luminance, or fused)
    and a binary mask. Returns a CoarseOrientation with angle, spacing,
    and confidence estimates.

    Args:
        image: Single-channel uint8 or float image.
        mask: Binary mask (uint8, 0 or 255).
        mpp: Meters per pixel.
        lat: Centroid latitude (for row count estimation).
        zoom: Zoom level.
        tile_size: Tile pixel size.
        plausible_min_m: Minimum plausible row spacing in meters.
        plausible_max_m: Maximum plausible row spacing in meters.

    Returns:
        CoarseOrientation or None if no clear signal.
    """
    if cv2.countNonZero(mask) == 0:
        logger.warning("detect (2D FFT): mask is empty")
        return None

    h, w = image.shape[:2]

    # Mask, subtract mean, window — use float32 throughout to halve memory
    masked_img = image.astype(np.float32) * (mask > 0).astype(np.float32)
    valid_pixels = masked_img[mask > 0]
    if len(valid_pixels) == 0:
        return None
    masked_img[mask > 0] -= np.mean(valid_pixels)

    window_2d = np.outer(
        np.hanning(h).astype(np.float32),
        np.hanning(w).astype(np.float32),
    )
    windowed = masked_img * window_2d

    # 2D FFT with optional zero-padding
    # Padding improves frequency resolution but 4× memory cost.
    # Only pad small images (< 10M pixels → padded < 40M → ~320 MB complex64).
    max_base_pixels = 8_000_000
    pad_factor = 2 if h * w <= max_base_pixels else 1
    if pad_factor == 1:
        logger.info("detect (2D FFT): skipping zero-pad (image too large: %dx%d)", w, h)
    pad_h, pad_w = h * pad_factor, w * pad_factor
    # Use scipy.fft for memory efficiency — it works with float32 input
    # and we convert to float32 magnitude immediately to free complex arrays
    from scipy.fft import fft2 as _scipy_fft2, fftshift as _scipy_fftshift
    windowed_f32 = windowed.astype(np.float32)
    del windowed
    fft_result = _scipy_fft2(windowed_f32, s=(pad_h, pad_w))
    del windowed_f32
    magnitude = np.abs(_scipy_fftshift(fft_result)).astype(np.float32)
    del fft_result

    cy, cx = pad_h // 2, pad_w // 2

    # Band-pass filter
    max_spacing_m = plausible_max_m * 2.0  # generous low-freq cutoff
    max_spacing_px = max_spacing_m / mpp
    min_dim = min(pad_h, pad_w)
    min_freq_radius = max(5, int(min_dim / max_spacing_px))

    min_spacing_m = plausible_min_m * 0.5  # generous high-freq cutoff
    min_spacing_px = min_spacing_m / mpp
    max_freq_radius = int(min_dim / min_spacing_px)

    logger.info(
        "detect (2D FFT): image %dx%d, mpp=%.3f, band-pass radii=[%d, %d]",
        w, h, mpp, min_freq_radius, max_freq_radius,
    )

    Y, X = np.ogrid[:pad_h, :pad_w]
    dist_from_center = np.sqrt(
        (X.astype(np.float32) - cx) ** 2 + (Y.astype(np.float32) - cy) ** 2
    )

    magnitude[dist_from_center <= min_freq_radius] = 0.0
    magnitude[dist_from_center >= max_freq_radius] = 0.0

    log_magnitude = np.log1p(magnitude).copy()

    # Find peak in top half (conjugate symmetry)
    search_region = magnitude[:cy, :].copy()
    if np.max(search_region) < 1e-6:
        logger.warning("detect (2D FFT): no significant peak in band-pass region")
        return None

    peak_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
    peak_y_int, peak_x_int = int(peak_idx[0]), int(peak_idx[1])
    peak_mag = float(search_region[peak_y_int, peak_x_int])

    peak_y, peak_x = _subpixel_peak(search_region, peak_y_int, peak_x_int)

    logger.info(
        "detect (2D FFT): peak at (%.1f, %.1f) magnitude=%.1f",
        peak_x, peak_y, peak_mag,
    )

    # Convert peak to angle and spacing
    dx = peak_x - cx
    ky = peak_y - cy

    row_angle = math.degrees(math.atan2(-dx / pad_w, ky / pad_h)) % 180.0

    freq_x = dx / pad_w
    freq_y = ky / pad_h
    freq_magnitude = math.sqrt(freq_x ** 2 + freq_y ** 2)

    if freq_magnitude < 1e-12:
        logger.warning("detect (2D FFT): peak at DC — no periodic signal")
        return None

    spacing_px = 1.0 / freq_magnitude
    spacing_m = pixel_spacing_to_meters(spacing_px, lat, zoom, tile_size)

    logger.info(
        "detect (2D FFT): dx=%.1f, ky=%.1f -> row_angle=%.1f deg, "
        "spacing=%.1f px (%.2f m)",
        dx, ky, row_angle, spacing_px, spacing_m,
    )

    # Harmonic resolution: check for half/double-period confusion
    old_spacing_m = spacing_m
    spacing_px, spacing_m = _resolve_harmonic(
        spacing_px, spacing_m, mpp,
        magnitude, cy, cx, peak_x, peak_y, pad_h, pad_w,
    )
    if abs(spacing_m - old_spacing_m) > 0.01:
        logger.info(
            "detect (2D FFT): harmonic correction: %.2fm -> %.2fm",
            old_spacing_m, spacing_m,
        )

    # Plausibility-based spacing correction via radial profile
    if spacing_m > plausible_max_m or spacing_m < plausible_min_m:
        logger.info(
            "detect (2D FFT): spacing %.2fm outside plausible range "
            "[%.1f, %.1f], searching radial profile",
            spacing_m, plausible_min_m, plausible_max_m,
        )
        correction = _best_plausible_spacing(
            magnitude, cy, cx, peak_x, peak_y,
            pad_h, pad_w, mpp, min_freq_radius, max_freq_radius,
            plausible_min_m, plausible_max_m,
        )
        if correction is not None:
            old_spacing_m = spacing_m
            spacing_px, spacing_m = correction
            logger.info(
                "detect (2D FFT): spacing corrected: %.2fm -> %.2fm",
                old_spacing_m, spacing_m,
            )

    # Estimate angle uncertainty from peak angular width
    # Sample magnitude at same radius but varying angles
    peak_radius = math.sqrt(dx**2 + ky**2)
    if peak_radius > 3:
        n_angle_samples = 360
        angle_profile = np.zeros(n_angle_samples)
        for i in range(n_angle_samples):
            a = 2.0 * math.pi * i / n_angle_samples
            sx = cx + peak_radius * math.cos(a)
            sy = cy + peak_radius * math.sin(a)
            ix, iy = int(round(sx)), int(round(sy))
            if 0 <= iy < pad_h and 0 <= ix < pad_w:
                angle_profile[i] = magnitude[iy, ix]
        half_max = peak_mag * 0.5
        above_half = angle_profile >= half_max
        fwhm_bins = float(np.sum(above_half))
        angle_uncertainty_deg = (fwhm_bins / n_angle_samples) * 180.0
    else:
        angle_uncertainty_deg = 10.0

    # Estimate spacing uncertainty from radial peak width
    spacing_uncertainty_m = spacing_m * 0.05  # default 5%

    # Row count estimate
    perp_rad = math.radians(row_angle + 90.0)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        row_count = 0
    else:
        projections = xs.astype(np.float32) * math.cos(perp_rad) + ys.astype(
            np.float32
        ) * math.sin(perp_rad)
        width_px = float(projections.max() - projections.min())
        row_count = max(1, int(round(width_px / spacing_px)) + 1)

    # Confidence: peak SNR
    band_pass_mask = (dist_from_center > min_freq_radius) & (
        dist_from_center < max_freq_radius
    )
    band_pass_vals = magnitude[band_pass_mask]
    non_zero = band_pass_vals[band_pass_vals > 0]
    if len(non_zero) > 0:
        snr = peak_mag / float(np.median(non_zero))
        confidence = min(snr / 20.0, 1.0)
    else:
        confidence = 0.0

    result = CoarseOrientation(
        angle_deg=round(row_angle, 2),
        angle_confidence=round(confidence, 4),
        angle_uncertainty_deg=round(angle_uncertainty_deg, 2),
        spacing_m=round(spacing_m, 3),
        spacing_px=round(spacing_px, 2),
        spacing_confidence=round(confidence, 4),
        spacing_uncertainty_m=round(spacing_uncertainty_m, 3),
        row_count_estimate=row_count,
        log_magnitude=log_magnitude,
        peak_position=(round(peak_x, 1), round(peak_y, 1)),
    )
    logger.info(
        "detect (2D FFT): angle=%.1f deg, spacing=%.2f m (%.1f px), rows~%d, "
        "confidence=%.3f, angle_uncertainty=%.1f deg",
        result.angle_deg,
        result.spacing_m,
        result.spacing_px,
        result.row_count_estimate,
        result.angle_confidence,
        result.angle_uncertainty_deg,
    )
    return result
