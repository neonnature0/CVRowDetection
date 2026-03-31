"""
Row-likelihood map generation via ridge detection.

Produces a per-pixel score (0-1) indicating how likely each pixel is to
lie on a vine row centerline. Supports multiple ridge detection strategies:

  - hessian:       Hessian eigenvalue on all 4 channels, max fusion (default)
  - luminance:     Hessian on luminance channel only
  - exg_only:      Hessian on ExG channel only
  - gabor:         Gabor filter tuned to row frequency on luminance
  - ensemble:      max(hessian_all, gabor_luminance) per pixel
  - hessian_small: Hessian with ridge_scale_factor=0.10 (sharper ridges)
  - hessian_large: Hessian with ridge_scale_factor=0.25 (broader ridges)
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from vinerow.config import PipelineConfig
from vinerow.types import CoarseOrientation, PreprocessedChannels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hessian ridge filter
# ---------------------------------------------------------------------------


def _hessian_ridge_response(
    image: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hessian-based ridge strength and orientation at scale sigma.

    Ridge strength = max(0, -lambda_min). Only detects bright ridges (vine
    canopy brighter than inter-row).

    Returns:
        (ridge_strength, ridge_angle) — float32 arrays.
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    smoothed = gaussian_filter(img, sigma=sigma)

    Ixx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3)

    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    discriminant = np.maximum(trace * trace - 4.0 * det, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    lambda2 = (trace - sqrt_disc) / 2.0
    ridge_strength = np.maximum(-lambda2, 0.0)

    ridge_angle = 0.5 * np.arctan2(2.0 * Ixy, Ixx - Iyy)
    ridge_angle = np.degrees(ridge_angle) % 180.0

    return ridge_strength.astype(np.float32), ridge_angle.astype(np.float32)


def _oriented_suppression(
    ridge_strength: np.ndarray,
    ridge_angle: np.ndarray,
    target_angle_deg: float,
    tolerance_deg: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Suppress ridge responses not aligned with the expected row direction."""
    diff = np.abs(ridge_angle - target_angle_deg)
    diff = np.minimum(diff, 180.0 - diff)

    weight = np.where(
        diff <= tolerance_deg,
        np.cos(diff / tolerance_deg * (np.pi / 2.0)),
        0.0,
    )

    suppressed = ridge_strength * weight.astype(np.float32)
    suppressed *= (mask > 0).astype(np.float32)
    return suppressed


def _normalize_to_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] within masked region."""
    mask_pixels = mask > 0
    if mask_pixels.any():
        amax = float(arr[mask_pixels].max())
        if amax > 1e-6:
            arr = arr.copy()
            arr[mask_pixels] = arr[mask_pixels] / amax
    return arr


def _hessian_on_channels(
    channel_list: list[tuple[str, np.ndarray]],
    qualities: list,
    sigma: float,
    target_angle: float,
    tolerance: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Run Hessian ridge detection on multiple channels, fuse via max."""
    h, w = mask.shape[:2]
    fused = np.zeros((h, w), dtype=np.float32)

    for name, channel in channel_list:
        if channel is None:
            continue
        quality = next((q for q in qualities if q.name == name), None)
        if quality and quality.contrast < 0.02:
            logger.debug("  Skipping channel %s (contrast=%.3f too low)", name, quality.contrast)
            continue

        ridge_strength, ridge_angle = _hessian_ridge_response(channel, sigma)
        suppressed = _oriented_suppression(
            ridge_strength, ridge_angle, target_angle, tolerance, mask,
        )
        suppressed = _normalize_to_mask(suppressed, mask)

        logger.debug("  Channel %-16s: max=%.3f", name,
                      float(suppressed[mask > 0].max()) if (mask > 0).any() else 0)

        fused = np.maximum(fused, suppressed)

    return fused


# ---------------------------------------------------------------------------
# Gabor ridge filter
# ---------------------------------------------------------------------------


def _gabor_ridge(
    image: np.ndarray,
    angle_deg: float,
    spacing_px: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Gabor filter tuned to the expected row spacing and orientation.

    Gabor filters are bandpass in both spatial frequency and orientation,
    making them ideal for detecting periodic parallel structures like vine
    rows. Unlike the Hessian (which responds to all ridge-like features),
    Gabor only responds at the target frequency, suppressing noise at
    other spatial scales.

    Args:
        image: Single-channel image (uint8 or float32).
        angle_deg: Row direction in image coordinates (0-180 degrees).
        spacing_px: Expected row spacing in pixels.
        mask: Binary mask (uint8).

    Returns:
        Gabor response map (float32, 0-1).
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    # Gabor parameters
    lambd = spacing_px                  # wavelength = row spacing
    sigma = spacing_px * 0.4            # bandwidth (wider = more selective)
    gamma = 0.3                         # aspect ratio (< 1 = elongated along ridge)
    # Gabor theta is perpendicular to the ridge (row) direction.
    # OpenCV uses the angle of the normal to the parallel stripes.
    theta = math.radians(angle_deg + 90.0)

    ksize = int(spacing_px * 3) | 1     # odd kernel size, ~3 wavelengths
    ksize = max(ksize, 5)
    ksize = min(ksize, 127)             # cap at OpenCV limit

    # Generate two kernels: psi=0 (even/cosine) and psi=pi/2 (odd/sine)
    # Take the magnitude (energy) to get phase-independent response
    kernel_cos = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F,
    )
    kernel_sin = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi=math.pi / 2, ktype=cv2.CV_32F,
    )

    resp_cos = cv2.filter2D(img, cv2.CV_32F, kernel_cos)
    resp_sin = cv2.filter2D(img, cv2.CV_32F, kernel_sin)

    # Energy envelope (phase-independent magnitude)
    response = np.sqrt(resp_cos ** 2 + resp_sin ** 2)

    # Apply mask and normalize
    response *= (mask > 0).astype(np.float32)
    response = _normalize_to_mask(response, mask)

    return response


# ---------------------------------------------------------------------------
# Smoothing + normalization (shared finalization)
# ---------------------------------------------------------------------------


def _finalize_likelihood(
    likelihood: np.ndarray,
    mask: np.ndarray,
    smooth_sigma: float,
) -> np.ndarray:
    """Apply light smoothing, re-mask, and normalize to [0,1]."""
    likelihood = gaussian_filter(likelihood, sigma=smooth_sigma)
    likelihood *= (mask > 0).astype(np.float32)

    mask_pixels = mask > 0
    if mask_pixels.any():
        fmax = float(likelihood[mask_pixels].max())
        if fmax > 1e-6:
            likelihood[mask_pixels] = likelihood[mask_pixels] / fmax

    return likelihood


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_row_likelihood(
    preprocessed: PreprocessedChannels,
    coarse: CoarseOrientation,
    mpp: float,
    config: PipelineConfig,
) -> np.ndarray:
    """Generate a per-pixel row-likelihood map.

    The strategy is selected by config.ridge_mode. All strategies produce a
    float32 array in [0, 1] with the same spatial dimensions as the input.
    """
    h, w = preprocessed.mask.shape[:2]
    mask = preprocessed.mask
    mode = config.ridge_mode

    # Determine sigma based on mode
    if mode == "hessian_small":
        scale_factor = 0.10
    elif mode == "hessian_large":
        scale_factor = 0.25
    else:
        scale_factor = config.ridge_scale_factor

    sigma = scale_factor * coarse.spacing_px
    sigma = max(1.0, min(sigma, 20.0))

    target_angle = coarse.angle_deg
    tolerance = config.ridge_angle_tolerance_deg

    logger.info(
        "Ridge detection [%s]: sigma=%.1f px (%.2f m), target_angle=%.1f deg, "
        "spacing_px=%.1f, tolerance=%.1f deg",
        mode, sigma, sigma * mpp, target_angle, coarse.spacing_px, tolerance,
    )

    # Build channel lists
    all_channels: list[tuple[str, np.ndarray]] = [
        ("exg", preprocessed.exg),
        ("luminance", preprocessed.luminance),
        ("normalized_veg", preprocessed.normalized_veg),
    ]
    if preprocessed.structure_mag is not None:
        struct_uint8 = (preprocessed.structure_mag * 255).astype(np.uint8)
        all_channels.append(("structure_mag", struct_uint8))

    qualities = preprocessed.channel_qualities

    # --- Strategy dispatch ---

    if mode in ("hessian", "hessian_small", "hessian_large"):
        # Hessian on all channels, max fusion
        likelihood = _hessian_on_channels(
            all_channels, qualities, sigma, target_angle, tolerance, mask,
        )

    elif mode == "luminance":
        # Hessian on luminance only
        likelihood = _hessian_on_channels(
            [("luminance", preprocessed.luminance)],
            qualities, sigma, target_angle, tolerance, mask,
        )

    elif mode == "exg_only":
        # Hessian on ExG only
        likelihood = _hessian_on_channels(
            [("exg", preprocessed.exg)],
            qualities, sigma, target_angle, tolerance, mask,
        )

    elif mode == "gabor":
        # Gabor filter on luminance
        likelihood = _gabor_ridge(
            preprocessed.luminance, target_angle, coarse.spacing_px, mask,
        )

    elif mode == "ensemble":
        # Max of Hessian (all channels) and Gabor (luminance)
        hessian_lk = _hessian_on_channels(
            all_channels, qualities, sigma, target_angle, tolerance, mask,
        )
        hessian_lk = _normalize_to_mask(hessian_lk, mask)

        gabor_lk = _gabor_ridge(
            preprocessed.luminance, target_angle, coarse.spacing_px, mask,
        )

        likelihood = np.maximum(hessian_lk, gabor_lk)

    else:
        logger.warning("Unknown ridge_mode '%s', falling back to hessian", mode)
        likelihood = _hessian_on_channels(
            all_channels, qualities, sigma, target_angle, tolerance, mask,
        )

    # Normalize before smoothing
    likelihood = _normalize_to_mask(likelihood, mask)

    # Light smoothing to reduce pixel-level noise
    smooth_sigma = max(0.5, sigma * 0.15)
    likelihood = _finalize_likelihood(likelihood, mask, smooth_sigma)

    mask_pixels = mask > 0
    coverage = float(np.mean(likelihood[mask_pixels] > 0.1)) if mask_pixels.any() else 0
    logger.info(
        "Ridge likelihood [%s]: mean=%.3f, max=%.3f, coverage=%.1f%%",
        mode,
        float(likelihood[mask_pixels].mean()) if mask_pixels.any() else 0,
        float(likelihood[mask_pixels].max()) if mask_pixels.any() else 0,
        coverage * 100,
    )

    return likelihood
