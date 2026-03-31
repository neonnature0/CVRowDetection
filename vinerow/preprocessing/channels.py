"""
Multi-channel image preprocessing for vineyard row detection.

Produces four complementary channels from a BGR aerial image:
  1. ExG (Excess Green) — vine canopy vs soil
  2. Luminance (grayscale) — general brightness
  3. Normalized Vegetation — ExG / (R+G+B+eps) for illumination invariance
  4. Structure Tensor Magnitude — local anisotropy (detects linear features)

Each channel is scored for signal quality, and a weighted fusion is produced
for downstream stages.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from vinerow.config import PipelineConfig
from vinerow.types import ChannelQuality, PreprocessedChannels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual channel computation
# ---------------------------------------------------------------------------


def compute_exg(image_bgr: np.ndarray) -> np.ndarray:
    """Excess Green vegetation index: ExG = 2*G - R - B, normalized to 0-255 uint8."""
    if image_bgr.size == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    b, g, r = (
        image_bgr[:, :, 0].astype(np.float32),
        image_bgr[:, :, 1].astype(np.float32),
        image_bgr[:, :, 2].astype(np.float32),
    )
    exg = 2.0 * g - r - b
    exg_min, exg_max = exg.min(), exg.max()
    if exg_max - exg_min < 1e-6:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    return ((exg - exg_min) / (exg_max - exg_min) * 255.0).astype(np.uint8)


def compute_luminance(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale luminance."""
    if image_bgr.size == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def compute_normalized_vegetation(image_bgr: np.ndarray) -> np.ndarray:
    """Normalized vegetation index: ExG / (R+G+B+eps).

    Divides by total brightness to reduce sensitivity to shadows and
    illumination variation. Output normalized to 0-255 uint8.
    """
    if image_bgr.size == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    b, g, r = (
        image_bgr[:, :, 0].astype(np.float32),
        image_bgr[:, :, 1].astype(np.float32),
        image_bgr[:, :, 2].astype(np.float32),
    )
    total = r + g + b + 1e-6  # avoid division by zero
    exg = 2.0 * g - r - b
    nveg = exg / total
    nmin, nmax = nveg.min(), nveg.max()
    if nmax - nmin < 1e-6:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    return ((nveg - nmin) / (nmax - nmin) * 255.0).astype(np.uint8)


def compute_structure_tensor_magnitude(
    gray: np.ndarray, sigma: float = 1.5, window: int = 5,
) -> np.ndarray:
    """Structure tensor anisotropy magnitude.

    Computes the structure tensor J = [[Ix^2, IxIy], [IxIy, Iy^2]] at each pixel,
    smoothed over a local window. Returns the anisotropy ratio:
        (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
    where lambda1 >= lambda2 are the eigenvalues.

    High values indicate strong local orientation (linear features like vine rows).
    Low values indicate uniform or isotropic texture.

    Returns float32 in [0, 1].
    """
    if gray.size == 0:
        return np.zeros(gray.shape[:2], dtype=np.float32)

    img = gray.astype(np.float32) / 255.0

    # Compute gradients
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components (smoothed)
    Ixx = gaussian_filter(Ix * Ix, sigma=sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma=sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma=sigma)

    # Eigenvalues of 2x2 symmetric matrix [[Ixx, Ixy], [Ixy, Iyy]]
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    discriminant = np.maximum(trace * trace - 4.0 * det, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0

    # Anisotropy: high when one eigenvalue dominates (linear structure)
    denom = lambda1 + lambda2 + 1e-8
    anisotropy = (lambda1 - lambda2) / denom

    # Normalize to [0, 1]
    amax = anisotropy.max()
    if amax > 1e-6:
        anisotropy = anisotropy / amax

    return anisotropy.astype(np.float32)


# ---------------------------------------------------------------------------
# Channel quality scoring
# ---------------------------------------------------------------------------


def _score_channel(
    channel: np.ndarray, mask: np.ndarray, name: str,
) -> ChannelQuality:
    """Score a channel's signal quality within the masked region."""
    mask_pixels = mask > 0
    if not mask_pixels.any():
        return ChannelQuality(name=name, std_dev=0.0, contrast=0.0, weight=0.0)

    vals = channel[mask_pixels].astype(np.float64)
    std = float(np.std(vals))

    # Contrast: (p95 - p5) / range
    p5, p95 = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    if channel.dtype == np.float32:
        contrast = p95 - p5  # already 0-1 range
    else:
        contrast = (p95 - p5) / 255.0

    return ChannelQuality(name=name, std_dev=std, contrast=contrast, weight=0.0)


def _compute_weights(qualities: list[ChannelQuality]) -> list[ChannelQuality]:
    """Assign weights to channels proportional to their contrast score."""
    total_contrast = sum(q.contrast for q in qualities)
    if total_contrast < 1e-6:
        # Equal weights if all channels are dead
        n = len(qualities)
        for q in qualities:
            q.weight = 1.0 / n if n > 0 else 0.0
        return qualities

    for q in qualities:
        q.weight = q.contrast / total_contrast

    return qualities


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def _erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Erode a binary mask to avoid boundary artifacts."""
    if erosion_px <= 0:
        return mask.copy()
    kernel_size = erosion_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    if cv2.countNonZero(eroded) == 0:
        logger.warning("Eroded mask is empty, falling back to original")
        return mask.copy()
    return eroded


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_channels(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    mpp: float,
    config: PipelineConfig,
) -> PreprocessedChannels:
    """Full multi-channel preprocessing pipeline.

    Args:
        image_bgr: Input BGR image.
        mask: Binary mask (uint8, 0 or 255).
        mpp: Meters per pixel.
        config: Pipeline configuration.

    Returns:
        PreprocessedChannels with all channels, quality scores, and fusion.
    """
    h, w = image_bgr.shape[:2]
    logger.info("Preprocessing: image %dx%d, mpp=%.3f", w, h, mpp)

    # Ensure mask matches image dimensions
    if mask.shape[:2] != (h, w):
        raise ValueError(f"Mask shape {mask.shape[:2]} != image shape {(h, w)}")

    # Erode mask
    original_mask = mask.copy()
    eroded_mask = _erode_mask(mask, config.mask_erosion_px)

    # Apply mask to image
    masked_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=eroded_mask)

    # Compute channels
    exg = compute_exg(masked_bgr)
    luminance = compute_luminance(masked_bgr)
    normalized_veg = compute_normalized_vegetation(masked_bgr)

    # Structure tensor uses luminance as input, scale sigma to resolution
    # Skip for large images (>12M pixels) to avoid OOM — not needed by gabor/luminance modes
    n_pixels = luminance.shape[0] * luminance.shape[1]
    if n_pixels <= 12_000_000:
        struct_sigma = max(1.0, 0.3 / mpp)  # ~0.3m ground distance
        structure_mag = compute_structure_tensor_magnitude(luminance, sigma=struct_sigma)
    else:
        logger.info("Skipping structure tensor (image %dM pixels > 12M limit)", n_pixels // 1_000_000)
        structure_mag = np.zeros(luminance.shape[:2], dtype=np.float32)

    # Apply mask to all channels
    exg = cv2.bitwise_and(exg, exg, mask=eroded_mask)
    luminance = cv2.bitwise_and(luminance, luminance, mask=eroded_mask)
    normalized_veg = cv2.bitwise_and(normalized_veg, normalized_veg, mask=eroded_mask)
    structure_mag = structure_mag * (eroded_mask > 0).astype(np.float32)

    # Score channels
    qualities = [
        _score_channel(exg, eroded_mask, "exg"),
        _score_channel(luminance, eroded_mask, "luminance"),
        _score_channel(normalized_veg, eroded_mask, "normalized_veg"),
        _score_channel(structure_mag, eroded_mask, "structure_mag"),
    ]
    qualities = _compute_weights(qualities)

    for q in qualities:
        logger.info(
            "  Channel %-16s: std=%.1f, contrast=%.3f, weight=%.3f",
            q.name, q.std_dev, q.contrast, q.weight,
        )

    # Fused channel: weighted combination normalized to [0, 1]
    fused = np.zeros((h, w), dtype=np.float32)
    for q, ch in zip(qualities, [exg, luminance, normalized_veg, structure_mag]):
        if q.weight < 0.01:
            continue
        if ch.dtype == np.uint8:
            ch_f = ch.astype(np.float32) / 255.0
        else:
            ch_f = ch.astype(np.float32)
        fused += q.weight * ch_f

    # Normalize fused to [0, 1] within mask
    mask_pixels = eroded_mask > 0
    if mask_pixels.any():
        fmin = float(fused[mask_pixels].min())
        fmax = float(fused[mask_pixels].max())
        if fmax - fmin > 1e-6:
            fused[mask_pixels] = (fused[mask_pixels] - fmin) / (fmax - fmin)

    # For the FFT detector, also produce a uint8 version of the best channel
    # (the FFT works on uint8 or float, but the fused float32 is preferred)

    return PreprocessedChannels(
        exg=exg,
        luminance=luminance,
        normalized_veg=normalized_veg,
        structure_mag=structure_mag,
        mask=eroded_mask,
        original_mask=original_mask,
        fused=fused,
        channel_qualities=qualities,
        image_bgr=image_bgr,
    )
