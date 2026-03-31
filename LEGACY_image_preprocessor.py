"""
Image preprocessing pipeline for vineyard row detection.

Converts a BGR satellite tile image into multiple representations optimised
for downstream line/frequency detectors:
  - Excess Green vegetation index (vine canopy vs bare soil)
  - CLAHE-enhanced grayscale
  - Canny edge map
All outputs are masked to the vineyard block polygon.
"""

from dataclasses import dataclass
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    grayscale: np.ndarray  # uint8
    enhanced: np.ndarray  # CLAHE-enhanced grayscale
    vegetation: np.ndarray  # ExG vegetation index, uint8
    edges: np.ndarray  # Canny edge map, uint8
    mask: np.ndarray  # Binary polygon mask, uint8
    use_vegetation: bool  # True if ExG signal was strong enough


# ---------------------------------------------------------------------------
# Individual stages
# ---------------------------------------------------------------------------


def compute_vegetation_index(image_bgr: np.ndarray) -> np.ndarray:
    """Excess Green Index: ExG = 2*G - R - B.

    Convert to float, compute, normalize to 0-255 uint8.
    Vine rows appear bright (green canopy), inter-row appears dark (brown soil).
    """
    if image_bgr.size == 0:
        logger.warning("compute_vegetation_index: empty input image")
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    b, g, r = (
        image_bgr[:, :, 0].astype(np.float32),
        image_bgr[:, :, 1].astype(np.float32),
        image_bgr[:, :, 2].astype(np.float32),
    )

    exg = 2.0 * g - r - b  # range roughly -510 to +510

    # Normalize to 0-255
    exg_min = exg.min()
    exg_max = exg.max()
    if exg_max - exg_min < 1e-6:
        logger.debug("compute_vegetation_index: flat ExG (no contrast)")
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    exg_norm = (exg - exg_min) / (exg_max - exg_min) * 255.0
    return exg_norm.astype(np.uint8)


def apply_clahe(
    gray: np.ndarray, clip_limit: float = 3.0, grid_size: int = 8
) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) on grayscale image."""
    if gray.size == 0:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
    )
    return clahe.apply(gray)


def compute_edges(
    enhanced: np.ndarray, low_thresh: int = 30, high_thresh: int = 90,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Gaussian blur then Canny edge detection.

    Args:
        enhanced: Input grayscale image.
        low_thresh: Canny low threshold.
        high_thresh: Canny high threshold.
        blur_kernel: Gaussian blur kernel size (must be odd). Larger values
            suppress fine detail (individual vines) and emphasize row-level edges.
    """
    if enhanced.size == 0:
        return enhanced
    # Ensure kernel is odd
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(enhanced, (k, k), 0)
    return cv2.Canny(blurred, low_thresh, high_thresh)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess(image_bgr: np.ndarray, mask: np.ndarray, mpp: float = 0.15) -> PreprocessResult:
    """Full preprocessing pipeline.

    Steps:
        1. Apply mask to image (zero outside polygon).
        2. Compute ExG vegetation index.
        3. Check if ExG has enough signal (std_dev > 15) -- if not,
           use_vegetation=False, fall back to grayscale.
        4. Apply CLAHE to either vegetation or grayscale (whichever is stronger).
        5. Erode mask by 3 px to avoid boundary artefacts.
        6. Apply eroded mask.
        7. Canny edge detection on enhanced image with resolution-scaled blur.
        8. Mask the edges too.

    Args:
        image_bgr: Input BGR image.
        mask: Binary mask (uint8, 0 or 255).
        mpp: Meters per pixel (used to scale blur kernel — higher resolution
            needs larger blur to suppress individual vine detail).

    Returns:
        PreprocessResult with all intermediate images.
    """
    h, w = image_bgr.shape[:2]
    logger.info("preprocess: image %dx%d, mask non-zero %d px", w, h, cv2.countNonZero(mask))

    # Ensure mask is the right shape
    if mask.shape[:2] != (h, w):
        logger.error(
            "preprocess: mask shape %s != image shape %s", mask.shape[:2], (h, w)
        )
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {(h, w)}"
        )

    # 1. Apply mask to image
    masked_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    # Convert to grayscale
    grayscale = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # 2. Compute ExG vegetation index
    vegetation = compute_vegetation_index(masked_image)

    # 3. Check vegetation signal strength (only within mask)
    mask_pixels = mask > 0
    if mask_pixels.any():
        veg_std = float(np.std(vegetation[mask_pixels]))
    else:
        veg_std = 0.0
        logger.warning("preprocess: mask is entirely empty")

    use_vegetation = veg_std > 15.0
    logger.info(
        "preprocess: vegetation std=%.1f -> %s",
        veg_std,
        "using ExG" if use_vegetation else "falling back to grayscale",
    )

    # 4. Apply CLAHE to the stronger signal
    source_for_clahe = vegetation if use_vegetation else grayscale
    enhanced = apply_clahe(source_for_clahe)

    # 5. Erode mask by 3 px to avoid boundary artefacts
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded_mask = cv2.erode(mask, erode_kernel, iterations=1)

    # Guard against eroding the mask to nothing (very small polygons)
    if cv2.countNonZero(eroded_mask) == 0:
        logger.warning(
            "preprocess: eroded mask is empty, falling back to original mask"
        )
        eroded_mask = mask.copy()

    # 6. Apply eroded mask to enhanced image
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=eroded_mask)

    # 7. Canny edge detection with resolution-scaled blur.
    # At high resolution (small mpp), we need a larger blur kernel to suppress
    # individual vine canopy edges and only detect row-level features.
    # Target: blur radius should cover ~0.5m of ground distance.
    blur_radius_px = max(2, int(0.5 / mpp))
    blur_kernel = blur_radius_px * 2 + 1  # ensure odd
    blur_kernel = min(blur_kernel, 15)  # cap at 15 to avoid over-smoothing
    logger.info("preprocess: blur kernel=%d (mpp=%.3f, radius=%dpx)", blur_kernel, mpp, blur_radius_px)
    edges = compute_edges(enhanced, blur_kernel=blur_kernel)

    # 8. Mask edges
    edges = cv2.bitwise_and(edges, edges, mask=eroded_mask)

    logger.info(
        "preprocess: edges non-zero %d px, enhanced range [%d, %d]",
        cv2.countNonZero(edges),
        int(enhanced.min()),
        int(enhanced.max()),
    )

    return PreprocessResult(
        grayscale=grayscale,
        enhanced=enhanced,
        vegetation=vegetation,
        edges=edges,
        mask=eroded_mask,
        use_vegetation=use_vegetation,
    )
