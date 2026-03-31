"""
Approach B -- FFT-based frequency analysis for vineyard row detection.

Sweeps candidate angles, projects the image onto the perpendicular axis at
each angle (a single-angle Radon transform), then uses FFT to find the
dominant periodic frequency. The angle with the strongest spectral peak is
taken as the row orientation; the corresponding frequency gives the row
spacing.
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
class FFTResult:
    angle_degrees: float
    spacing_meters: float
    spacing_pixels: float
    row_count: int
    confidence: float
    angle_response: np.ndarray  # Signal strength per angle
    best_fft_spectrum: np.ndarray  # 1D FFT magnitude at best angle
    peak_frequency: float  # cycles/pixel


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def project_onto_axis(
    image: np.ndarray, mask: np.ndarray, angle_degrees: float
) -> np.ndarray:
    """Sum pixel intensities along lines parallel to *angle_degrees*.

    This is equivalent to the Radon transform at a single angle.

    Implementation:
        1. Rotate image by -angle_degrees so rows become vertical.
        2. Rotate mask identically.
        3. Sum columns (axis=0) but only where mask is active.
        4. Normalise by the number of active pixels per column to avoid
           bias from columns that happen to intersect more of the polygon.

    Returns:
        1D array -- the intensity profile perpendicular to the rows.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.array([], dtype=np.float64)

    centre = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(centre, -angle_degrees, 1.0)

    # Compute new bounding box so nothing is clipped
    cos_a = abs(rot_mat[0, 0])
    sin_a = abs(rot_mat[0, 1])
    new_w = int(math.ceil(h * sin_a + w * cos_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))

    # Adjust rotation matrix for the new canvas
    rot_mat[0, 2] += (new_w - w) / 2.0
    rot_mat[1, 2] += (new_h - h) / 2.0

    rotated_img = cv2.warpAffine(
        image, rot_mat, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = cv2.warpAffine(
        mask, rot_mat, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Sum columns where mask is active
    img_float = rotated_img.astype(np.float64)
    mask_float = (rotated_mask > 0).astype(np.float64)

    col_sums = np.sum(img_float * mask_float, axis=0)
    col_counts = np.sum(mask_float, axis=0)

    # Filter out columns with too few valid pixels — these produce noisy
    # averages that create edge artifacts in the FFT.
    max_count = float(np.max(col_counts)) if col_counts.size > 0 else 0.0
    min_valid = max(max_count * 0.1, 5.0)
    valid_cols = col_counts >= min_valid

    # Normalise only columns with enough valid pixels
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = np.where(valid_cols, col_sums / col_counts, 0.0)

    # Trim to the range of valid columns
    valid_indices = np.nonzero(valid_cols)[0]
    if len(valid_indices) == 0:
        return np.array([], dtype=np.float64)

    profile = profile[valid_indices[0] : valid_indices[-1] + 1]
    return profile


def analyze_frequency(
    profile_1d: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """FFT of 1D profile to find dominant periodic frequency.

    1. Subtract mean (remove DC).
    2. Apply Hamming window.
    3. Compute FFT magnitude (positive frequencies only).
    4. Ignore very low frequencies (wavelength > len/3).
    5. Find peak frequency.

    Returns:
        (peak_freq_cycles_per_pixel, peak_magnitude, fft_spectrum)
    """
    n = len(profile_1d)
    if n < 6:
        return 0.0, 0.0, np.array([], dtype=np.float64)

    # 1. Remove DC
    signal = profile_1d - np.mean(profile_1d)

    # 2. Hamming window
    window = np.hamming(n)
    signal = signal * window

    # 3. FFT -- positive half only
    fft_vals = np.fft.rfft(signal)
    magnitudes = np.abs(fft_vals)

    # Frequency axis: bin k corresponds to k / n cycles per pixel
    freqs = np.fft.rfftfreq(n)  # 0 to 0.5

    # 4. Ignore DC (index 0) and very low freqs (wavelength > n/3)
    min_freq = 3.0 / n  # wavelength = n/3
    valid_mask = freqs >= min_freq

    if not np.any(valid_mask):
        return 0.0, 0.0, magnitudes

    masked_mag = np.where(valid_mask, magnitudes, 0.0)
    peak_idx = int(np.argmax(masked_mag))

    peak_freq = float(freqs[peak_idx])
    peak_mag = float(magnitudes[peak_idx])

    return peak_freq, peak_mag, magnitudes


def find_dominant_angle(
    image: np.ndarray,
    mask: np.ndarray,
    angle_start: float = 0.0,
    angle_end: float = 180.0,
    angle_step: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Sweep angles, compute FFT peak magnitude at each.

    1. For each candidate angle: project_onto_axis -> analyze_frequency
       -> store peak magnitude.
    2. Find angle with highest peak magnitude.
    3. Refine: re-sweep +/-2 deg around peak with 0.1 deg step.

    Returns:
        (best_angle_degrees, angle_response_array)
    """
    angles = np.arange(angle_start, angle_end, angle_step)
    responses = np.zeros(len(angles), dtype=np.float64)

    logger.info(
        "find_dominant_angle: sweeping %d angles [%.1f, %.1f) step %.1f",
        len(angles),
        angle_start,
        angle_end,
        angle_step,
    )

    for i, angle in enumerate(angles):
        profile = project_onto_axis(image, mask, angle)
        if len(profile) < 6:
            continue
        _, peak_mag, _ = analyze_frequency(profile)
        responses[i] = peak_mag

    if np.max(responses) < 1e-6:
        logger.warning("find_dominant_angle: no significant response at any angle")
        return 0.0, responses

    coarse_best_idx = int(np.argmax(responses))
    coarse_best_angle = float(angles[coarse_best_idx])
    logger.info(
        "find_dominant_angle: coarse best=%.1f deg (response=%.1f)",
        coarse_best_angle,
        responses[coarse_best_idx],
    )

    # Refine: +/-2 deg around peak with 0.1 deg step
    refine_start = max(angle_start, coarse_best_angle - 2.0)
    refine_end = min(angle_end, coarse_best_angle + 2.0)
    refine_angles = np.arange(refine_start, refine_end, 0.1)
    refine_responses = np.zeros(len(refine_angles), dtype=np.float64)

    for i, angle in enumerate(refine_angles):
        profile = project_onto_axis(image, mask, angle)
        if len(profile) < 6:
            continue
        _, peak_snr, _ = analyze_frequency(profile)
        refine_responses[i] = peak_snr

    if np.max(refine_responses) > 0:
        best_idx = int(np.argmax(refine_responses))
        best_angle = float(refine_angles[best_idx])
        logger.info(
            "find_dominant_angle: refined best=%.2f deg (response=%.1f)",
            best_angle,
            refine_responses[best_idx],
        )
    else:
        best_angle = coarse_best_angle

    # Normalise to [0, 180)
    best_angle = best_angle % 180

    return best_angle, responses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def detect(
    preprocessed: PreprocessResult, lat: float, zoom: int, tile_size: int = 256,
) -> FFTResult | None:
    """Full FFT detection pipeline.

    1. Use vegetation index if use_vegetation else enhanced grayscale.
    2. find_dominant_angle.
    3. project_onto_axis at best angle.
    4. analyze_frequency for spacing.
    5. spacing_pixels = 1.0 / peak_frequency (if peak_frequency > 0).
    6. pixel_spacing_to_meters.
    7. Estimate row count (mask width / spacing).
    8. Confidence: peak_prominence * 0.6 + angle_sharpness * 0.4.

    Returns:
        FFTResult or None if no clear signal.
    """
    # 1. Choose input channel
    if preprocessed.use_vegetation:
        image = preprocessed.vegetation
        logger.info("detect (FFT): using vegetation index")
    else:
        image = preprocessed.enhanced
        logger.info("detect (FFT): using enhanced grayscale")

    mask = preprocessed.mask

    if cv2.countNonZero(mask) == 0:
        logger.warning("detect (FFT): mask is empty")
        return None

    # 2. Find dominant angle (projection axis — perpendicular to rows)
    projection_angle, angle_response = find_dominant_angle(image, mask)

    # Convert from projection angle to row angle (add 90°).
    # The FFT finds the projection direction that produces the strongest
    # periodic signal. Rows are perpendicular to this projection.
    # Example: projection at 0° (horizontal columns) → rows are at 90° (N-S).
    best_angle = (projection_angle + 90.0) % 180.0
    logger.info(
        "detect (FFT): projection angle=%.1f -> row angle=%.1f",
        projection_angle, best_angle,
    )

    # 3. Project at the projection angle (NOT the row angle)
    profile = project_onto_axis(image, mask, projection_angle)
    if len(profile) < 6:
        logger.warning(
            "detect (FFT): profile too short (%d) at best angle %.1f",
            len(profile),
            best_angle,
        )
        return None

    # 4. Analyze frequency
    peak_freq, peak_mag, fft_spectrum = analyze_frequency(profile)

    if peak_freq <= 0:
        logger.warning("detect (FFT): no valid peak frequency")
        return None

    # 5. Spacing in pixels
    spacing_px = 1.0 / peak_freq

    # Sanity: spacing should be at least a few pixels
    if spacing_px < 3.0:
        logger.warning("detect (FFT): spacing %.1f px suspiciously small", spacing_px)
        return None

    # 6. Convert to meters
    spacing_m = pixel_spacing_to_meters(spacing_px, lat, zoom, tile_size)

    # Harmonic correction: vine rows often produce a dominant sub-harmonic
    # at 2× the actual spacing (every other row appears slightly different).
    # Check if halving the spacing puts it in the plausible vine row range
    # (1.5-4.0m) AND if there's a significant peak at 2× the frequency.
    if spacing_m > 4.0:
        half_spacing_m = spacing_m / 2.0
        double_freq = peak_freq * 2.0
        # Check if there's a peak near 2× frequency
        freq_idx = int(round(double_freq * len(profile)))
        if 0 < freq_idx < len(fft_spectrum):
            double_freq_mag = float(fft_spectrum[freq_idx])
            # Accept if the 2× peak is at least 15% as strong as the original.
            # The sub-harmonic at the actual row spacing is often weaker than
            # the dominant 2× period, so use a low threshold.
            if double_freq_mag > peak_mag * 0.15 and 1.5 <= half_spacing_m <= 4.5:
                logger.info(
                    "detect (FFT): harmonic correction: %.2fm -> %.2fm "
                    "(2x peak mag=%.1f, primary=%.1f)",
                    spacing_m, half_spacing_m, double_freq_mag, peak_mag,
                )
                spacing_m = half_spacing_m
                spacing_px = spacing_px / 2.0
                peak_freq = double_freq
            else:
                logger.info(
                    "detect (FFT): 2x peak too weak (%.1f vs %.1f) or half-spacing "
                    "%.2fm out of range — keeping %.2fm",
                    double_freq_mag, peak_mag, half_spacing_m, spacing_m,
                )

    # Sanity check
    if spacing_m < 0.5:
        logger.warning("detect (FFT): spacing %.2f m seems too small", spacing_m)
    elif spacing_m > 10.0:
        logger.warning("detect (FFT): spacing %.2f m seems too large", spacing_m)

    # 7. Row count: project mask onto axis perpendicular to rows.
    # best_angle is the row direction, so perpendicular = best_angle + 90.
    perp_rad = math.radians(best_angle + 90.0)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        row_count = 0
    else:
        projections = xs.astype(np.float64) * math.cos(perp_rad) + ys.astype(
            np.float64
        ) * math.sin(perp_rad)
        width_px = float(projections.max() - projections.min())
        row_count = max(1, int(round(width_px / spacing_px)) + 1)

    # 8. Confidence
    #   peak_prominence: how much the peak stands out from the median spectrum
    valid_spectrum = fft_spectrum[fft_spectrum > 0]
    if len(valid_spectrum) > 1:
        median_spec = float(np.median(valid_spectrum))
        if median_spec > 0:
            prominence_ratio = min(peak_mag / median_spec, 10.0) / 10.0
        else:
            prominence_ratio = 1.0 if peak_mag > 0 else 0.0
    else:
        prominence_ratio = 0.0

    #   angle_sharpness: how much the best angle's response stands out
    valid_responses = angle_response[angle_response > 0]
    if len(valid_responses) > 1:
        best_response = float(np.max(angle_response))
        median_response = float(np.median(valid_responses))
        if median_response > 0:
            angle_sharpness = min(best_response / median_response, 10.0) / 10.0
        else:
            angle_sharpness = 1.0 if best_response > 0 else 0.0
    else:
        angle_sharpness = 0.0

    confidence = prominence_ratio * 0.6 + angle_sharpness * 0.4

    result = FFTResult(
        angle_degrees=round(best_angle, 2),
        spacing_meters=round(spacing_m, 3),
        spacing_pixels=round(spacing_px, 2),
        row_count=row_count,
        confidence=round(confidence, 4),
        angle_response=angle_response,
        best_fft_spectrum=fft_spectrum,
        peak_frequency=round(peak_freq, 6),
    )
    logger.info(
        "detect (FFT): angle=%.1f deg, spacing=%.2f m (%.1f px), rows=%d, "
        "freq=%.4f c/px, confidence=%.3f",
        result.angle_degrees,
        result.spacing_meters,
        result.spacing_pixels,
        result.row_count,
        result.peak_frequency,
        result.confidence,
    )
    return result
