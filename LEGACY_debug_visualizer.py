"""
Debug visualization for the CV row detection pipeline.

Generates annotated images and plots showing each processing stage,
making it easy to diagnose detection issues and compare results
against ground truth.

Uses matplotlib (Agg backend) for plots and cv2 for image overlays.
"""

import os
import logging
import math

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image overlay helpers
# ---------------------------------------------------------------------------


def draw_detected_lines(
    image_bgr: np.ndarray,
    lines: list[tuple] | np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Overlay detected lines on a copy of the image.

    Args:
        image_bgr: Source image in BGR format.
        lines: Iterable of (x1, y1, x2, y2) line segments.
        color: BGR color tuple for lines.
        thickness: Line thickness in pixels.

    Returns:
        Copy of image_bgr with lines drawn on it.
    """
    canvas = image_bgr.copy()
    if lines is None:
        return canvas

    line_array = np.array(lines) if not isinstance(lines, np.ndarray) else lines
    if line_array.ndim == 1:
        # Single line
        line_array = line_array.reshape(1, -1)

    for line in line_array:
        if len(line) >= 4:
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

    logger.debug("Drew %d lines on image", len(line_array))
    return canvas


def draw_spacing_markers(
    image_bgr: np.ndarray,
    angle_deg: float,
    spacing_px: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Draw perpendicular measurement lines showing detected row spacing.

    Draws evenly spaced lines perpendicular to the detected row direction,
    with the spacing between them representing the detected row spacing.
    Lines are clipped to the mask region.

    Args:
        image_bgr: Source image in BGR format.
        angle_deg: Detected row angle in degrees (0 = horizontal).
        spacing_px: Detected spacing in pixels.
        mask: Binary mask (uint8, 0 or 255) defining the region of interest.

    Returns:
        Copy of image_bgr with spacing markers drawn.
    """
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    if spacing_px < 2.0:
        logger.warning("Spacing too small (%.1f px) to draw markers", spacing_px)
        return canvas

    # Direction perpendicular to rows
    perp_rad = math.radians(angle_deg + 90.0)
    dx = math.cos(perp_rad)
    dy = math.sin(perp_rad)

    # Row direction (along the rows)
    row_rad = math.radians(angle_deg)
    rdx = math.cos(row_rad)
    rdy = math.sin(row_rad)

    # Find mask centroid as starting point
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return canvas
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # Draw marker lines along the perpendicular direction from center
    diag = math.sqrt(w * w + h * h)
    max_lines = int(diag / spacing_px) + 2
    line_half_len = diag / 2.0

    color_marker = (255, 0, 255)   # Magenta in BGR — visible on green vegetation
    color_text = (255, 255, 255)

    # Scale line thickness with image size so markers stay visible on large images
    thickness = max(1, min(3, min(h, w) // 1000))

    logger.info(
        "draw_spacing_markers: angle=%.1f°, spacing=%.1f px, "
        "row_dir=(%.3f, %.3f), perp_dir=(%.3f, %.3f), thickness=%d",
        angle_deg, spacing_px, rdx, rdy, dx, dy, thickness,
    )

    count = 0
    for i in range(-max_lines, max_lines + 1):
        # Center of this marker line
        lx = cx + dx * spacing_px * i
        ly = cy + dy * spacing_px * i

        # Endpoints along the row direction
        x1 = int(lx - rdx * line_half_len)
        y1 = int(ly - rdy * line_half_len)
        x2 = int(lx + rdx * line_half_len)
        y2 = int(ly + rdy * line_half_len)

        # Check if the line center is within the mask
        lx_int, ly_int = int(round(lx)), int(round(ly))
        if 0 <= lx_int < w and 0 <= ly_int < h and mask[ly_int, lx_int] > 0:
            cv2.line(canvas, (x1, y1), (x2, y2), color_marker, thickness, cv2.LINE_AA)
            count += 1

    # Add spacing annotation
    text = f"Spacing: {spacing_px:.1f} px"
    font_scale = max(0.8, min(h, w) / 2000.0)
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color_text, 2, cv2.LINE_AA)

    logger.info("draw_spacing_markers: drew %d lines", count)
    return canvas


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------


def plot_angle_response(
    angle_response: np.ndarray,
    best_angle: float,
    output_path: str,
    angle_start: float = 0.0,
    angle_step: float = 0.5,
) -> None:
    """Plot FFT angle sweep response curve.

    X-axis: angle in degrees.
    Y-axis: FFT signal strength (sum of magnitudes along that angle).
    A vertical dashed line marks the detected peak angle.

    Args:
        angle_response: 1D array of signal strengths per angle.
        best_angle: The angle (degrees) with highest response.
        output_path: File path to save the plot.
        angle_start: Starting angle of the sweep (degrees).
        angle_step: Step between angles (degrees).
    """
    n = len(angle_response)
    angles = np.arange(angle_start, angle_start + n * angle_step, angle_step)[:n]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(angles, angle_response, color='steelblue', linewidth=1.5)
    row_angle = (best_angle + 90.0) % 180.0
    ax.axvline(best_angle, color='red', linestyle='--', linewidth=1.5,
               label=f'Proj: {best_angle:.1f}° (Row: {row_angle:.1f}°)')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('FFT Signal Strength')
    ax.set_title('Angle Sweep — FFT Directional Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved angle response plot to %s", output_path)


def plot_fft_spectrum(
    spectrum: np.ndarray,
    peak_freq: float,
    spacing_meters: float,
    output_path: str,
) -> None:
    """Plot 1D FFT spectrum along the detected perpendicular direction.

    X-axis: spatial frequency (cycles/pixel).
    Y-axis: magnitude.
    The peak frequency is marked with a vertical line, and the
    corresponding row spacing in meters is annotated.

    Args:
        spectrum: 1D magnitude spectrum array.
        peak_freq: Detected peak frequency in cycles/pixel.
        spacing_meters: Detected spacing converted to meters.
        output_path: File path to save the plot.
    """
    n = len(spectrum)
    # Frequency axis: skip DC component, go to Nyquist
    freqs = np.arange(n) / n  # cycles per pixel (0 to ~0.5 for the useful half)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, spectrum, color='darkgreen', linewidth=1.0)
    ax.axvline(peak_freq, color='red', linestyle='--', linewidth=1.5,
               label=f'Peak: {peak_freq:.4f} cyc/px')
    ax.set_xlabel('Spatial Frequency (cycles/pixel)')
    ax.set_ylabel('Magnitude')
    ax.set_title('FFT Spectrum — Row Spacing Detection')

    # Annotate with spacing
    ax.annotate(
        f'{spacing_meters:.2f} m',
        xy=(peak_freq, spectrum[min(int(peak_freq * n), n - 1)] if peak_freq * n < n else 0),
        xytext=(peak_freq + 0.02, max(spectrum) * 0.8),
        fontsize=12, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red'),
    )

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)  # Only show up to Nyquist

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved FFT spectrum plot to %s", output_path)


def _save_fft2d_spectrum(fft2d_result, output_path: str) -> None:
    """Save log-magnitude spectrum with peak marker for 2D FFT result."""
    spectrum = fft2d_result.magnitude_spectrum
    peak_x, peak_y = fft2d_result.peak_position

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(spectrum, cmap='hot', aspect='auto')
    ax.plot(peak_x, peak_y, 'c+', markersize=20, markeredgewidth=2)
    ax.set_title(
        f"2D FFT Magnitude — angle={fft2d_result.angle_degrees:.1f}°, "
        f"spacing={fft2d_result.spacing_meters:.2f}m, "
        f"conf={fft2d_result.confidence:.2f}"
    )
    ax.set_xlabel("Frequency (x)")
    ax.set_ylabel("Frequency (y)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved 2D FFT spectrum to %s", output_path)


# ---------------------------------------------------------------------------
# Full pipeline debug output
# ---------------------------------------------------------------------------


def save_pipeline_debug(
    original_bgr: np.ndarray,
    vegetation: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    hough_result,  # HoughResult or None (duck typing to avoid circular import)
    fft_result,    # FFTResult or None
    block_name: str,
    output_dir: str,
    fft2d_result=None,  # FFT2DResult or None
    row_locator_result=None,  # RowLocatorResult or None
) -> None:
    """Save a complete set of debug images for one block.

    Files created:
        01_original.png    — Masked aerial image
        02_vegetation.png  — ExG vegetation index as heatmap
        03_edges.png       — Canny edge map
        04_hough_lines.png — Detected Hough lines overlaid on original (if available)
        05_fft_angle_response.png — Angle sweep plot (if available)
        06_fft_spectrum.png       — FFT spectrum plot (if available)
        07_comparison.png         — Side-by-side summary of all stages

    Args:
        original_bgr: The masked aerial image (BGR).
        vegetation: ExG vegetation index (grayscale uint8).
        edges: Canny edge map (grayscale uint8).
        mask: Binary polygon mask (uint8, 0 or 255).
        hough_result: Object with .dominant_lines, .angle_degrees, .spacing_pixels attrs, or None.
        fft_result: Object with .angle_degrees, .spacing_pixels, .angle_response,
                    .best_fft_spectrum, .peak_frequency attrs, or None.
        block_name: Human-readable block name for labeling.
        output_dir: Directory to write debug images into.
    """
    block_dir = os.path.join(output_dir, _safe_filename(block_name))
    os.makedirs(block_dir, exist_ok=True)
    logger.info("Saving debug images for '%s' to %s", block_name, block_dir)

    # 01 — Original masked image
    cv2.imwrite(os.path.join(block_dir, "01_original.png"), original_bgr)

    # 02 — Vegetation index as heatmap
    veg_color = cv2.applyColorMap(vegetation, cv2.COLORMAP_VIRIDIS)
    # Zero out areas outside mask
    veg_color = cv2.bitwise_and(veg_color, veg_color, mask=mask)
    cv2.imwrite(os.path.join(block_dir, "02_vegetation.png"), veg_color)

    # 03 — Canny edges
    cv2.imwrite(os.path.join(block_dir, "03_edges.png"), edges)

    # 04 — Hough lines overlay
    if hough_result is not None and hasattr(hough_result, 'dominant_lines') and hough_result.dominant_lines is not None:
        lines_img = draw_detected_lines(original_bgr, hough_result.dominant_lines, (0, 255, 0), 2)
        # Add angle/spacing annotation
        angle_text = f"Angle: {hough_result.angle_degrees:.1f} deg"
        spacing_text = f"Spacing: {hough_result.spacing_pixels:.1f} px"
        cv2.putText(lines_img, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lines_img, spacing_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(block_dir, "04_hough_lines.png"), lines_img)
    else:
        logger.debug("No Hough result — skipping 04_hough_lines.png")

    # 05 — FFT angle response plot
    if fft_result is not None and hasattr(fft_result, 'angle_response') and fft_result.angle_response is not None:
        # The angle_response curve is indexed by PROJECTION angle (0-180°).
        # fft_result.angle_degrees is the ROW angle (projection + 90°).
        # Convert back to projection angle for the marker on the plot.
        projection_angle = (fft_result.angle_degrees - 90.0) % 180.0
        plot_angle_response(
            fft_result.angle_response,
            projection_angle,
            os.path.join(block_dir, "05_fft_angle_response.png"),
        )
    else:
        logger.debug("No FFT angle response — skipping 05_fft_angle_response.png")

    # 06 — FFT spectrum plot
    if fft_result is not None and hasattr(fft_result, 'best_fft_spectrum') and fft_result.best_fft_spectrum is not None:
        spacing_m = getattr(fft_result, 'spacing_meters', 0.0)
        plot_fft_spectrum(
            fft_result.best_fft_spectrum,
            fft_result.peak_frequency,
            spacing_m,
            os.path.join(block_dir, "06_fft_spectrum.png"),
        )
    else:
        logger.debug("No FFT spectrum — skipping 06_fft_spectrum.png")

    # 08 — 2D FFT magnitude spectrum
    if fft2d_result is not None and hasattr(fft2d_result, 'magnitude_spectrum'):
        _save_fft2d_spectrum(
            fft2d_result,
            os.path.join(block_dir, "08_fft2d_magnitude.png"),
        )

    # 09 — 2D FFT detected angle overlay
    if fft2d_result is not None and hasattr(fft2d_result, 'angle_degrees'):
        spacing_overlay = draw_spacing_markers(
            original_bgr, fft2d_result.angle_degrees,
            fft2d_result.spacing_pixels, mask,
        )
        h_img = spacing_overlay.shape[0]
        font_scale = max(1.0, h_img / 2000.0)
        angle_text = f"2D FFT: {fft2d_result.angle_degrees:.1f} deg, {fft2d_result.spacing_meters:.2f}m, conf={fft2d_result.confidence:.2f}"
        cv2.putText(spacing_overlay, angle_text, (10, int(40 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(block_dir, "09_fft2d_overlay.png"), spacing_overlay)

    # 10-12 — Row locator debug images
    if row_locator_result is not None and hasattr(row_locator_result, 'rows'):
        _save_row_positions(
            original_bgr, row_locator_result, mask,
            os.path.join(block_dir, "10_row_positions.png"),
        )
        _save_spacing_histogram(
            row_locator_result, block_name,
            os.path.join(block_dir, "11_spacing_histogram.png"),
        )
        _save_perpendicular_profile(
            row_locator_result, block_name,
            os.path.join(block_dir, "12_perpendicular_profile.png"),
        )

    # 07 — Side-by-side comparison summary
    _save_comparison_image(
        original_bgr, vegetation, edges, mask,
        hough_result, fft_result,
        block_name, os.path.join(block_dir, "07_comparison.png"),
    )

    logger.info("Debug images saved for '%s'", block_name)


def _save_comparison_image(
    original_bgr: np.ndarray,
    vegetation: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    hough_result,
    fft_result,
    block_name: str,
    output_path: str,
) -> None:
    """Create a 2x2 or 2x3 comparison figure with matplotlib."""
    has_hough = hough_result is not None and hasattr(hough_result, 'dominant_lines')
    has_fft = fft_result is not None and hasattr(fft_result, 'angle_degrees')

    ncols = 3 if (has_hough or has_fft) else 2
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))
    fig.suptitle(f'Pipeline Summary — {block_name}', fontsize=16, fontweight='bold')

    # Row 1: Original, Vegetation heatmap, Edges
    axes[0, 0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original (masked)')
    axes[0, 0].axis('off')

    veg_color = cv2.applyColorMap(vegetation, cv2.COLORMAP_VIRIDIS)
    veg_color = cv2.bitwise_and(veg_color, veg_color, mask=mask)
    axes[0, 1].imshow(cv2.cvtColor(veg_color, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Vegetation (ExG)')
    axes[0, 1].axis('off')

    if ncols > 2:
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Canny Edges')
        axes[0, 2].axis('off')

    # Row 2: Results
    if has_hough and hough_result.dominant_lines is not None:
        lines_img = draw_detected_lines(original_bgr, hough_result.dominant_lines, (0, 255, 0), 2)
        axes[1, 0].imshow(cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Hough: {hough_result.angle_degrees:.1f}°, {hough_result.spacing_pixels:.1f}px')
    else:
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edges (no Hough)')
    axes[1, 0].axis('off')

    if has_fft:
        # Show spacing markers overlay
        spacing_img = draw_spacing_markers(
            original_bgr, fft_result.angle_degrees, fft_result.spacing_pixels, mask
        )
        axes[1, 1].imshow(cv2.cvtColor(spacing_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'FFT: {fft_result.angle_degrees:.1f}°, {fft_result.spacing_pixels:.1f}px')
    else:
        axes[1, 1].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('No FFT result')
    axes[1, 1].axis('off')

    if ncols > 2:
        # Summary text panel
        axes[1, 2].axis('off')
        summary_lines = [f"Block: {block_name}"]
        if has_hough:
            summary_lines.append(f"Hough angle: {hough_result.angle_degrees:.1f}°")
            summary_lines.append(f"Hough spacing: {hough_result.spacing_pixels:.1f} px")
            if hasattr(hough_result, 'confidence'):
                summary_lines.append(f"Hough confidence: {hough_result.confidence:.2f}")
        if has_fft:
            summary_lines.append(f"FFT angle: {fft_result.angle_degrees:.1f}°")
            summary_lines.append(f"FFT spacing: {fft_result.spacing_pixels:.1f} px")
            if hasattr(fft_result, 'spacing_meters'):
                summary_lines.append(f"FFT spacing: {fft_result.spacing_meters:.2f} m")
            if hasattr(fft_result, 'confidence'):
                summary_lines.append(f"FFT confidence: {fft_result.confidence:.2f}")
        axes[1, 2].text(
            0.1, 0.9, '\n'.join(summary_lines),
            transform=axes[1, 2].transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.debug("Saved comparison image to %s", output_path)


# ---------------------------------------------------------------------------
# Results comparison
# ---------------------------------------------------------------------------


def create_comparison_summary(results: list[dict], output_path: str) -> None:
    """Create a markdown results table comparing detected vs ground truth.

    Each dict in results should have keys:
        block_name, approach, detected_angle, detected_spacing_m,
        detected_row_count, confidence, ground_truth_spacing,
        ground_truth_orientation, ground_truth_row_count,
        spacing_error_pct, angle_error_deg, processing_time_s

    Args:
        results: List of result dicts (from DetectionResult.asdict()).
        output_path: File path for the markdown output.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    lines = [
        "# CV Row Detection — Results Summary",
        "",
        f"**Blocks tested:** {len(set(r.get('block_name', '?') for r in results))}",
        f"**Total runs:** {len(results)}",
        "",
    ]

    # Summary statistics
    spacing_errors = [r['spacing_error_pct'] for r in results if r.get('spacing_error_pct') is not None]
    angle_errors = [r['angle_error_deg'] for r in results if r.get('angle_error_deg') is not None]

    if spacing_errors:
        lines.append(f"**Mean spacing error:** {sum(spacing_errors) / len(spacing_errors):.1f}%")
        lines.append(f"**Max spacing error:** {max(spacing_errors):.1f}%")
    if angle_errors:
        lines.append(f"**Mean angle error:** {sum(angle_errors) / len(angle_errors):.1f}°")
        lines.append(f"**Max angle error:** {max(angle_errors):.1f}°")

    # Success rate (within thresholds)
    if spacing_errors and angle_errors:
        successes = sum(
            1 for r in results
            if r.get('spacing_error_pct') is not None
            and r.get('angle_error_deg') is not None
            and r['spacing_error_pct'] <= 15.0  # ~0.3m on a 2m spacing
            and r['angle_error_deg'] <= 5.0
        )
        total_with_gt = sum(
            1 for r in results
            if r.get('spacing_error_pct') is not None
        )
        if total_with_gt > 0:
            lines.append(f"**Success rate:** {successes}/{total_with_gt} "
                         f"({100 * successes / total_with_gt:.0f}%) "
                         f"within ±5° angle and ±15% spacing")

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    # Table header
    lines.append(
        "| Block | Vineyard | Approach | Det. Angle | GT Orient. | Angle Err | "
        "Det. Spacing (m) | GT Spacing (m) | Spacing Err | "
        "Det. Rows | GT Rows | Confidence | Time (s) |"
    )
    lines.append(
        "|-------|----------|----------|------------|------------|-----------|"
        "-------------------|----------------|-------------|"
        "-----------|---------|------------|----------|"
    )

    for r in results:
        angle_err = f"{r['angle_error_deg']:.1f}°" if r.get('angle_error_deg') is not None else "—"
        spacing_err = f"{r['spacing_error_pct']:.1f}%" if r.get('spacing_error_pct') is not None else "—"
        gt_spacing = f"{r['ground_truth_spacing']:.2f}" if r.get('ground_truth_spacing') is not None else "—"
        gt_orient = r.get('ground_truth_orientation') or "—"
        gt_rows = str(r.get('ground_truth_row_count', '')) or "—"

        # Flag results outside success thresholds
        angle_flag = " ⚠" if r.get('angle_error_deg') is not None and r['angle_error_deg'] > 5.0 else ""
        spacing_flag = " ⚠" if r.get('spacing_error_pct') is not None and r['spacing_error_pct'] > 15.0 else ""

        lines.append(
            f"| {r.get('block_name', '?')} "
            f"| {r.get('vineyard_name', '?')} "
            f"| {r.get('approach', '?')} "
            f"| {r.get('detected_angle', 0):.1f}° "
            f"| {gt_orient} "
            f"| {angle_err}{angle_flag} "
            f"| {r.get('detected_spacing_m', 0):.2f} "
            f"| {gt_spacing} "
            f"| {spacing_err}{spacing_flag} "
            f"| {r.get('detected_row_count', 0)} "
            f"| {gt_rows} "
            f"| {r.get('confidence', 0):.2f} "
            f"| {r.get('processing_time_s', 0):.1f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by cv-row-detection prototype*")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info("Saved comparison summary to %s", output_path)


# ---------------------------------------------------------------------------
# Row locator visualizations (images 10-12)
# ---------------------------------------------------------------------------


def _save_row_positions(
    original_bgr: np.ndarray,
    result,
    mask: np.ndarray,
    output_path: str,
) -> None:
    """Save overlay with color-coded row center lines (image 10)."""
    canvas = original_bgr.copy()
    h, w = canvas.shape[:2]
    mean_sp = result.mean_spacing_m

    for row in result.rows:
        pts = row.center_positions
        if len(pts) < 2:
            continue

        # Color by spacing deviation
        if row.spacing_to_previous is None:
            color = (255, 255, 255)  # White for first row
        else:
            pct_dev = abs(row.spacing_to_previous - mean_sp) / mean_sp * 100 if mean_sp > 0 else 0
            if pct_dev <= 5:
                color = (0, 255, 0)    # Green — nominal
            elif pct_dev <= 15:
                color = (0, 255, 255)  # Yellow — slightly off
            else:
                color = (0, 0, 255)    # Red — very different

        # Draw polyline through center positions
        int_pts = [(int(round(x)), int(round(y))) for x, y in pts]
        for i in range(len(int_pts) - 1):
            cv2.line(canvas, int_pts[i], int_pts[i + 1], color, 1, cv2.LINE_AA)

        # Row index label at midpoint
        mid_idx = len(int_pts) // 2
        mx, my = int_pts[mid_idx]
        if 0 <= mx < w and 0 <= my < h:
            font_scale = max(0.3, min(h, w) / 5000.0)
            cv2.putText(canvas, str(row.row_index), (mx + 3, my - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    # Title annotation
    font_scale = max(0.7, min(h, w) / 2500.0)
    text = f"Row Locator: {result.total_row_count} rows, mean={result.mean_spacing_m:.2f}m"
    cv2.putText(canvas, text, (10, int(30 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, int(30 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    logger.info("Saved row positions overlay to %s", output_path)


def _save_spacing_histogram(result, block_name: str, output_path: str) -> None:
    """Save histogram of per-row spacings (image 11)."""
    spacings = [r.spacing_to_previous for r in result.rows if r.spacing_to_previous is not None]
    if not spacings:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(spacings, bins=max(10, len(spacings) // 3), color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(result.mean_spacing_m, color='red', linestyle='-',
               linewidth=2, label=f'Row locator mean: {result.mean_spacing_m:.2f}m')

    ax.set_xlabel('Row spacing (m)')
    ax.set_ylabel('Count')
    ax.set_title(
        f'{block_name} — {result.total_row_count} rows, '
        f'mean={result.mean_spacing_m:.2f}m (\u03c3={result.spacing_std_m:.3f}m)'
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved spacing histogram to %s", output_path)


def _save_perpendicular_profile(result, block_name: str, output_path: str) -> None:
    """Save the 1D perpendicular intensity profile with peak markers (image 12)."""
    profile = result.perpendicular_profile
    positions = result.profile_perp_positions
    peaks = result.peak_perp_positions

    if len(profile) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(positions, profile, color='green', linewidth=0.5, alpha=0.8)

    # Mark detected peaks
    for pk in peaks:
        idx = np.argmin(np.abs(positions - pk))
        if 0 <= idx < len(profile):
            ax.plot(pk, profile[idx], 'rv', markersize=4)

    ax.set_xlabel('Perpendicular position (px)')
    ax.set_ylabel('Mean ExG intensity')
    ax.set_title(f'{block_name} — Perpendicular profile ({result.total_row_count} peaks detected)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved perpendicular profile to %s", output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    """Convert a block name to a filesystem-safe directory name."""
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name).strip('_')
