#!/usr/bin/env python3
"""
Synthetic test for 2D FFT angle conversion.

Creates images with lines at known angles on non-square canvases,
runs the 2D FFT detector, and verifies the detected angle matches.
"""

import math
import sys

import cv2
import numpy as np

# Import the core detection logic directly
from fft2d_detector import detect
from image_preprocessor import PreprocessResult


def create_synthetic_lines(w: int, h: int, angle_deg: float, spacing_px: float = 20.0) -> np.ndarray:
    """Create a synthetic image with parallel lines at a given angle.

    Args:
        w: Image width in pixels.
        h: Image height in pixels.
        angle_deg: Line angle in image coordinates (0=horizontal, 90=vertical).
        spacing_px: Spacing between lines in pixels.

    Returns:
        Grayscale uint8 image with white lines on black background.
    """
    img = np.zeros((h, w), dtype=np.uint8)
    angle_rad = math.radians(angle_deg)

    # Row direction vector
    rdx = math.cos(angle_rad)
    rdy = math.sin(angle_rad)

    # Perpendicular direction (spacing direction)
    pdx = -rdy  # -sin(angle)
    pdy = rdx   #  cos(angle)

    # Draw many parallel lines
    diag = math.sqrt(w * w + h * h)
    max_lines = int(diag / spacing_px) + 2
    line_len = int(diag)

    for i in range(-max_lines, max_lines + 1):
        # Center of this line
        cx = w / 2.0 + pdx * spacing_px * i
        cy = h / 2.0 + pdy * spacing_px * i

        # Endpoints along the row direction
        x1 = int(cx - rdx * line_len)
        y1 = int(cy - rdy * line_len)
        x2 = int(cx + rdx * line_len)
        y2 = int(cy + rdy * line_len)

        cv2.line(img, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)

    return img


def test_angle(w: int, h: int, angle_deg: float, spacing_px: float = 20.0) -> tuple[float, float]:
    """Test the 2D FFT detector on a synthetic image.

    Returns:
        (detected_angle, error) in degrees.
    """
    img = create_synthetic_lines(w, h, angle_deg, spacing_px)

    # Create a full mask (entire image is valid)
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # Build a minimal PreprocessResult
    prep = PreprocessResult(
        grayscale=img,
        enhanced=img,
        vegetation=img,
        edges=np.zeros_like(img),
        mask=mask,
        use_vegetation=True,
    )

    # Run detector (lat/zoom don't matter for angle, just spacing conversion)
    result = detect(prep, lat=-41.5, zoom=20)

    if result is None:
        return float('nan'), float('nan')

    detected = result.angle_degrees

    # Compute angular error (handle 180° wrap)
    diff = abs(detected - angle_deg)
    if diff > 90:
        diff = 180 - diff

    return detected, diff


def main():
    test_cases = [
        # (width, height, angle, description)
        (1000, 500, 10.0, "10° on 1000x500"),
        (1000, 500, 45.0, "45° on 1000x500"),
        (1000, 500, 85.0, "85° on 1000x500"),
        (1000, 500, 93.0, "93° on 1000x500"),
        (500, 1000, 10.0, "10° on 500x1000"),
        (500, 1000, 45.0, "45° on 500x1000"),
        (500, 1000, 85.0, "85° on 500x1000"),
        (500, 1000, 93.0, "93° on 500x1000"),
        (1000, 1000, 45.0, "45° on 1000x1000 (square)"),
        (1000, 1000, 93.0, "93° on 1000x1000 (square)"),
        (3072, 5888, 93.5, "93.5° on 3072x5888 (Block C size)"),
    ]

    print(f"{'Test':<35} {'Expected':>8} {'Detected':>8} {'Error':>8} {'Status':>8}")
    print("-" * 75)

    all_pass = True
    for w, h, angle, desc in test_cases:
        detected, error = test_angle(w, h, angle, spacing_px=20.0)
        status = "PASS" if error <= 0.5 else "FAIL"
        if error > 0.5:
            all_pass = False
        print(f"{desc:<35} {angle:>8.1f} {detected:>8.1f} {error:>8.2f} {status:>8}")

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
