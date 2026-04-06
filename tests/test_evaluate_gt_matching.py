"""Tests for evaluate_gt matching logic — no cv2 dependency.

Requires scipy for the Hungarian algorithm used by match_rows.
"""

from __future__ import annotations

import pytest

try:
    from evaluate_gt import match_rows, _compute_f1
except ImportError:
    pytest.skip("scipy or other evaluate_gt dependencies not available", allow_module_level=True)


class TestMatchRows:
    """Tests for bipartite row matching."""

    def test_perfect_alignment(self):
        """All GT rows match all detected rows exactly."""
        gt = [10.0, 20.0, 30.0, 40.0, 50.0]
        det = [10.0, 20.0, 30.0, 40.0, 50.0]
        mpp = 0.1

        matched, unmatched_gt, unmatched_det, distances = match_rows(gt, det, mpp)

        assert len(matched) == 5
        assert len(unmatched_gt) == 0
        assert len(unmatched_det) == 0
        assert all(d == 0.0 for d in distances)

    def test_partial_overlap(self):
        """Some GT rows match, some don't."""
        gt = [10.0, 20.0, 30.0, 40.0, 50.0]
        det = [10.5, 20.5, 60.0]  # first two close, third far away
        mpp = 0.1

        matched, unmatched_gt, unmatched_det, distances = match_rows(gt, det, mpp)

        # With spacing ~10px and threshold 0.4x = 4px, 0.5px distance matches
        assert len(matched) == 2
        assert len(unmatched_gt) == 3
        assert len(unmatched_det) == 1

    def test_zero_gt_rows(self):
        """No GT rows — everything is unmatched detection."""
        gt = []
        det = [10.0, 20.0, 30.0]
        mpp = 0.1

        matched, unmatched_gt, unmatched_det, distances = match_rows(gt, det, mpp)

        assert len(matched) == 0
        assert len(unmatched_gt) == 0
        assert len(unmatched_det) == 3
        assert len(distances) == 0

    def test_zero_det_rows(self):
        """No detected rows — everything is unmatched GT."""
        gt = [10.0, 20.0, 30.0]
        det = []
        mpp = 0.1

        matched, unmatched_gt, unmatched_det, distances = match_rows(gt, det, mpp)

        assert len(matched) == 0
        assert len(unmatched_gt) == 3
        assert len(unmatched_det) == 0

    def test_single_row(self):
        """Single GT and single detected row."""
        gt = [25.0]
        det = [26.0]
        mpp = 0.1

        # Single row means median_spacing defaults to 25.0
        # Threshold = 0.4 * 25 = 10 — distance of 1.0 is well within
        matched, unmatched_gt, unmatched_det, distances = match_rows(gt, det, mpp)

        assert len(matched) == 1
        assert distances[0] == pytest.approx(1.0)

    def test_threshold_boundary(self):
        """Distance exactly at threshold edge."""
        gt = [0.0, 10.0, 20.0]
        det = [0.0, 10.0, 20.0 + 4.0]  # median spacing=10, threshold=0.4*10=4.0
        mpp = 0.1

        matched, unmatched_gt, unmatched_det, distances = match_rows(gt, det, mpp)

        # Distance of 4.0 is exactly at the threshold (<=), should match
        assert len(matched) == 3

    def test_beyond_threshold(self):
        """Distance just beyond threshold."""
        gt = [0.0, 10.0, 20.0]
        det = [0.0, 10.0, 20.0 + 4.1]  # just over 4.0 threshold
        mpp = 0.1

        matched, unmatched_gt, unmatched_det, distances = match_rows(gt, det, mpp)

        assert len(matched) == 2
        assert len(unmatched_gt) == 1
        assert len(unmatched_det) == 1


class TestComputeF1:
    """Tests for the F1 computation at different thresholds."""

    def test_perfect_f1(self):
        gt = [10.0, 20.0, 30.0]
        det = [10.0, 20.0, 30.0]
        assert _compute_f1(gt, det, 0.1, 0.4) == pytest.approx(1.0)

    def test_zero_f1_no_overlap(self):
        gt = [10.0, 20.0, 30.0]
        det = [100.0, 110.0, 120.0]  # far away
        assert _compute_f1(gt, det, 0.1, 0.4) == pytest.approx(0.0)

    def test_empty_inputs(self):
        assert _compute_f1([], [], 0.1, 0.4) == pytest.approx(0.0)
        assert _compute_f1([10.0], [], 0.1, 0.4) == pytest.approx(0.0)
        assert _compute_f1([], [10.0], 0.1, 0.4) == pytest.approx(0.0)

    def test_stricter_threshold_lowers_f1(self):
        """Stricter threshold should give same or lower F1."""
        gt = [0.0, 10.0, 20.0]
        det = [0.5, 10.5, 20.5]  # slightly offset
        f1_loose = _compute_f1(gt, det, 0.1, 0.4)
        f1_strict = _compute_f1(gt, det, 0.1, 0.1)
        assert f1_loose >= f1_strict
