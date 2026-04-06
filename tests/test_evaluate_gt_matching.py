import sys
import types

import pytest

sys.modules.setdefault("cv2", types.SimpleNamespace())

from evaluate_gt import _compute_f1, match_rows


def test_match_rows_assigns_pairs_and_reports_unmatched():
    gt_perps = [0.0, 10.0, 20.0]
    det_perps = [0.8, 10.5, 50.0]

    matched, unmatched_gt, unmatched_det, dists = match_rows(
        gt_perps,
        det_perps,
        mpp=0.1,
        match_threshold_factor=0.2,
    )

    assert matched == [(0, 0), (1, 1)]
    assert unmatched_gt == [2]
    assert unmatched_det == [2]
    assert dists == pytest.approx([0.8, 0.5])


def test_match_rows_no_match_when_distances_exceed_threshold():
    gt_perps = [0.0, 10.0]
    det_perps = [6.0, 18.0]  # spacing=10, threshold=2

    matched, unmatched_gt, unmatched_det, dists = match_rows(
        gt_perps,
        det_perps,
        mpp=0.1,
        match_threshold_factor=0.2,
    )

    assert matched == []
    assert unmatched_gt == [0, 1]
    assert unmatched_det == [0, 1]
    assert dists == []


def test_compute_f1_handles_no_detection_or_no_gt():
    assert _compute_f1([0.0, 10.0], [], mpp=0.1, threshold_factor=0.4) == 0.0
    assert _compute_f1([], [0.0, 10.0], mpp=0.1, threshold_factor=0.4) == 0.0


def test_compute_f1_zero_when_no_pairs_match():
    f1 = _compute_f1([0.0, 10.0], [100.0, 110.0], mpp=0.1, threshold_factor=0.1)
    assert f1 == 0.0
