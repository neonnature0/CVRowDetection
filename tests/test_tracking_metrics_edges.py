import numpy as np

from tracking.metrics import (
    bootstrap_confidence_interval,
    expected_calibration_error,
    paired_bootstrap_test,
)


def test_bootstrap_ci_none_for_empty_values():
    assert bootstrap_confidence_interval([]) is None


def test_bootstrap_ci_single_value_is_exact(monkeypatch):
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda: np.random.Generator(np.random.PCG64(1234)),
    )
    ci = bootstrap_confidence_interval([0.75], n_iterations=200, confidence=0.95)
    assert ci == (0.75, 0.75)


def test_paired_bootstrap_returns_none_for_mismatch_lengths():
    assert paired_bootstrap_test([0.1, 0.2], [0.3]) is None


def test_paired_bootstrap_identical_series_is_not_significant(monkeypatch):
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda: np.random.Generator(np.random.PCG64(7)),
    )
    result = paired_bootstrap_test([0.2, 0.4, 0.6], [0.2, 0.4, 0.6], n_iterations=300)

    assert result is not None
    assert result["mean_diff"] == 0.0
    assert result["ci_lower"] == 0.0
    assert result["ci_upper"] == 0.0
    assert result["significant"] is False
    assert result["n_blocks"] == 3


def test_expected_calibration_error_handles_right_edge_bin():
    ece = expected_calibration_error(
        confidences=[0.0, 0.5, 1.0],
        correctness=[False, True, True],
        n_bins=2,
    )

    # Bin [0.0,0.5): (0.0,0.0) gap=0 weighted 1/3
    # Bin [0.5,1.0]: (0.75 conf,1.0 acc) gap=0.25 weighted 2/3
    assert ece == 0.166667


def test_expected_calibration_error_returns_none_on_length_mismatch():
    assert expected_calibration_error([0.1, 0.2], [True]) is None
