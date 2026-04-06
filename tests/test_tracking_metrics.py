"""Tests for tracking/metrics.py — bootstrap, ECE, learning curve fit."""

from __future__ import annotations

import numpy as np
import pytest

from tracking.metrics import (
    bootstrap_confidence_interval,
    expected_calibration_error,
    fit_learning_curve,
    paired_bootstrap_test,
)


class TestBootstrapCI:
    def test_normal_input(self):
        np.random.seed(42)
        values = [0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93, 0.89]
        ci = bootstrap_confidence_interval(values, n_iterations=5000)
        assert ci is not None
        lower, upper = ci
        assert lower < upper
        assert lower > 0.8
        assert upper < 1.0

    def test_empty_returns_none(self):
        assert bootstrap_confidence_interval([]) is None

    def test_single_value(self):
        ci = bootstrap_confidence_interval([0.5], n_iterations=1000)
        assert ci is not None
        lower, upper = ci
        # Single value — CI collapses to the value itself
        assert lower == pytest.approx(0.5)
        assert upper == pytest.approx(0.5)

    def test_ci_contains_mean(self):
        np.random.seed(123)
        values = list(np.random.normal(0.9, 0.05, 20))
        ci = bootstrap_confidence_interval(values, n_iterations=5000)
        mean = np.mean(values)
        assert ci[0] <= mean <= ci[1]


class TestPairedBootstrapTest:
    def test_identical_arrays_not_significant(self):
        np.random.seed(42)
        values = [0.9, 0.85, 0.92, 0.88, 0.91]
        result = paired_bootstrap_test(values, values, n_iterations=5000)
        assert result is not None
        assert result["mean_diff"] == pytest.approx(0.0)
        assert not result["significant"]

    def test_clearly_different_is_significant(self):
        np.random.seed(42)
        old = [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.67, 0.70]
        new = [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87, 0.90]
        result = paired_bootstrap_test(old, new, n_iterations=5000)
        assert result is not None
        assert result["mean_diff"] > 0.15
        assert result["significant"]
        assert result["n_blocks"] == 8

    def test_empty_returns_none(self):
        assert paired_bootstrap_test([], []) is None

    def test_mismatched_lengths_returns_none(self):
        assert paired_bootstrap_test([1.0, 2.0], [1.0]) is None


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        # Confidence equals accuracy in each bin
        confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # correctness roughly matches confidence
        correct = [False, False, False, False, True, True, True, True, True, True]
        ece = expected_calibration_error(confs, correct, n_bins=5)
        assert ece is not None
        assert 0 <= ece <= 1.0

    def test_none_input(self):
        assert expected_calibration_error(None, None) is None

    def test_empty_input(self):
        assert expected_calibration_error([], []) is None

    def test_overconfident_model(self):
        # All predictions at 0.95 confidence, but only 50% correct
        confs = [0.95] * 100
        correct = [True] * 50 + [False] * 50
        ece = expected_calibration_error(confs, correct, n_bins=10)
        assert ece is not None
        assert ece > 0.4  # large calibration error


class TestFitLearningCurve:
    def test_too_few_points_returns_none(self):
        assert fit_learning_curve([5, 10, 15], [0.7, 0.8, 0.85]) is None

    def test_valid_fit(self):
        # Simulate diminishing returns
        sizes = [5, 10, 15, 20, 25, 30]
        scores = [0.70, 0.82, 0.87, 0.90, 0.91, 0.92]
        result = fit_learning_curve(sizes, scores)
        # fit_learning_curve returns None if scipy is unavailable
        try:
            import scipy  # noqa: F401
        except ImportError:
            assert result is None, "Should return None without scipy"
            return
        assert result is not None
        assert "asymptote" in result
        assert result["asymptote"] > 0.9  # predicted ceiling
        assert result["asymptote"] < 1.0

    def test_empty_input_returns_none(self):
        assert fit_learning_curve([], []) is None
