import numpy as np
import pytest

from fifa_value.eval.metrics import (
    compute_interval_metrics,
    compute_regression_metrics,
    paired_bootstrap_ci,
)


def test_perfect_regression_metrics():
    y_true = np.array([100000.0, 500000.0, 2000000.0, 50000000.0])
    y_pred = y_true.copy()
    m = compute_regression_metrics(y_true, y_pred)

    assert m["rmsle"] == pytest.approx(0.0, abs=1e-6)
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["median_ae"] == pytest.approx(0.0, abs=1e-6)
    assert m["within_10_pct"] == 100.0
    assert m["within_25_pct"] == 100.0
    assert m["spearman_rho"] == pytest.approx(1.0, abs=1e-6)


def test_interval_coverage_and_winkler():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([5.0, 15.0, 25.0, 35.0])
    upper = np.array([15.0, 25.0, 35.0, 45.0])
    im = compute_interval_metrics(y_true, lower, upper, nominal_coverage=0.90)

    assert im["coverage"] == 1.0  # all inside
    assert im["mean_width_eur"] == 10.0
    assert im["winkler_score_eur"] == 10.0  # no penalty


def test_bootstrap_difference():
    y_true = np.array([1000.0, 2000.0, 5000.0, 10000.0] * 20)
    # Model A is perfect
    preds_a = y_true.copy()
    # Model B has error
    preds_b = y_true * 1.5
    diff_mean, _ci_l, ci_u = paired_bootstrap_ci(y_true, preds_a, preds_b, n_bootstraps=100)

    # RMSLE(A) - RMSLE(B) should be strictly negative
    assert diff_mean < 0
    assert ci_u < 0
