"""Unit tests for statistical metrics (RMSE, MAE, R², Bias)."""

import numpy as np
import pytest
from src.analytics.statistical_metrics import compute_metrics


def test_compute_metrics_perfect():
    y_true = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    y_pred = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    m = compute_metrics(y_true, y_pred)
    assert m.rmse == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)
    assert m.bias == pytest.approx(0.0)
    assert m.n_samples == 5


def test_compute_metrics_known_error():
    y_true = np.array([10.0, 20.0, 30.0])
    # Constant offset +2.0
    y_pred = np.array([12.0, 22.0, 32.0])

    m = compute_metrics(y_true, y_pred)
    assert m.rmse == pytest.approx(2.0)
    assert m.mae == pytest.approx(2.0)
    assert m.bias == pytest.approx(2.0)
    assert m.r2 == pytest.approx(0.94)
    assert m.max_abs_diff == pytest.approx(2.0)
    assert m.min_abs_diff == pytest.approx(2.0)



def test_empty_metrics():
    m = compute_metrics([], [])
    assert m.n_samples == 0
    assert m.rmse == 0.0
