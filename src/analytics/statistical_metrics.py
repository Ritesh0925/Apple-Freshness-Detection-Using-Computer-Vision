"""Statistical assessment metrics according to Equations 5 and 6 of the paper."""

from typing import Union, Sequence
import numpy as np
from sklearn.metrics import r2_score

from src.core.types import StatisticalMetrics


def compute_metrics(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]]
) -> StatisticalMetrics:
    """Computes RMSE, MAE, R², Bias, Min/Max differences between TC (y_true) and IR (y_pred).

    Args:
        y_true: Thermocouple ground-truth readings y_i.
        y_pred: Sensor system temperature readings y_hat_i.

    Returns:
        StatisticalMetrics dataclass instance.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    # Filter any NaN or infinite values
    valid_mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid_mask]
    yp = yp[valid_mask]

    n = len(yt)
    if n == 0:
        return StatisticalMetrics(
            rmse=0.0, mae=0.0, r2=0.0, n_samples=0,
            bias=0.0, max_abs_diff=0.0, min_abs_diff=0.0
        )

    # Equation 5: RMSE = sqrt( 1/n * sum((y_i - y_hat_i)^2) )
    diff = yt - yp
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Equation 6: MAE = 1/n * sum(|y_i - y_hat_i|)
    mae = float(np.mean(np.abs(diff)))

    # R2 coefficient of determination
    if len(yt) > 1 and np.var(yt) > 1e-12:
        r2 = float(r2_score(yt, yp))
    else:
        r2 = 1.0

    # Mean signed bias (y_pred - y_true)
    bias = float(np.mean(yp - yt))

    max_diff = float(np.max(np.abs(diff)))
    min_diff = float(np.min(np.abs(diff)))

    return StatisticalMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        n_samples=n,
        bias=bias,
        max_abs_diff=max_diff,
        min_abs_diff=min_diff
    )
