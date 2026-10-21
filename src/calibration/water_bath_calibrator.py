"""Water-bath temperature calibration for thermal infrared camera module.

Implements Sections 2.3.1 and 3.1, Figure 8, and Table 3 of Canatan et al. (2025):
- Evaluates 0 - 100 °C water heating experiment against Type-T thermocouple reference.
- Analyzes piecewise temperature intervals (0-53 °C vs. 53-100 °C).
- Fits linear and piecewise calibration models to correct IR low-temperature and high-temperature bias.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.core.types import StatisticalMetrics, CalibrationResult
from src.analytics.statistical_metrics import compute_metrics


class WaterBathCalibrator:
    """Manages thermal camera calibration using the water heating experiment methodology."""

    def __init__(self, split_temp_c: float = 53.0):
        """
        Args:
            split_temp_c: Boundary temperature (53.0 °C in paper) distinguishing
                          low-temperature regime from high-temperature convective regime.
        """
        self.split_temp_c = split_temp_c
        self.model_full: Optional[LinearRegression] = None
        self.model_low: Optional[LinearRegression] = None
        self.model_high: Optional[LinearRegression] = None
        self.results_summary: Dict[str, CalibrationResult] = {}

    def fit(
        self,
        thermocouple_temps: np.ndarray,
        thermal_camera_temps: np.ndarray
    ) -> Dict[str, CalibrationResult]:
        """Fits calibration regression models and computes statistics.

        Args:
            thermocouple_temps: Ground-truth reference temperature y_i (°C).
            thermal_camera_temps: Uncalibrated IR sensor readings y_hat_i (°C).
        """
        y = np.asarray(thermocouple_temps, dtype=np.float64)
        x = np.asarray(thermal_camera_temps, dtype=np.float64)

        if len(y) != len(x):
            raise ValueError(f"Length mismatch: {len(y)} vs {len(x)}")

        # 1. Full Range (0 - 100 °C)
        m_full = LinearRegression().fit(x.reshape(-1, 1), y)
        self.model_full = m_full
        pred_full = m_full.predict(x.reshape(-1, 1))
        # Metrics before calibration vs after
        raw_metrics_full = compute_metrics(y, x)
        self.results_summary["0-100 C"] = CalibrationResult(
            temperature_range="0 - 100 °C",
            slope=float(m_full.coef_[0]),
            intercept=float(m_full.intercept_),
            metrics=raw_metrics_full
        )

        # 2. Low Range (0 - 53 °C)
        mask_low = (y >= 0.0) & (y <= self.split_temp_c)
        if np.sum(mask_low) > 5:
            x_low, y_low = x[mask_low], y[mask_low]
            m_low = LinearRegression().fit(x_low.reshape(-1, 1), y_low)
            self.model_low = m_low
            raw_metrics_low = compute_metrics(y_low, x_low)
            self.results_summary["0-53 C"] = CalibrationResult(
                temperature_range="0 - 53 °C",
                slope=float(m_low.coef_[0]),
                intercept=float(m_low.intercept_),
                metrics=raw_metrics_low
            )

        # 3. High Range (53 - 100 °C)
        mask_high = y > self.split_temp_c
        if np.sum(mask_high) > 5:
            x_high, y_high = x[mask_high], y[mask_high]
            m_high = LinearRegression().fit(x_high.reshape(-1, 1), y_high)
            self.model_high = m_high
            raw_metrics_high = compute_metrics(y_high, x_high)
            self.results_summary["53-100 C"] = CalibrationResult(
                temperature_range="53 - 100 °C",
                slope=float(m_high.coef_[0]),
                intercept=float(m_high.intercept_),
                metrics=raw_metrics_high
            )

        return self.results_summary

    def calibrate(self, raw_ir_temp_c: float, piecewise: bool = True) -> float:
        """Applies calibrated correction to a raw thermal camera reading.

        Args:
            raw_ir_temp_c: Measured uncalibrated IR surface temperature in °C.
            piecewise: If True, uses the appropriate low/high range model.
        """
        if piecewise and self.model_low is not None and self.model_high is not None:
            if raw_ir_temp_c <= self.split_temp_c:
                return float(self.model_low.predict([[raw_ir_temp_c]])[0])
            else:
                return float(self.model_high.predict([[raw_ir_temp_c]])[0])
        elif self.model_full is not None:
            return float(self.model_full.predict([[raw_ir_temp_c]])[0])
        return raw_ir_temp_c

    def to_dataframe(self) -> pd.DataFrame:
        """Exports validation summary matching Table 3 of the paper."""
        records = []
        for range_key, res in self.results_summary.items():
            records.append({
                "Temperature Range": res.temperature_range,
                "RMSE": round(res.metrics.rmse, 3),
                "MAE": round(res.metrics.mae, 3),
                "R2": round(res.metrics.r2, 3),
                "Samples": res.metrics.n_samples,
                "Mean Bias": round(res.metrics.bias, 3)
            })
        return pd.DataFrame(records)
