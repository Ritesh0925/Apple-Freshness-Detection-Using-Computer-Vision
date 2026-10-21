"""Temporal synchronization, 7 FPS to 1 Hz downsampling, and moving average smoothing.

Implements the temporal processing pipeline described in Sections 2.3.1, 3.1, and 3.3.1:
- Aggregation of 7 frames per second (49 pixels each) to 1 Hz to match Graphtec GL240 thermocouple logs.
- Timestamp alignment, deduplication, and removal of non-corresponding entries.
- Moving average filtering over configurable temporal windows.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class TemporalSynchronizer:
    """Synchronizes high-rate thermal readings (~7 FPS) with 1 Hz thermocouple readings."""

    def __init__(self, moving_avg_window: int = 5):
        """
        Args:
            moving_avg_window: Window size in seconds for smoothing.
        """
        self.moving_avg_window = moving_avg_window

    def aggregate_7fps_to_1hz(self, subsecond_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Averages high-rate thermal frames into 1-second bins.

        Args:
            subsecond_records: List of dicts with 'timestamp_s' (float) and 'temp_c' (float).

        Returns:
            pd.DataFrame with 1-second integer intervals and mean temperatures.
        """
        if not subsecond_records:
            return pd.DataFrame(columns=["second", "temp_c"])

        df = pd.DataFrame(subsecond_records)
        df["second"] = df["timestamp_s"].astype(int)
        agg_df = df.groupby("second")["temp_c"].mean().reset_index()
        return agg_df

    def synchronize_and_clean(
        self,
        tc_series: pd.DataFrame,
        ir_series: pd.DataFrame,
        time_col: str = "time_s",
        tc_col: str = "tc_temp_c",
        ir_col: str = "ir_temp_c"
    ) -> pd.DataFrame:
        """Synchronizes timestamps, drops duplicates, removes unpaired seconds, and calculates moving average.

        Paper Section 3.3.1: 'Post-processing involved the removal of duplicates and
        non-corresponding entries, with an average data loss of one second per 1000.'
        """
        tc_clean = tc_series.drop_duplicates(subset=[time_col]).sort_values(by=time_col)
        ir_clean = ir_series.drop_duplicates(subset=[time_col]).sort_values(by=time_col)

        merged = pd.merge(tc_clean, ir_clean, on=time_col, how="inner")

        # Compute moving average for IR signal
        merged["ir_moving_avg_c"] = merged[ir_col].rolling(
            window=self.moving_avg_window, min_periods=1, center=False
        ).mean()

        return merged


def apply_moving_average(series: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Applies 1D moving average filter."""
    if len(series) == 0:
        return series
    if window_size <= 1:
        return series.copy()

    kernel = np.ones(window_size) / float(window_size)
    # Use mode='same' to preserve series length
    smoothed = np.convolve(series, kernel, mode="same")
    # Fix boundary effects
    half = window_size // 2
    for i in range(half):
        smoothed[i] = np.mean(series[: i + half + 1])
        smoothed[-(i + 1)] = np.mean(series[-(i + half + 1):])
    return smoothed
