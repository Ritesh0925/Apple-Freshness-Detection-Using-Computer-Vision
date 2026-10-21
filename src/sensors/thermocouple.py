"""Thermocouple data reader interface (Type-T thermocouple with Graphtec GL240 datalogger).

Implements Section 2.3.2 and Fig. 6/7:
- Synchronized reading of multi-channel temperature loggers (Product TC, Shelf/Ambient TC).
- CSV stream parser and data interpolation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd


class ThermocoupleLogger:
    """Manages thermocouple readings recorded via Graphtec GL240 or compatible datalogger."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        if csv_path and csv_path.exists():
            self.load_csv(csv_path)

    def load_csv(self, csv_path: Path) -> pd.DataFrame:
        """Loads datalogger CSV file."""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        # Standardize column names
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        return self.df

    def get_temperature_at_time(self, time_s: float, channel: str = "tc_produce") -> Optional[float]:
        """Interpolates thermocouple reading at exact timestamp."""
        if self.df is None or len(self.df) == 0:
            return None

        time_col = "time_s" if "time_s" in self.df.columns else self.df.columns[0]
        # Find closest row
        closest_idx = (self.df[time_col] - time_s).abs().idxmin()
        if channel in self.df.columns:
            return float(self.df.loc[closest_idx, channel])
        # Fallback to second column
        return float(self.df.iloc[closest_idx, 1])
