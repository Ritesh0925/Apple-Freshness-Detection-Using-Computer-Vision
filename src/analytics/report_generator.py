"""Report and validation table generator matching the research paper format.

Formats results into structured Markdown and Pandas DataFrames matching:
- Table 3: Water heating calibration experiment (0-100 °C, 0-53 °C, 53-100 °C)
- Table 4: Apple cooling experiment (Triplicate trials 1, 2, 3)
- Table 5: Bell pepper cooling experiment (Pepper 1 Red, Pepper 2 Yellow, Pepper 3 Green)
- Table 6: Broccoli cooling experiment (Broccoli 1, Broccoli 2)
"""

from typing import Dict, List, Any
import pandas as pd
from src.core.types import StatisticalMetrics


class ValidationReportGenerator:
    """Generates benchmark tables and statistical comparisons."""

    @staticmethod
    def generate_water_calibration_table(results: Dict[str, Any]) -> pd.DataFrame:
        """Reproduces Table 3 format."""
        rows = []
        for key in ["0-100 C", "0-53 C", "53-100 C"]:
            if key in results:
                r = results[key]
                rows.append({
                    "Temperature Range": r.temperature_range,
                    "RMSE": f"{r.metrics.rmse:.3f}",
                    "MAE": f"{r.metrics.mae:.3f}",
                    "R2": f"{r.metrics.r2:.3f}",
                })
        return pd.DataFrame(rows)

    @staticmethod
    def generate_produce_table(
        produce_name: str,
        trials_data: Dict[str, Dict[str, StatisticalMetrics]]
    ) -> pd.DataFrame:
        """Produces multi-index summary table matching Tables 4, 5, and 6.

        Args:
            produce_name: "Apple", "Bell Pepper", or "Broccoli"
            trials_data: Nested dict: item_name -> { "Trial 1": metrics, "Trial 2": metrics, "Trial 3": metrics }
        """
        records = []
        for item_name, trials in trials_data.items():
            row = {"Sample": item_name}
            for trial_name, m in trials.items():
                row[f"{trial_name}_RMSE"] = round(m.rmse, 3)
                row[f"{trial_name}_MAE"] = round(m.mae, 3)
                row[f"{trial_name}_R2"] = round(m.r2, 3)
            records.append(row)
        return pd.DataFrame(records)

    @staticmethod
    def print_markdown_summary(title: str, df: pd.DataFrame) -> str:
        """Returns pretty formatted markdown table."""
        md = f"### {title}\n\n"
        md += df.to_markdown(index=False)
        md += "\n\n"
        return md
