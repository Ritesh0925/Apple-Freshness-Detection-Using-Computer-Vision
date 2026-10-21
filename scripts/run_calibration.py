"""Water heating calibration runner.

Reproduces Section 3.1, Figure 8, and Table 3 of Canatan et al. (2025).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.calibration.water_bath_calibrator import WaterBathCalibrator
from src.visualization.plotter import PaperVisualizer
from src.analytics.report_generator import ValidationReportGenerator


def main():
    root = Path(__file__).resolve().parent.parent
    calib_csv = root / "data" / "calibration" / "water_heating_calibration.csv"
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not calib_csv.exists():
        print(f"Calibration data not found at {calib_csv}. Please run generate_benchmark_data.py first.")
        sys.exit(1)

    print("=====================================================================")
    print("Executing Water Heating Calibration Experiment (Section 3.1)")
    print("=====================================================================")

    df = pd.read_csv(calib_csv)
    calibrator = WaterBathCalibrator(split_temp_c=53.0)
    results = calibrator.fit(df["tc_mean"].values, df["ir_mean"].values)

    # Format Table 3
    table_3_df = ValidationReportGenerator.generate_water_calibration_table(results)
    print("\n--- REPRODUCTION OF TABLE 3: Water Heating Calibration Metrics ---")
    print(table_3_df.to_string(index=False))

    # Plot Figure 8
    fig_path = fig_dir / "figure_8_water_calibration.png"
    PaperVisualizer.plot_figure_8_water_calibration(df, save_path=fig_path)
    print(f"\nFigure 8 successfully saved to: {fig_path}")

    # Export report summary
    report_path = root / "reports" / "table_3_calibration_summary.csv"
    table_3_df.to_csv(report_path, index=False)
    print(f"Saved calibration metrics table to: {report_path}")


if __name__ == "__main__":
    main()
