"""Produce Cooling Validation Runner.

Reproduces Section 3.3, Tables 4, 5, 6, and Figures 10, 11, 12 of Canatan et al. (2025):
- Table 4 & Figure 10: Apples cooling experiment (Triplicate)
- Table 5 & Figure 11: Bell peppers cooling experiment (Red, Yellow, Green)
- Table 6 & Figure 12: Individually wrapped broccoli cooling experiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.analytics.statistical_metrics import compute_metrics
from src.visualization.plotter import PaperVisualizer
from src.analytics.report_generator import ValidationReportGenerator


def validate_apples(trials_dir: Path, fig_dir: Path, reports_dir: Path):
    print("\n---------------------------------------------------------------------")
    print("Evaluating Apple Cooling Experiments (Table 4 & Figure 10)")
    print("---------------------------------------------------------------------")
    trial_dfs = []
    trials_metrics = {}

    for i in range(1, 4):
        fpath = trials_dir / f"apples_trial_{i}.csv"
        df = pd.read_csv(fpath)
        trial_dfs.append(df)
        m = compute_metrics(df["tc_apple"].values, df["ir_apple"].values)
        trials_metrics[f"Trial_{i}"] = m

    summary_df = ValidationReportGenerator.generate_produce_table(
        "Apple", {"Apple": trials_metrics}
    )
    print("\n--- REPRODUCTION OF TABLE 4: Apple Cooling Experiments ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(reports_dir / "table_4_apples.csv", index=False)

    fig_path = fig_dir / "figure_10_apples_cooling.png"
    PaperVisualizer.plot_figure_10_apples_cooling(trial_dfs, save_path=fig_path)
    print(f"Saved Figure 10 to: {fig_path}")


def validate_peppers(trials_dir: Path, fig_dir: Path, reports_dir: Path):
    print("\n---------------------------------------------------------------------")
    print("Evaluating Bell Pepper Cooling Experiments (Table 5 & Figure 11)")
    print("---------------------------------------------------------------------")
    trial_dfs = []
    peppers_dict = {
        "Pepper 1 (Red)": {},
        "Pepper 2 (Yellow)": {},
        "Pepper 3 (Green)": {},
    }
    key_map = {
        "Pepper 1 (Red)": "red_pepper",
        "Pepper 2 (Yellow)": "yellow_pepper",
        "Pepper 3 (Green)": "green_pepper",
    }

    for i in range(1, 4):
        fpath = trials_dir / f"bell_peppers_trial_{i}.csv"
        df = pd.read_csv(fpath)
        trial_dfs.append(df)

        for p_name, col_key in key_map.items():
            m = compute_metrics(df[f"tc_{col_key}"].values, df[f"ir_{col_key}"].values)
            peppers_dict[p_name][f"Trial_{i}"] = m

    summary_df = ValidationReportGenerator.generate_produce_table("Bell Pepper", peppers_dict)
    print("\n--- REPRODUCTION OF TABLE 5: Bell Pepper Cooling Experiments ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(reports_dir / "table_5_peppers.csv", index=False)

    fig_path = fig_dir / "figure_11_peppers_cooling.png"
    PaperVisualizer.plot_figure_11_peppers_cooling(trial_dfs, save_path=fig_path)
    print(f"Saved Figure 11 to: {fig_path}")


def validate_broccoli(trials_dir: Path, fig_dir: Path, reports_dir: Path):
    print("\n---------------------------------------------------------------------")
    print("Evaluating Individually Wrapped Broccoli Cooling (Table 6 & Figure 12)")
    print("---------------------------------------------------------------------")
    trial_dfs = []
    broc_dict = {
        "Broccoli 1": {},
        "Broccoli 2": {},
    }

    for i in range(1, 4):
        fpath = trials_dir / f"broccoli_trial_{i}.csv"
        df = pd.read_csv(fpath)
        trial_dfs.append(df)

        m1 = compute_metrics(df["tc_broccoli_1"].values, df["ir_broccoli_1"].values)
        broc_dict["Broccoli 1"][f"Trial_{i}"] = m1

        m2 = compute_metrics(df["tc_broccoli_2"].values, df["ir_broccoli_2"].values)
        broc_dict["Broccoli 2"][f"Trial_{i}"] = m2

    summary_df = ValidationReportGenerator.generate_produce_table("Broccoli", broc_dict)
    print("\n--- REPRODUCTION OF TABLE 6: Wrapped Broccoli Cooling Experiments ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(reports_dir / "table_6_broccoli.csv", index=False)

    fig_path = fig_dir / "figure_12_broccoli_cooling.png"
    PaperVisualizer.plot_figure_12_broccoli_cooling(trial_dfs, save_path=fig_path)
    print(f"Saved Figure 12 to: {fig_path}")


def main():
    root = Path(__file__).resolve().parent.parent
    trials_dir = root / "data" / "produce_trials"
    fig_dir = root / "reports" / "figures"
    reports_dir = root / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    validate_apples(trials_dir, fig_dir, reports_dir)
    validate_peppers(trials_dir, fig_dir, reports_dir)
    validate_broccoli(trials_dir, fig_dir, reports_dir)
    print("\nAll produce cooling validations completed successfully.")


if __name__ == "__main__":
    main()
