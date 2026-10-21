"""Visualization routines reproducing Figures 8, 9, 10, 11, and 12 of the research paper."""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd


class PaperVisualizer:
    """Generates publication-quality figures matching Canatan et al. (2025)."""

    @staticmethod
    def plot_figure_8_water_calibration(
        df_trials: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Reproduces Figure 8: Water heating experiment comparison plot.

        Plots TC Temp vs IR Temp over time (0 to 2400s) with triplicate standard deviation shaded bands.
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        time_s = df_trials["time_s"].values
        tc_mean = df_trials["tc_mean"].values
        tc_std = df_trials.get("tc_std", np.zeros_like(tc_mean)).values

        ir_mean = df_trials["ir_mean"].values
        ir_std = df_trials.get("ir_std", np.zeros_like(ir_mean)).values

        # Plot curves
        ax.plot(time_s, tc_mean, label="TC Temp", color="#1f77b4", linewidth=1.8)
        ax.fill_between(time_s, tc_mean - tc_std, tc_mean + tc_std, color="#1f77b4", alpha=0.25)

        ax.plot(time_s, ir_mean, label="IR Temp", color="#2ca02c", linestyle="--", linewidth=1.8)
        ax.fill_between(time_s, ir_mean - ir_std, ir_mean + ir_std, color="#2ca02c", alpha=0.25)

        ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        ax.set_title("Water Heating Calibration Experiment (Figure 8 Reproduction)", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 2400)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_figure_9_multimodal_quad(
        rgb_image: np.ndarray,
        thermal_raw: np.ndarray,
        colored_ir: np.ndarray,
        fused_annotated: np.ndarray,
        title: str = "Multimodal Sensor Fusion & Central 7x7 ROI Extraction",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Reproduces Figure 9: Quad view of RGB, Raw IR, Colored IR, and Fused Detections."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

        # 1. RGB Image
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title("RGB Image", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        # 2. Raw 16-bit IR (Grayscale Normalized)
        raw_norm = cv2.normalize(thermal_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        axes[0, 1].imshow(raw_norm, cmap="gray")
        axes[0, 1].set_title("Raw IR (Grayscale)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        # 3. Colored IR
        if len(colored_ir.shape) == 3 and colored_ir.shape[2] == 3:
            # OpenCV BGR to RGB
            colored_rgb = cv2.cvtColor(colored_ir, cv2.COLOR_BGR2RGB)
        else:
            colored_rgb = colored_ir
        axes[1, 0].imshow(colored_rgb)
        axes[1, 0].set_title("Colored IR (JET)", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        # 4. Fused Detection + 7x7 ROI
        if len(fused_annotated.shape) == 3 and fused_annotated.shape[2] == 3:
            fused_rgb = cv2.cvtColor(fused_annotated, cv2.COLOR_BGR2RGB)
        else:
            fused_rgb = fused_annotated
        axes[1, 1].imshow(fused_rgb)
        axes[1, 1].set_title("Fused RGB-Thermal + 7x7 Central ROI", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_figure_10_apples_cooling(
        trials: List[pd.DataFrame],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Reproduces Figure 10: Apple cooling and storage curves for 3 trials."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)

        for i, df in enumerate(trials):
            ax = axes[i]
            t = df["time_s"].values
            ax.plot(t, df["tc_refrigerator"].values, color="#cccccc", label="TC Refrigerator", linewidth=1.0)
            ax.plot(t, df["tc_apple"].values, color="#d62728", label="TC Apple", linewidth=1.5)
            ax.plot(t, df["ir_apple"].values, color="#ff7f0e", linestyle=":", label="IR Apple", alpha=0.7)
            if "ir_apple_mavg" in df.columns:
                ax.plot(t, df["ir_apple_mavg"].values, color="#1f77b4", linestyle="--", label="Moving Avg IR Apple", linewidth=1.5)

            ax.set_ylabel("Temperature (°C)", fontsize=10, fontweight="bold")
            ax.set_title(f"Apple Test {i + 1}", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", frameon=True, fontsize=8)

        axes[2].set_xlabel("Time (s)", fontsize=11, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_figure_11_peppers_cooling(
        trials: List[pd.DataFrame],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Reproduces Figure 11: Bell peppers cooling (Red, Yellow, Green peppers)."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), dpi=300, sharex=True, sharey=True)
        pepper_names = ["Red Bell Pepper", "Yellow Bell Pepper", "Green Bell Pepper"]
        col_keys = [("red_pepper", "#d62728"), ("yellow_pepper", "#bcbd22"), ("green_pepper", "#2ca02c")]

        for row_idx, df in enumerate(trials):
            t = df["time_s"].values
            for col_idx, (pkey, pcolor) in enumerate(col_keys):
                ax = axes[row_idx, col_idx]
                ax.plot(t, df["tc_refrigerator"].values, color="#cccccc", label="TC Fridge", linewidth=0.8)
                ax.plot(t, df[f"tc_{pkey}"].values, color=pcolor, label=f"TC {pepper_names[col_idx]}", linewidth=1.4)
                ax.plot(t, df[f"ir_{pkey}"].values, color="#ff7f0e", linestyle=":", alpha=0.6, label="IR")
                if f"ir_{pkey}_mavg" in df.columns:
                    ax.plot(t, df[f"ir_{pkey}_mavg"].values, color="#1f77b4", linestyle="--", label="Moving Avg IR")

                if row_idx == 0:
                    ax.set_title(f"{pepper_names[col_idx]} (Test 1)", fontsize=10, fontweight="bold")
                elif row_idx == 1:
                    ax.set_title(f"{pepper_names[col_idx]} (Test 2)", fontsize=10, fontweight="bold")
                else:
                    ax.set_title(f"{pepper_names[col_idx]} (Test 3)", fontsize=10, fontweight="bold")

                if col_idx == 0:
                    ax.set_ylabel("Temp (°C)", fontsize=10)
                if row_idx == 2:
                    ax.set_xlabel("Time (s)", fontsize=10)
                ax.grid(True, linestyle=":", alpha=0.5)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc="upper right", fontsize=7)

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_figure_12_broccoli_cooling(
        trials: List[pd.DataFrame],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Reproduces Figure 12: Broccoli cooling curves showing film wrapping effects."""
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=300, sharex=True, sharey=True)

        for row_idx, df in enumerate(trials):
            t = df["time_s"].values
            # Left Broccoli (Broccoli 1)
            ax_l = axes[row_idx, 0]
            ax_l.plot(t, df["tc_refrigerator"].values, color="#cccccc", label="TC Fridge", linewidth=0.8)
            ax_l.plot(t, df["tc_broccoli_1"].values, color="#2ca02c", label="TC Broccoli 1", linewidth=1.5)
            ax_l.plot(t, df["ir_broccoli_1"].values, color="#ff7f0e", linestyle=":", label="IR")
            if "ir_broccoli_1_mavg" in df.columns:
                ax_l.plot(t, df["ir_broccoli_1_mavg"].values, color="#1f77b4", linestyle="--", label="Moving Avg")
            ax_l.set_title(f"Broccoli 1 - Trial {row_idx + 1}", fontsize=10, fontweight="bold")
            ax_l.grid(True, linestyle=":", alpha=0.5)

            # Right Broccoli (Broccoli 2 - includes failure/shelf reflection in Trial 1)
            ax_r = axes[row_idx, 1]
            ax_r.plot(t, df["tc_refrigerator"].values, color="#cccccc", label="TC Fridge", linewidth=0.8)
            ax_r.plot(t, df["tc_broccoli_2"].values, color="#2ca02c", label="TC Broccoli 2", linewidth=1.5)
            ax_r.plot(t, df["ir_broccoli_2"].values, color="#ff7f0e", linestyle=":", label="IR")
            if "ir_broccoli_2_mavg" in df.columns:
                ax_r.plot(t, df["ir_broccoli_2_mavg"].values, color="#1f77b4", linestyle="--", label="Moving Avg")
            ax_r.set_title(f"Broccoli 2 - Trial {row_idx + 1}", fontsize=10, fontweight="bold")
            ax_r.grid(True, linestyle=":", alpha=0.5)

            if row_idx == 0:
                ax_l.legend(loc="upper right", fontsize=7)
            ax_l.set_ylabel("Temp (°C)", fontsize=10)

        axes[2, 0].set_xlabel("Time (s)", fontsize=10)
        axes[2, 1].set_xlabel("Time (s)", fontsize=10)

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        return fig
