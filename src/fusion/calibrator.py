"""Interactive and programmatic calibration tools for homography correspondence selection.

Equivalent to MATLAB 'cpselect' workflow cited in Section 2.1.2 and Fig. 4.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import json
import numpy as np
from scipy.io import savemat

from src.fusion.homography import HomographyAligner


class CameraCalibrator:
    """Manages control points between RGB and Thermal views and produces homography matrices."""

    def __init__(self):
        self.rgb_points: List[Tuple[float, float]] = []
        self.ir_points: List[Tuple[float, float]] = []

    def add_point_pair(self, rgb_pt: Tuple[float, float], ir_pt: Tuple[float, float]) -> None:
        """Adds a corresponding coordinate pair."""
        self.rgb_points.append(rgb_pt)
        self.ir_points.append(ir_pt)

    def set_points(
        self,
        rgb_pts: List[Tuple[float, float]],
        ir_pts: List[Tuple[float, float]]
    ) -> None:
        """Sets all control point pairs."""
        if len(rgb_pts) != len(ir_pts):
            raise ValueError(f"Mismatched points length: {len(rgb_pts)} vs {len(ir_pts)}")
        self.rgb_points = list(rgb_pts)
        self.ir_points = list(ir_pts)

    def compute_aligner(self, ransac_threshold: float = 3.0) -> HomographyAligner:
        """Computes HomographyAligner from current point pairs."""
        return HomographyAligner.from_points(
            self.rgb_points,
            self.ir_points,
            ransac_reproj_threshold=ransac_threshold
        )

    def evaluate_reprojection_error(self, aligner: HomographyAligner) -> Dict[str, float]:
        """Calculates mean, max, and std reprojection error across all control points."""
        errors = []
        for rgb_pt, ir_pt in zip(self.rgb_points, self.ir_points):
            proj_x, proj_y = aligner.transform_point(rgb_pt)
            err = np.sqrt((proj_x - ir_pt[0]) ** 2 + (proj_y - ir_pt[1]) ** 2)
            errors.append(err)

        err_arr = np.array(errors)
        return {
            "mean_reprojection_error_px": float(np.mean(err_arr)),
            "max_reprojection_error_px": float(np.max(err_arr)),
            "std_reprojection_error_px": float(np.std(err_arr)),
            "n_points": len(errors)
        }

    def export_to_json(self, path: Union[str, Path]) -> None:
        """Saves points to JSON file."""
        data = {
            "rgb_points": self.rgb_points,
            "ir_points": self.ir_points,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_to_mat(self, path: Union[str, Path]) -> None:
        """Saves points to MATLAB .mat file for interoperability with MATLAB cpselect."""
        savemat(str(path), {
            "rgb_points": np.array(self.rgb_points, dtype=np.float64),
            "ir_points": np.array(self.ir_points, dtype=np.float64)
        })

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "CameraCalibrator":
        """Loads points from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calib = cls()
        calib.set_points(data["rgb_points"], data["ir_points"])
        return calib
