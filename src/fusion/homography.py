"""Homography-based multimodal sensor registration between RGB and thermal infrared sensors.

Implements Section 2.1.2 ('RGB and thermal camera fusion via homography') of Canatan et al. (2025).
Supports point correspondence loading from MATLAB .mat or JSON/dict, RANSAC homography estimation,
spatial coordinate mapping, image perspective warping, and multimodal alpha blending.
"""

from pathlib import Path
from typing import Tuple, List, Union, Optional, Dict, Any
import numpy as np
import cv2
from scipy.io import loadmat

from src.core.exceptions import HomographyComputationError
from src.core.types import BoundingBox


class HomographyAligner:
    """Manages homography estimation and spatial warping between RGB and Thermal camera frames."""

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        ransac_reproj_threshold: float = 3.0
    ):
        """
        Args:
            homography_matrix: Optional precomputed 3x3 homography matrix H.
            ransac_reproj_threshold: Maximum allowed reprojection error in RANSAC.
        """
        self.H = homography_matrix
        self.H_inv: Optional[np.ndarray] = None
        self.ransac_threshold = ransac_reproj_threshold

        if self.H is not None:
            self._update_inverse()

    def _update_inverse(self) -> None:
        """Computes and caches inverse homography matrix."""
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def from_points(
        cls,
        src_points: Union[np.ndarray, List[List[float]]],
        dst_points: Union[np.ndarray, List[List[float]]],
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 3.0
    ) -> "HomographyAligner":
        """Calculates homography H from corresponding point pairs (at least 4 points).

        Args:
            src_points: Coordinates in source (RGB) image, shape (N, 2).
            dst_points: Coordinates in destination (IR) image, shape (N, 2).
            method: OpenCV method (cv2.RANSAC or 0 for standard least squares).
            ransac_reproj_threshold: Maximum allowable reprojection error.
        """
        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)

        if len(src) < 4 or len(dst) < 4:
            raise HomographyComputationError(
                f"At least 4 corresponding point pairs required, got {len(src)} and {len(dst)}"
            )

        H, mask = cv2.findHomography(src, dst, method, ransac_reproj_threshold)
        if H is None:
            raise HomographyComputationError("cv2.findHomography failed to calculate a valid matrix.")

        aligner = cls(homography_matrix=H, ransac_reproj_threshold=ransac_reproj_threshold)
        return aligner

    @classmethod
    def from_mat_file(
        cls,
        mat_path: Union[str, Path],
        rgb_key: str = "rgb_points",
        ir_key: str = "ir_points"
    ) -> "HomographyAligner":
        """Loads point coordinates from MATLAB .mat file exported via 'cpselect' (Section 2.1.2).

        Args:
            mat_path: Path to .mat file.
            rgb_key: Variable name for RGB points in the MAT file.
            ir_key: Variable name for IR points in the MAT file.
        """
        mat_file = Path(mat_path)
        if not mat_file.exists():
            raise FileNotFoundError(f"MAT file not found at {mat_file}")

        data = loadmat(str(mat_file))
        if rgb_key not in data or ir_key not in data:
            # Fallback check for alternate keys produced by cpselect
            keys = [k for k in data.keys() if not k.startswith("__")]
            if len(keys) >= 2:
                src_pts = data[keys[0]]
                dst_pts = data[keys[1]]
            else:
                raise HomographyComputationError(f"Keys {rgb_key}, {ir_key} not found in {keys}")
        else:
            src_pts = data[rgb_key]
            dst_pts = data[ir_key]

        return cls.from_points(src_pts, dst_pts)

    def warp_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        inverse: bool = False
    ) -> np.ndarray:
        """Warps image using the estimated homography matrix.

        Args:
            image: Source image array (H, W) or (H, W, C).
            target_size: (width, height) of destination canvas.
            inverse: If True, uses H_inv to map destination back to source.
        """
        if self.H is None:
            raise HomographyComputationError("Homography matrix has not been computed or loaded.")

        matrix = self.H_inv if inverse else self.H
        warped = cv2.warpPerspective(image, matrix, target_size, flags=cv2.INTER_LINEAR)
        return warped

    def transform_point(self, point: Tuple[float, float], inverse: bool = False) -> Tuple[float, float]:
        """Projects a single 2D point (x, y) using homography."""
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        matrix = self.H_inv if inverse else self.H
        transformed = cv2.perspectiveTransform(pts, matrix)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])

    def transform_bbox(self, bbox: BoundingBox, inverse: bool = False) -> BoundingBox:
        """Projects a 2D bounding box from RGB to Thermal space (or inverse)."""
        corners = np.array([
            [[bbox.xmin, bbox.ymin]],
            [[bbox.xmax, bbox.ymin]],
            [[bbox.xmax, bbox.ymax]],
            [[bbox.xmin, bbox.ymax]],
        ], dtype=np.float32)

        matrix = self.H_inv if inverse else self.H
        proj = cv2.perspectiveTransform(corners, matrix)[:, 0, :]

        xmin, ymin = np.min(proj, axis=0)
        xmax, ymax = np.max(proj, axis=0)
        return BoundingBox(xmin=float(xmin), ymin=float(ymin), xmax=float(xmax), ymax=float(ymax))

    def create_multimodal_overlay(
        self,
        rgb_frame: np.ndarray,
        thermal_display: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Creates an aligned alpha-blended fusion overlay of RGB and Thermal images.

        Args:
            rgb_frame: Visible RGB frame.
            thermal_display: Colormapped 8-bit thermal image (e.g. cv2.COLORMAP_JET).
            alpha: Weight of the warped RGB layer (1 - alpha for thermal).
        """
        th_h, th_w = thermal_display.shape[:2]
        warped_rgb = self.warp_image(rgb_frame, target_size=(th_w, th_h))

        if len(thermal_display.shape) == 2:
            thermal_display = cv2.applyColorMap(thermal_display, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(warped_rgb, alpha, thermal_display, 1.0 - alpha, 0.0)
        return overlay
