"""Central 7x7 ROI extraction and temperature aggregation.

Implements Sections 2.2 and 3.2 of Canatan et al. (2025):
- Bounding box geometric center localization.
- 7x7 grid partitioning and extraction of central 7x7 pixel block (49 pixels).
- Radiometric temperature statistics calculation while eliminating edge/background shelf leakage.
"""

from typing import Tuple, Optional
import numpy as np

from src.core.types import BoundingBox, CentralROIExtraction
from src.core.exceptions import OutOfBoundsROIError


class CentralROIExtractor:
    """Extracts a localized 7x7 pixel region at the geometric center of a detected object."""

    def __init__(self, block_size: Tuple[int, int] = (7, 7)):
        """
        Args:
            block_size: (width, height) of the central pixel block. Default (7, 7) = 49 pixels.
        """
        self.block_width = block_size[0]
        self.block_height = block_size[1]
        self.half_w = self.block_width // 2
        self.half_h = self.block_height // 2

    def extract_from_thermal_matrix(
        self,
        thermal_celsius: np.ndarray,
        bbox: BoundingBox,
        clamp_to_bounds: bool = True
    ) -> CentralROIExtraction:
        """Extracts the 7x7 central window from a calibrated thermal matrix (H, W in °C).

        Args:
            thermal_celsius: 2D numpy array (e.g. 120x160) containing temperature in °C.
            bbox: Bounding box in the thermal frame's coordinate space.
            clamp_to_bounds: Whether to shift ROI if center falls near image boundary.

        Returns:
            CentralROIExtraction containing pixel values, mean, min, max, std.
        """
        img_h, img_w = thermal_celsius.shape[:2]
        cx, cy = bbox.center
        center_x = int(round(cx))
        center_y = int(round(cy))

        x1 = center_x - self.half_w
        x2 = x1 + self.block_width
        y1 = center_y - self.half_h
        y2 = y1 + self.block_height

        if clamp_to_bounds:
            # Shift window if exceeding frame boundaries
            if x1 < 0:
                x2 += -x1
                x1 = 0
            if y1 < 0:
                y2 += -y1
                y1 = 0
            if x2 > img_w:
                x1 -= (x2 - img_w)
                x2 = img_w
            if y2 > img_h:
                y1 -= (y2 - img_h)
                y2 = img_h

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
        else:
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                raise OutOfBoundsROIError(
                    f"ROI window [{x1}:{x2}, {y1}:{y2}] out of image bounds ({img_w}x{img_h})"
                )

        roi_patch = thermal_celsius[y1:y2, x1:x2]

        if roi_patch.size == 0:
            raise OutOfBoundsROIError("Extracted empty ROI patch.")

        mean_val = float(np.mean(roi_patch))
        min_val = float(np.min(roi_patch))
        max_val = float(np.max(roi_patch))
        std_val = float(np.std(roi_patch))

        return CentralROIExtraction(
            center_coord=(center_x, center_y),
            roi_bbox=(x1, y1, x2, y2),
            grid_size=(roi_patch.shape[1], roi_patch.shape[0]),
            pixel_values_celsius=roi_patch,
            mean_temp_c=mean_val,
            min_temp_c=min_val,
            max_temp_c=max_val,
            std_temp_c=std_val
        )

    def divide_into_7x7_grid(
        self,
        thermal_celsius: np.ndarray,
        bbox: BoundingBox
    ) -> np.ndarray:
        """Subdivides the bounding box into a 7x7 grid of sections and calculates mean temperature

        Paper Section 2.2: 'bounding box dimensions were modified to 7x7 sections'

        Returns:
            7x7 array containing mean temperatures of each cell.
        """
        x1, y1, x2, y2 = bbox.to_int_tuple()
        img_h, img_w = thermal_celsius.shape[:2]

        x1 = max(0, min(img_w - 1, x1))
        x2 = max(x1 + 1, min(img_w, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(y1 + 1, min(img_h, y2))

        crop = thermal_celsius[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape

        grid_temps = np.zeros((7, 7), dtype=np.float32)
        step_y = crop_h / 7.0
        step_x = crop_w / 7.0

        for r in range(7):
            for c in range(7):
                sy1 = int(round(r * step_y))
                sy2 = int(round((r + 1) * step_y))
                sx1 = int(round(c * step_x))
                sx2 = int(round((c + 1) * step_x))

                cell = crop[sy1:max(sy1 + 1, sy2), sx1:max(sx1 + 1, sx2)]
                if cell.size > 0:
                    grid_temps[r, c] = float(np.mean(cell))
                else:
                    grid_temps[r, c] = float(np.mean(crop))

        return grid_temps
