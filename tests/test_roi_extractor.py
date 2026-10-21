"""Unit tests for Central 7x7 ROI extraction from thermal matrix."""

import numpy as np
import pytest
from src.detection.roi_extractor import CentralROIExtractor
from src.core.types import BoundingBox
from src.core.exceptions import OutOfBoundsROIError


def test_extract_from_thermal_matrix():
    extractor = CentralROIExtractor(block_size=(7, 7))

    # Create dummy 120x160 thermal matrix with background at 4.0 °C and center at 10.0 °C
    th_matrix = np.ones((120, 160), dtype=np.float32) * 4.0
    th_matrix[55:65, 75:85] = 10.0

    bbox = BoundingBox(xmin=60.0, ymin=40.0, xmax=100.0, ymax=80.0)
    # Center is (80.0, 60.0)
    roi = extractor.extract_from_thermal_matrix(th_matrix, bbox)

    assert roi.center_coord == (80, 60)
    assert roi.grid_size == (7, 7)
    assert roi.pixel_values_celsius.shape == (7, 7)
    assert roi.mean_temp_c == pytest.approx(10.0, abs=1e-3)
    assert roi.min_temp_c == pytest.approx(10.0, abs=1e-3)
    assert roi.max_temp_c == pytest.approx(10.0, abs=1e-3)
    assert roi.std_temp_c == pytest.approx(0.0, abs=1e-3)


def test_roi_boundary_clamping():
    extractor = CentralROIExtractor(block_size=(7, 7))
    th_matrix = np.ones((120, 160), dtype=np.float32) * 5.0

    # Bounding box near frame edge
    bbox = BoundingBox(xmin=0.0, ymin=0.0, xmax=4.0, ymax=4.0)
    roi = extractor.extract_from_thermal_matrix(th_matrix, bbox, clamp_to_bounds=True)

    assert roi.pixel_values_celsius.shape == (7, 7)
    assert roi.roi_bbox[0] >= 0
    assert roi.roi_bbox[1] >= 0


def test_divide_into_7x7_grid():
    extractor = CentralROIExtractor(block_size=(7, 7))
    th_matrix = np.ones((120, 160), dtype=np.float32) * 3.5
    bbox = BoundingBox(xmin=20.0, ymin=20.0, xmax=90.0, ymax=90.0)

    grid = extractor.divide_into_7x7_grid(th_matrix, bbox)
    assert grid.shape == (7, 7)
    assert np.allclose(grid, 3.5)
