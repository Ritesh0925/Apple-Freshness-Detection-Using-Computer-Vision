"""Unit tests for homography alignment and coordinate projection."""

import numpy as np
import pytest
from src.fusion.homography import HomographyAligner
from src.core.types import BoundingBox
from src.core.exceptions import HomographyComputationError


def test_homography_estimation():
    # Affine/perspective ground truth: scale 0.25 and translate (+10, +5)
    src_pts = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0],
    ], dtype=np.float32)

    dst_pts = np.array([
        [10.0, 5.0],
        [35.0, 5.0],
        [35.0, 30.0],
        [10.0, 30.0],
    ], dtype=np.float32)

    aligner = HomographyAligner.from_points(src_pts, dst_pts)
    assert aligner.H is not None
    assert aligner.H.shape == (3, 3)

    # Test point transformation
    tx, ty = aligner.transform_point((50.0, 50.0))
    assert pytest.approx(tx, abs=0.5) == 22.5
    assert pytest.approx(ty, abs=0.5) == 17.5

    # Test inverse transformation
    rx, ry = aligner.transform_point((tx, ty), inverse=True)
    assert pytest.approx(rx, abs=0.5) == 50.0
    assert pytest.approx(ry, abs=0.5) == 50.0


def test_bbox_transformation():
    src_pts = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

    aligner = HomographyAligner.from_points(src_pts, dst_pts)
    bbox_src = BoundingBox(xmin=40.0, ymin=80.0, xmax=120.0, ymax=160.0)
    bbox_dst = aligner.transform_bbox(bbox_src)

    assert pytest.approx(bbox_dst.xmin, abs=0.5) == 10.0
    assert pytest.approx(bbox_dst.ymin, abs=0.5) == 20.0
    assert pytest.approx(bbox_dst.xmax, abs=0.5) == 30.0
    assert pytest.approx(bbox_dst.ymax, abs=0.5) == 40.0


def test_insufficient_points():
    with pytest.raises(HomographyComputationError):
        HomographyAligner.from_points([[0, 0], [1, 1]], [[0, 0], [1, 1]])
