"""Integration tests for Continuous Temperature Monitoring Pipeline."""

import numpy as np
import pytest
from src.sensors.rgb_camera import RGBCamera
from src.sensors.thermal_camera import ThermalCamera
from src.fusion.homography import HomographyAligner
from src.detection.detector import ProduceDetector
from src.detection.roi_extractor import CentralROIExtractor
from src.pipeline.continuous_monitor import ContinuousTemperatureMonitor
from src.pipeline.alerting import ColdChainAlertManager


def test_continuous_monitor_integration():
    rgb_cam = RGBCamera(is_simulated=True)
    th_cam = ThermalCamera(is_simulated=True)

    H_ident = np.eye(3, dtype=np.float32)
    aligner = HomographyAligner(homography_matrix=H_ident)
    detector = ProduceDetector(confidence_threshold=0.3)
    roi_extractor = CentralROIExtractor(block_size=(7, 7))
    alert_mgr = ColdChainAlertManager(produce_profiles={
        "apple": {"storage_temp_recommended_c": [0.0, 4.0]}
    })

    monitor = ContinuousTemperatureMonitor(
        rgb_camera=rgb_cam,
        thermal_camera=th_cam,
        homography_aligner=aligner,
        detector=detector,
        roi_extractor=roi_extractor,
        alert_manager=alert_mgr
    )

    ret_rgb, frame_rgb, t_rgb = rgb_cam.capture_frame()
    ret_th, frame_th, t_th = th_cam.capture_frame()

    assert ret_rgb and frame_rgb is not None
    assert ret_th and frame_th is not None

    vis, states = monitor.process_single_frame(frame_rgb, frame_th, timestamp=100.0)

    assert vis is not None
    assert len(vis.shape) == 3
    assert len(states) >= 1

    s = states[0]
    assert s.roi.grid_size == (7, 7)
    assert s.surface_temp_c > -10.0 and s.surface_temp_c < 50.0
