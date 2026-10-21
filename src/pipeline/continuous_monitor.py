"""Continuous Multimodal Temperature Monitoring Pipeline.

Implements the end-to-end 4-step sequence from Figure 5 of Canatan et al. (2025):
1. Initialize cameras, homography aligner, and object detector.
2. Stream synchronized RGB and 16-bit radiometric thermal frames.
3. Perform homography warping, produce detection, and central 7x7 ROI temperature extraction.
4. Temporal logging, moving average calculation, cold chain alert triggering, and visualization.
"""

from typing import List, Tuple, Optional, Dict, Any, Callable
from pathlib import Path
import time
import numpy as np
import cv2
import pandas as pd

from src.core.types import (
    BoundingBox, ProduceDetection, CentralROIExtraction, MonitoredObjectState
)
from src.core.exceptions import CameraAcquisitionError
from src.sensors.rgb_camera import RGBCamera
from src.sensors.thermal_camera import ThermalCamera
from src.sensors.thermocouple import ThermocoupleLogger
from src.fusion.homography import HomographyAligner
from src.detection.detector import BaseProduceDetector, ProduceDetector
from src.detection.roi_extractor import CentralROIExtractor
from src.calibration.water_bath_calibrator import WaterBathCalibrator
from src.calibration.emissivity import EmissivityCorrector
from src.analytics.temporal_filter import apply_moving_average
from src.pipeline.alerting import ColdChainAlertManager


class ContinuousTemperatureMonitor:
    """Orchestrates continuous multimodal temperature monitoring for refrigerated produce."""

    def __init__(
        self,
        rgb_camera: RGBCamera,
        thermal_camera: ThermalCamera,
        homography_aligner: HomographyAligner,
        detector: BaseProduceDetector,
        roi_extractor: Optional[CentralROIExtractor] = None,
        calibrator: Optional[WaterBathCalibrator] = None,
        emissivity_corrector: Optional[EmissivityCorrector] = None,
        alert_manager: Optional[ColdChainAlertManager] = None,
        tc_logger: Optional[ThermocoupleLogger] = None,
    ):
        self.rgb_cam = rgb_camera
        self.thermal_cam = thermal_camera
        self.aligner = homography_aligner
        self.detector = detector
        self.roi_extractor = roi_extractor or CentralROIExtractor(block_size=(7, 7))
        self.calibrator = calibrator
        self.emissivity_corrector = emissivity_corrector
        self.alert_manager = alert_manager
        self.tc_logger = tc_logger

        self.logs: List[Dict[str, Any]] = []
        self._history_by_track: Dict[str, List[float]] = {}
        self.is_running = False

    def process_single_frame(
        self,
        rgb_frame: np.ndarray,
        thermal_raw: np.ndarray,
        timestamp: float,
        manual_rois: Optional[List[BoundingBox]] = None
    ) -> Tuple[np.ndarray, List[MonitoredObjectState]]:
        """Processes a single paired time-slice of RGB and Thermal data.

        Args:
            rgb_frame: RGB image (H_rgb, W_rgb, 3).
            thermal_raw: 16-bit raw centiKelvin array (H_ir, W_ir).
            timestamp: Epoch timestamp in seconds.
            manual_rois: Optional manual bounding boxes (e.g. for apple baseline comparison).

        Returns:
            Tuple of (annotated_visualization_bgr, monitored_states).
        """
        # Step 02: Convert radiometric 16-bit raw to Celsius and false-color
        celsius_matrix = ThermalCamera.raw_to_celsius(thermal_raw, scale=self.thermal_cam.scale)
        th_h, th_w = celsius_matrix.shape

        # Create false color thermal map for visualization
        colored_ir = ThermalCamera.to_display_image(celsius_matrix, vmin=0.0, vmax=25.0)

        # Step 03: Register RGB onto IR space
        warped_rgb = self.aligner.warp_image(rgb_frame, target_size=(th_w, th_h))
        overlay = self.aligner.create_multimodal_overlay(rgb_frame, colored_ir, alpha=0.45)

        # Object Detection: If manual ROIs provided (e.g. apples test), use them; otherwise run detector
        detections: List[ProduceDetection] = []
        if manual_rois is not None:
            for idx, m_box in enumerate(manual_rois):
                detections.append(ProduceDetection(
                    class_id=idx + 1,
                    label="apple",
                    confidence=1.0,
                    bbox=m_box
                ))
        else:
            # Detect in RGB frame, then project bounding boxes to Thermal coordinates
            raw_rgb_detections = self.detector.detect(rgb_frame)
            for det in raw_rgb_detections:
                # Transform bbox from RGB space to Thermal space
                ir_bbox = self.aligner.transform_bbox(det.bbox)
                detections.append(ProduceDetection(
                    class_id=det.class_id,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=ir_bbox,
                    metadata={"orig_rgb_bbox": det.bbox}
                ))

        monitored_states: List[MonitoredObjectState] = []
        annotated_vis = overlay.copy()

        # Step 03 & 04: ROI Extraction & Temperature computation
        for idx, det in enumerate(detections):
            track_id = f"item_{det.label}_{idx + 1}"

            # Extract 7x7 central window
            try:
                roi_data = self.roi_extractor.extract_from_thermal_matrix(
                    celsius_matrix, det.bbox, clamp_to_bounds=True
                )
            except Exception:
                continue

            raw_temp = roi_data.mean_temp_c

            # Optional calibration correction (Water-bath model)
            if self.calibrator:
                calib_temp = self.calibrator.calibrate(raw_temp, piecewise=True)
            else:
                calib_temp = raw_temp

            # Optional emissivity correction
            if self.emissivity_corrector:
                final_temp = self.emissivity_corrector.correct_temperature(calib_temp)
            else:
                final_temp = calib_temp

            # Temporal history and moving average
            if track_id not in self._history_by_track:
                self._history_by_track[track_id] = []
            self._history_by_track[track_id].append(final_temp)

            window_vals = self._history_by_track[track_id][-5:]
            m_avg = float(np.mean(window_vals))

            # Thermocouple ground truth if logger available
            tc_val = None
            if self.tc_logger:
                tc_val = self.tc_logger.get_temperature_at_time(timestamp, channel=f"tc_{det.label}")

            # Check temperature abuse alert
            abuse_alert = False
            if self.alert_manager:
                alert = self.alert_manager.check_temperature(det.label, final_temp, timestamp)
                if alert:
                    abuse_alert = True

            state = MonitoredObjectState(
                track_id=track_id,
                label=det.label,
                timestamp=timestamp,
                bbox_rgb=det.metadata.get("orig_rgb_bbox", det.bbox),
                bbox_thermal=det.bbox,
                roi=roi_data,
                surface_temp_c=final_temp,
                moving_avg_temp_c=m_avg,
                thermocouple_temp_c=tc_val,
                abuse_alert=abuse_alert
            )
            monitored_states.append(state)

            # Record log entry
            self.logs.append({
                "timestamp": timestamp,
                "track_id": track_id,
                "label": det.label,
                "raw_temp_c": raw_temp,
                "surface_temp_c": final_temp,
                "moving_avg_c": m_avg,
                "tc_temp_c": tc_val,
                "roi_min_c": roi_data.min_temp_c,
                "roi_max_c": roi_data.max_temp_c,
                "roi_std_c": roi_data.std_temp_c,
                "abuse_alert": abuse_alert
            })

            # Draw visualization overlays (Matching Fig. 9)
            self._draw_overlay_graphics(annotated_vis, state)

        return annotated_vis, monitored_states

    def _draw_overlay_graphics(self, canvas: np.ndarray, state: MonitoredObjectState) -> None:
        """Renders bounding box, central 7x7 sample window, and temperature readout."""
        x1, y1, x2, y2 = state.bbox_thermal.to_int_tuple()
        # Draw produce outer bounding box in cyan or red if alert
        box_color = (0, 0, 255) if state.abuse_alert else (255, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 1)

        # Draw central 7x7 pixel ROI in bright green / yellow (Paper Fig. 9)
        rx1, ry1, rx2, ry2 = state.roi.roi_bbox
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)

        # Display temperature text
        label_text = f"{state.label}: {state.surface_temp_c:.1f}C"
        text_y = max(12, y1 - 4)
        cv2.putText(canvas, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    def run_stream(
        self,
        duration_seconds: float = 10.0,
        frame_callback: Optional[Callable[[np.ndarray, List[MonitoredObjectState]], None]] = None
    ) -> pd.DataFrame:
        """Executes the continuous monitoring loop for a specified duration."""
        self.rgb_cam.initialize()
        self.thermal_cam.initialize()
        self.is_running = True

        start_time = time.time()
        try:
            while self.is_running and (time.time() - start_time < duration_seconds):
                ret_rgb, frame_rgb, t_rgb = self.rgb_cam.capture_frame()
                ret_th, frame_th, t_th = self.thermal_cam.capture_frame()

                if not ret_rgb or not ret_th:
                    time.sleep(0.05)
                    continue

                vis, states = self.process_single_frame(frame_rgb, frame_th, t_th)

                if frame_callback:
                    frame_callback(vis, states)

                # Maintain ~7 FPS rate
                time.sleep(1.0 / 7.0)
        finally:
            self.stop()

        return self.get_logged_dataframe()

    def stop(self) -> None:
        """Stops streaming and releases camera resources."""
        self.is_running = False
        self.rgb_cam.release()
        self.thermal_cam.release()

    def get_logged_dataframe(self) -> pd.DataFrame:
        """Returns collected temperature logs as pandas DataFrame."""
        return pd.DataFrame(self.logs)
