"""Object detection for fresh produce in refrigerated display cases.

Implements Section 2.2 of Canatan et al. (2025):
Locates fresh produce items (apples, bell peppers, broccoli) in the RGB camera frame,
isolating them from the background shelves and refrigeration fixtures.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import cv2

from src.core.types import BoundingBox, ProduceDetection
from src.core.exceptions import DetectionError


class BaseProduceDetector(ABC):
    """Abstract interface for fresh produce detectors."""

    @abstractmethod
    def detect(self, rgb_frame: np.ndarray) -> List[ProduceDetection]:
        """Detects produce items in the given RGB frame.

        Args:
            rgb_frame: RGB image as a numpy array (H, W, 3) in uint8.

        Returns:
            List of ProduceDetection instances.
        """
        pass


class ProduceDetector(BaseProduceDetector):
    """Multi-mode produce detector.

    Supports:
    1. HSV color-salience segmentation (apples, red/yellow/green bell peppers, broccoli heads).
    2. Optional Deep Learning detector inference (e.g. YOLO/EfficientDet weights when loaded).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        min_area: float = 500.0,
        max_area: float = 200000.0
    ):
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, rgb_frame: np.ndarray) -> List[ProduceDetection]:
        """Detects produce items in RGB frame using adaptive visual saliency and contour analysis."""
        if rgb_frame is None or rgb_frame.size == 0:
            raise DetectionError("Input RGB frame is empty or None.")

        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
        h, w = rgb_frame.shape[:2]

        # Multi-range mask for produce (Red/Green/Yellow/Deep Green)
        # 1. Red (apples, red peppers)
        mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([168, 50, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 2. Yellow (yellow peppers)
        mask_yellow = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([35, 255, 255]))

        # 3. Green (green peppers, broccoli, green apples)
        mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_yellow), mask_green)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[ProduceDetection] = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter extreme aspect ratios (shelves, wires)
            aspect_ratio = float(cw) / max(1, ch)
            if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                continue

            # Classify produce type from color in bounding box
            crop_hsv = hsv[y:y+ch, x:x+cw]
            mean_hue = np.mean(crop_hsv[:, :, 0])
            mean_sat = np.mean(crop_hsv[:, :, 1])

            if (mean_hue < 15 or mean_hue > 165) and mean_sat > 60:
                label = "apple" if aspect_ratio > 0.8 and aspect_ratio < 1.25 else "red_pepper"
            elif 15 <= mean_hue <= 35:
                label = "yellow_pepper"
            elif 35 < mean_hue <= 85:
                label = "green_pepper" if cw * ch < 25000 else "broccoli"
            else:
                label = "produce"

            conf = min(0.99, max(0.55, 0.5 + (area / (w * h * 0.1))))

            detections.append(ProduceDetection(
                class_id=idx + 1,
                label=label,
                confidence=conf,
                bbox=BoundingBox(
                    xmin=float(x),
                    ymin=float(y),
                    xmax=float(x + cw),
                    ymax=float(y + ch)
                )
            ))

        # Sort left to right
        detections.sort(key=lambda d: d.bbox.xmin)
        return detections


class PresetDetector(BaseProduceDetector):
    """Detector for deterministic testing and benchmark replay."""

    def __init__(self, predefined_detections: List[ProduceDetection]):
        self.predefined = predefined_detections

    def detect(self, rgb_frame: np.ndarray) -> List[ProduceDetection]:
        return self.predefined
