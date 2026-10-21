"""Visible (RGB) camera module interface for Raspberry Pi Camera v2 / USB.

Captures high-resolution visual frames used for object detection and homography warping.
"""

import time
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

from src.sensors.base import BaseImagingSensor
from src.core.exceptions import CameraAcquisitionError


class RGBCamera(BaseImagingSensor):
    """Visible light camera interface (Pi Camera Module v2 / USB webcam / Replay)."""

    def __init__(
        self,
        device_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        is_simulated: bool = False,
        simulation_frames_dir: Optional[Path] = None
    ):
        self.device_index = device_index
        self.width, self.height = resolution
        self.fps = fps
        self.is_simulated = is_simulated
        self.sim_dir = simulation_frames_dir
        self.cap: Optional[cv2.VideoCapture] = None
        self._sim_files = []
        self._sim_index = 0

    def initialize(self) -> bool:
        if self.is_simulated:
            if self.sim_dir and self.sim_dir.exists():
                self._sim_files = sorted(
                    list(self.sim_dir.glob("*.jpg")) + list(self.sim_dir.glob("*.png"))
                )
            return True

        try:
            self.cap = cv2.VideoCapture(self.device_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            return self.cap.isOpened()
        except Exception:
            self.is_simulated = True
            return True

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        t = time.time()
        if self.is_simulated:
            if self._sim_files:
                fpath = self._sim_files[self._sim_index % len(self._sim_files)]
                self._sim_index += 1
                bgr = cv2.imread(str(fpath))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return True, rgb, t
            else:
                rgb = self._generate_synthetic_shelf_rgb()
                return True, rgb, t

        if self.cap is None or not self.cap.isOpened():
            raise CameraAcquisitionError("RGB camera is not opened.")

        ret, bgr_frame = self.cap.read()
        if not ret:
            return False, None, t
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return True, rgb_frame, t

    def _generate_synthetic_shelf_rgb(self) -> np.ndarray:
        """Generates realistic synthetic RGB frame depicting refrigerated display shelf with produce."""
        # Gray metallic refrigerator shelf
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 180
        # Draw shelf rack bars
        for y in range(40, self.height - 40, 25):
            cv2.line(img, (20, y), (self.width - 20, y), (120, 120, 120), 2)

        # Draw 3 produce items
        # Apple (Red circle)
        cv2.circle(img, (160, 240), 65, (190, 20, 30), -1)
        cv2.circle(img, (155, 235), 25, (220, 60, 60), -1)  # highlight

        # Bell Pepper (Yellow rounded rect / polygon)
        cv2.ellipse(img, (340, 240), (60, 75), 0, 0, 360, (230, 190, 20), -1)

        # Broccoli (Green crown with wrapping glint)
        cv2.circle(img, (510, 235), 75, (35, 110, 45), -1)
        # Plastic wrapping subtle reflection
        cv2.line(img, (480, 200), (530, 260), (220, 240, 230), 2)

        return img

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
