"""FLIR Lepton 3.5 thermal infrared camera interface and radiometric processing.

Supports:
- FLIR Lepton 3.5 (160x120 resolution, 8-14 um LWIR) via PureThermal 2 Smart I/O board.
- Radiometric 16-bit raw data acquisition (centiKelvin -> Celsius).
- 8-bit normalization and colormap visualization (Fig. 5, step 2 & Fig. 9).
- Replay/Mock sensor for testing and dataset reproduction.
"""

import time
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

from src.sensors.base import BaseImagingSensor
from src.core.exceptions import CameraAcquisitionError


class ThermalCamera(BaseImagingSensor):
    """FLIR Lepton 3.5 thermal camera interface."""

    def __init__(
        self,
        device_index: int = 0,
        resolution: Tuple[int, int] = (160, 120),
        radiometric_scale: float = 100.0,
        is_simulated: bool = False,
        simulation_frames_dir: Optional[Path] = None
    ):
        """
        Args:
            device_index: V4L2 device index for PureThermal 2 USB UVC.
            resolution: (width, height) = (160, 120).
            radiometric_scale: Scale factor (100.0 for centiKelvin).
            is_simulated: If True, streams frames from disk.
            simulation_frames_dir: Directory containing .npy or 16-bit raw frames.
        """
        self.device_index = device_index
        self.width, self.height = resolution
        self.scale = radiometric_scale
        self.is_simulated = is_simulated
        self.sim_dir = simulation_frames_dir
        self.cap: Optional[cv2.VideoCapture] = None
        self._sim_files = []
        self._sim_index = 0

    def initialize(self) -> bool:
        """Initializes camera connection or locates test frames."""
        if self.is_simulated:
            if self.sim_dir and self.sim_dir.exists():
                self._sim_files = sorted(list(self.sim_dir.glob("*.npy")))
            return True

        try:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device_index)
            # Request raw 16-bit radiometric format if supported by PureThermal UVC
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return self.cap.isOpened()
        except Exception:
            # Fallback to simulated mode if hardware camera is not physically plugged in
            self.is_simulated = True
            return True

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Captures a raw 16-bit radiometric thermal frame.

        Returns:
            Tuple of (success, raw_16bit_frame, timestamp).
        """
        t = time.time()
        if self.is_simulated:
            if self._sim_files:
                fpath = self._sim_files[self._sim_index % len(self._sim_files)]
                self._sim_index += 1
                raw_data = np.load(fpath)
                return True, raw_data, t
            else:
                # Generate a default refrigerated scene (shelves at 4°C, produce cooling)
                raw_data = self._generate_synthetic_shelf_frame()
                return True, raw_data, t

        if self.cap is None or not self.cap.isOpened():
            raise CameraAcquisitionError("Thermal camera is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            return False, None, t
        return True, frame, t

    @classmethod
    def raw_to_celsius(cls, raw_frame: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """Converts raw 16-bit Lepton centiKelvin values to degrees Celsius (°C).

        Formula: T_C = (raw_value / 100.0) - 273.15
        """
        kelvin = raw_frame.astype(np.float32) / scale
        celsius = kelvin - 273.15
        return celsius

    @classmethod
    def celsius_to_raw(cls, celsius_frame: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """Converts degrees Celsius (°C) back to 16-bit raw centiKelvin."""
        kelvin = (celsius_frame + 273.15) * scale
        return np.clip(kelvin, 0, 65535).astype(np.uint16)

    @classmethod
    def to_display_image(
        cls,
        celsius_frame: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Converts Celsius frame into normalized 8-bit colormapped visualization (Paper Fig. 5, 9)."""
        vmin = float(np.min(celsius_frame)) if vmin is None else vmin
        vmax = float(np.max(celsius_frame)) if vmax is None else vmax

        if vmax - vmin < 1e-4:
            vmax = vmin + 1.0

        normalized = np.clip((celsius_frame - vmin) / (vmax - vmin), 0.0, 1.0)
        gray8 = (normalized * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(gray8, colormap)
        return colored

    def _generate_synthetic_shelf_frame(self) -> np.ndarray:
        """Generates realistic synthetic 160x120 thermal frame with background shelf and cold produce."""
        # Baseline refrigerator ambient ~4.2 °C with subtle noise
        base_c = 4.2 + np.random.normal(0, 0.15, (self.height, self.width))
        # Add 3 produce items on shelves
        # Apple (center-left), Bell pepper (center), Broccoli (center-right)
        centers = [(40, 60, 18, 8.5), (85, 60, 20, 9.2), (130, 60, 22, 11.0)]
        for cx, cy, rad, temp in centers:
            y, x = np.ogrid[:self.height, :self.width]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = dist <= rad
            base_c[mask] = temp - (dist[mask] / rad) * 1.5 + np.random.normal(0, 0.1, np.sum(mask))

        return self.celsius_to_raw(base_c, self.scale)

    def release(self) -> None:
        """Closes camera handle."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
