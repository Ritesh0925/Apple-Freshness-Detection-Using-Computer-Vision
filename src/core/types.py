"""Typed data models and structures for multimodal fusion and temperature monitoring."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates [xmin, ymin, xmax, ymax]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center(self) -> Tuple[float, float]:
        """Geometric center (cx, cy) of the bounding box."""
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.xmin)), int(round(self.ymin)),
                int(round(self.xmax)), int(round(self.ymax)))


@dataclass
class ProduceDetection:
    """Detected produce item with class label, score, and bounding box."""
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CentralROIExtraction:
    """Geometric center 7x7 pixel ROI extracted from thermal image."""
    center_coord: Tuple[int, int]         # (cx, cy) in thermal frame
    roi_bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    grid_size: Tuple[int, int]            # (7, 7)
    pixel_values_celsius: np.ndarray      # 49 values (7x7) in °C
    mean_temp_c: float
    min_temp_c: float
    max_temp_c: float
    std_temp_c: float


@dataclass
class MonitoredObjectState:
    """State of an individual produce item monitored over time."""
    track_id: str
    label: str
    timestamp: float
    bbox_rgb: BoundingBox
    bbox_thermal: BoundingBox
    roi: CentralROIExtraction
    surface_temp_c: float
    moving_avg_temp_c: Optional[float] = None
    thermocouple_temp_c: Optional[float] = None
    abuse_alert: bool = False


@dataclass
class StatisticalMetrics:
    """Evaluation metrics according to Eqs. 5 and 6 of the paper."""
    rmse: float     # Root Mean Square Error (°C)
    mae: float      # Mean Absolute Error (°C)
    r2: float       # Coefficient of determination R²
    n_samples: int
    bias: float     # Mean signed difference (IR - TC)
    max_abs_diff: float
    min_abs_diff: float


@dataclass
class CalibrationResult:
    """Water-bath calibration regression parameters and performance."""
    temperature_range: str
    slope: float
    intercept: float
    metrics: StatisticalMetrics
