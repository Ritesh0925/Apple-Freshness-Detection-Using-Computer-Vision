"""Configuration loader and schema validation using Pydantic and PyYAML."""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field


class RGBCameraConfig(BaseModel):
    module: str = "Raspberry Pi Camera Module v2"
    native_resolution: Tuple[int, int] = (3280, 2464)
    capture_resolution: Tuple[int, int] = (640, 480)
    fps: int = 30


class ThermalCameraConfig(BaseModel):
    module: str = "FLIR Lepton 3.5 (PureThermal 2 Smart I/O)"
    native_resolution: Tuple[int, int] = (160, 120)
    radiometric_scale: float = 100.0
    fps: float = 7.0
    spectral_range_um: Tuple[float, float] = (8.0, 14.0)
    temperature_range_c: Tuple[float, float] = (-10.0, 140.0)


class CameraConfig(BaseModel):
    rgb: RGBCameraConfig = Field(default_factory=RGBCameraConfig)
    thermal: ThermalCameraConfig = Field(default_factory=ThermalCameraConfig)


class DefaultControlPoints(BaseModel):
    src_rgb: List[List[float]] = Field(default_factory=lambda: [
        [50.0, 40.0], [590.0, 45.0], [580.0, 440.0], [60.0, 430.0]
    ])
    dst_ir: List[List[float]] = Field(default_factory=lambda: [
        [15.0, 10.0], [145.0, 12.0], [140.0, 110.0], [18.0, 108.0]
    ])


class FusionConfig(BaseModel):
    method: str = "homography_perspective"
    ransac_reproj_threshold: float = 3.0
    target_space: str = "thermal"
    default_control_points: DefaultControlPoints = Field(default_factory=DefaultControlPoints)


class DetectionConfig(BaseModel):
    detector_type: str = "efficientdet_tflite"
    confidence_threshold: float = 0.50
    nms_threshold: float = 0.40
    supported_classes: List[str] = Field(default_factory=lambda: ["apple", "bell_pepper", "broccoli", "produce"])


class ROIConfig(BaseModel):
    grid_division: Tuple[int, int] = (7, 7)
    center_block_size: Tuple[int, int] = (7, 7)
    aggregation: str = "mean"


class TemporalFilterConfig(BaseModel):
    sampling_hz: float = 1.0
    moving_average_window: int = 5
    deduplicate_timestamps: bool = True


class WaterBathCalibConfig(BaseModel):
    low_range_c: Tuple[float, float] = (0.0, 53.0)
    high_range_c: Tuple[float, float] = (53.0, 100.0)
    correction_slope: float = 0.985
    correction_intercept: float = 0.15


class CalibrationConfig(BaseModel):
    water_bath: WaterBathCalibConfig = Field(default_factory=WaterBathCalibConfig)


class SystemConfig(BaseModel):
    name: str = "RGB-Thermal Fresh Produce Surface Temperature Monitoring"
    log_level: str = "INFO"
    device: str = "raspberry_pi_4b"


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    roi: ROIConfig = Field(default_factory=ROIConfig)
    temporal_filtering: TemporalFilterConfig = Field(default_factory=TemporalFilterConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_produce_profiles(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("profiles", {})
