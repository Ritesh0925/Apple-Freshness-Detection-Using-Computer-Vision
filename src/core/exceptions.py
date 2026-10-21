"""Domain-specific exceptions for RGB-Thermal sensor fusion and monitoring."""

class SensorFusionError(Exception):
    """Base exception for all sensor fusion and temperature monitoring errors."""
    pass

class CameraAcquisitionError(SensorFusionError):
    """Raised when frame acquisition from RGB or thermal sensor fails."""
    pass

class HomographyComputationError(SensorFusionError):
    """Raised when homography matrix computation fails (e.g. collinear points, RANSAC failure)."""
    pass

class DetectionError(SensorFusionError):
    """Raised when object detection fails or produces invalid coordinates."""
    pass

class OutOfBoundsROIError(SensorFusionError):
    """Raised when the calculated central 7x7 ROI falls outside sensor matrix boundaries."""
    pass

class CalibrationError(SensorFusionError):
    """Raised when temperature calibration parameters cannot be determined or are invalid."""
    pass

class TimestampSyncError(SensorFusionError):
    """Raised when thermocouple and optical/thermal timestamps cannot be aligned."""
    pass
