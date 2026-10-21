"""Abstract base class definitions for imaging and temperature sensors."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import numpy as np


class BaseImagingSensor(ABC):
    """Abstract interface for imaging sensors (RGB and Thermal)."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initializes sensor hardware or data stream."""
        pass

    @abstractmethod
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Captures a single frame.

        Returns:
            Tuple of (success_flag, frame_array, timestamp_epoch).
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Safely shuts down sensor connections."""
        pass
