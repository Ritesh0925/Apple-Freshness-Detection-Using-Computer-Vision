"""Cold chain quality compliance and temperature abuse alerting engine.

Based on Food Loss and Waste (FLW) standards cited in Sections 1 & 1.1:
Monitors produce storage temperatures against recommended thresholds to prevent spoilage.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class TemperatureAlert:
    """Emitted alert when produce temperature violates cold chain safety limits."""
    timestamp: float
    produce_label: str
    current_temp_c: float
    allowed_range_c: List[float]
    severity: str  # "WARNING" or "CRITICAL"
    message: str


class ColdChainAlertManager:
    """Monitors live temperatures and flags temperature abuse events."""

    def __init__(self, produce_profiles: Dict[str, Any]):
        self.profiles = produce_profiles
        self.active_alerts: List[TemperatureAlert] = []

    def check_temperature(
        self,
        produce_type: str,
        current_temp_c: float,
        timestamp: float
    ) -> Optional[TemperatureAlert]:
        """Checks if current temperature exceeds recommended bounds."""
        key = produce_type.lower().replace(" ", "_")
        profile = None

        for p_key, p_val in self.profiles.items():
            if p_key in key or key in p_key:
                profile = p_val
                break

        if not profile:
            # Default conservative fresh produce range: 0.0 to 10.0 °C
            allowed = [0.0, 10.0]
        else:
            allowed = profile.get("storage_temp_recommended_c", [0.0, 10.0])

        if current_temp_c > allowed[1]:
            diff = current_temp_c - allowed[1]
            severity = "CRITICAL" if diff > 5.0 else "WARNING"
            alert = TemperatureAlert(
                timestamp=timestamp,
                produce_label=produce_type,
                current_temp_c=current_temp_c,
                allowed_range_c=allowed,
                severity=severity,
                message=f"Temperature abuse detected for {produce_type}: {current_temp_c:.2f}°C exceeds max threshold {allowed[1]}°C by +{diff:.2f}°C"
            )
            self.active_alerts.append(alert)
            return alert
        elif current_temp_c < allowed[0]:
            diff = allowed[0] - current_temp_c
            alert = TemperatureAlert(
                timestamp=timestamp,
                produce_label=produce_type,
                current_temp_c=current_temp_c,
                allowed_range_c=allowed,
                severity="WARNING",
                message=f"Freezing injury risk for {produce_type}: {current_temp_c:.2f}°C is below minimum {allowed[0]}°C"
            )
            self.active_alerts.append(alert)
            return alert

        return None
