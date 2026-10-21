"""Radiative physics and emissivity compensation for produce surface temperature monitoring.

Accounts for:
- Fresh produce high-water-content surface emissivity (epsilon ~ 0.95 - 0.98).
- Packaging film reflectivity and transmission (polyethylene film wrap for broccoli, Section 3.3.2).
- Ambient refrigeration cabinet reflected radiation.
"""

from typing import Optional
import numpy as np


class EmissivityCorrector:
    """Corrects apparent radiometric temperature for surface emissivity and background reflections."""

    STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 * K^4)

    def __init__(
        self,
        default_emissivity: float = 0.96,
        ambient_temp_c: float = 4.0,
        packaging_transmittance: float = 1.0
    ):
        """
        Args:
            default_emissivity: Surface emissivity epsilon in [0.1, 1.0].
            ambient_temp_c: Surrounding ambient refrigeration temperature (°C).
            packaging_transmittance: Packaging transmission tau in [0.0, 1.0].
        """
        self.emissivity = default_emissivity
        self.ambient_temp_c = ambient_temp_c
        self.tau = packaging_transmittance

    def correct_temperature(
        self,
        apparent_temp_c: float,
        emissivity: Optional[float] = None,
        ambient_temp_c: Optional[float] = None,
        packaging_transmittance: Optional[float] = None
    ) -> float:
        """Calculates true surface temperature using Stefan-Boltzmann radiation balance:

        W_total = tau * [epsilon * sigma * T_obj^4 + (1 - epsilon) * sigma * T_amb^4] + (1 - tau) * sigma * T_film^4

        For thin wrap, approximating T_film ~ T_amb:
        T_obj = [ (T_apparent^4 - (1 - tau * epsilon) * T_amb^4) / (tau * epsilon) ]^(1/4)
        """
        eps = emissivity if emissivity is not None else self.emissivity
        t_amb = ambient_temp_c if ambient_temp_c is not None else self.ambient_temp_c
        tau = packaging_transmittance if packaging_transmittance is not None else self.tau

        # Convert to Kelvin
        t_app_k = apparent_temp_c + 273.15
        t_amb_k = t_amb + 273.15

        effective_emission = tau * eps
        if effective_emission <= 0.0 or effective_emission > 1.0:
            effective_emission = 0.95

        numerator = (t_app_k ** 4) - ((1.0 - effective_emission) * (t_amb_k ** 4))
        if numerator <= 0:
            return apparent_temp_c

        t_true_k = (numerator / effective_emission) ** 0.25
        return float(t_true_k - 273.15)
