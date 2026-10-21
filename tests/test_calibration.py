"""Unit tests for water bath temperature calibration and emissivity correction."""

import numpy as np
import pytest
from src.calibration.water_bath_calibrator import WaterBathCalibrator
from src.calibration.emissivity import EmissivityCorrector


def test_water_bath_calibration_fit():
    calibrator = WaterBathCalibrator(split_temp_c=53.0)

    # Synthetic range from 0 to 100 °C with known offset
    tc = np.linspace(0.0, 100.0, 500)
    # IR reads lower: IR = 0.95 * TC - 1.0
    ir = 0.95 * tc - 1.0

    res = calibrator.fit(tc, ir)

    assert "0-100 C" in res
    assert "0-53 C" in res
    assert "53-100 C" in res

    # Correct reading at 25 °C
    ir_val = 0.95 * 25.0 - 1.0
    corr = calibrator.calibrate(ir_val, piecewise=True)
    assert pytest.approx(corr, abs=0.5) == 25.0


def test_emissivity_correction():
    corrector = EmissivityCorrector(default_emissivity=0.96, ambient_temp_c=4.0)
    # For high emissivity and close to ambient, correction should be close to apparent
    corr = corrector.correct_temperature(apparent_temp_c=6.0)
    assert 5.0 <= corr <= 7.0

    # Wrapped packaging film (lower transmittance)
    corr_wrapped = corrector.correct_temperature(
        apparent_temp_c=6.0, emissivity=0.88, packaging_transmittance=0.80
    )
    assert isinstance(corr_wrapped, float)
