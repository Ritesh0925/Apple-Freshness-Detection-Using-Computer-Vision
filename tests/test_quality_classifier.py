"""Unit tests for Apple Freshness and Quality Classifier."""

import numpy as np
import pytest
import cv2
from src.analytics.quality_classifier import AppleQualityClassifier


def test_fresh_apple_prediction():
    classifier = AppleQualityClassifier()

    # Create synthetic healthy vibrant red apple RGB
    rgb = np.ones((240, 320, 3), dtype=np.uint8) * 240
    cv2.circle(rgb, (160, 120), 80, (200, 25, 35), -1)

    # Uniform healthy thermal image
    thermal = np.ones((240, 320), dtype=np.float32) * 20.0
    cv2.circle(thermal, (160, 120), 80, 21.0, -1)

    pred = classifier.predict(rgb, thermal)
    assert pred.freshness_index_pct >= 70.0
    assert pred.predicted_category in [
        "Ultra Fresh (Harvest-Grade)",
        "Prime Retail (Fresh)"
    ]
    assert pred.estimated_shelf_life_days >= 15
    assert pred.action_status.startswith("PASS")


def test_rotten_apple_prediction():
    classifier = AppleQualityClassifier()

    # Create synthetic rotten brown discolored apple RGB
    rgb = np.ones((240, 320, 3), dtype=np.uint8) * 240
    # Brownish/decayed hue
    cv2.circle(rgb, (160, 120), 80, (90, 60, 30), -1)

    # Irregular patchy thermal image with high gradient anomalies
    thermal = np.random.normal(20.0, 4.0, (240, 320)).astype(np.float32)

    pred = classifier.predict(rgb, thermal)
    assert pred.freshness_index_pct <= 45.0
    assert pred.predicted_category in [
        "Deteriorating (Bruised / Overripe)",
        "Rotten (Spoiled / Decomposed)"
    ]
    assert pred.estimated_shelf_life_days <= 5
    assert "REJECT" in pred.action_status or "DOWNGRADE" in pred.action_status


def test_prediction_with_only_rgb():
    classifier = AppleQualityClassifier()
    rgb = np.ones((240, 320, 3), dtype=np.uint8) * 200
    cv2.circle(rgb, (160, 120), 80, (180, 30, 40), -1)

    pred = classifier.predict(rgb_img=rgb, thermal_img=None)
    assert pred.predicted_category in AppleQualityClassifier.CATEGORIES
    assert 0.0 <= pred.confidence_score <= 1.0


def test_prediction_with_only_thermal():
    classifier = AppleQualityClassifier()
    thermal = np.ones((240, 320), dtype=np.float32) * 21.0
    cv2.circle(thermal, (160, 120), 80, 22.0, -1)

    pred = classifier.predict(rgb_img=None, thermal_img=thermal)
    assert pred.predicted_category in AppleQualityClassifier.CATEGORIES
    assert 0.0 <= pred.confidence_score <= 1.0


def test_empty_input_raises_error():
    classifier = AppleQualityClassifier()
    with pytest.raises(ValueError):
        classifier.predict(None, None)
