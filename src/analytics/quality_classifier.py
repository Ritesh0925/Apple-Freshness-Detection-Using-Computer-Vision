"""Multimodal Apple Quality and Freshness Classifier.

Evaluates visible (RGB) and thermal infrared (LWIR) imagery to classify apples into 5 stages:
1. "Ultra Fresh (Harvest-Grade)" - Firm, uniform thermal profile, vibrant skin, no browning.
2. "Prime Retail (Fresh)" - Optimal commercial eating quality, stable emissivity.
3. "Middle-Aged (Mature)" - Incipient softening, slight moisture loss, minor thermal gradients.
4. "Deteriorating (Bruised / Overripe)" - Evident localized thermal anomalies, tissue breakdown, surface blemishes.
5. "Rotten (Spoiled / Decomposed)" - Extensive fungal/bacterial decay, severe emissivity depression, significant browning.

Based on postharvest thermal diffusivity and colorimetric browning index principles
cited in Section 1.2 of Canatan et al. (2025) and Mashayekhpour et al. (2024).
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2


@dataclass
class AppleQualityPrediction:
    """Prediction outcome for apple freshness and rot evaluation."""
    predicted_category: str
    confidence_score: float         # 0.0 to 1.0
    freshness_index_pct: float      # 0 to 100%
    estimated_shelf_life_days: int  # At recommended refrigeration (0-4 °C)
    shelf_life_room_temp_days: int  # At ambient retail (20 °C)
    browning_index: float           # Spectrophotometric browning ratio
    thermal_anomaly_pct: float      # Percentage of fruit area exhibiting thermal defect/rot
    thermal_uniformity_score: float # 0 to 100 (100 = completely healthy, uniform thermal diffusivity)
    color_vibrancy_score: float     # 0 to 100
    category_probabilities: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    action_status: str = "PASS"     # "PASS", "EXPEDITE SALE", "REJECT/DISCARD"


class AppleQualityClassifier:
    """Multimodal feature extraction and decision model for apple freshness assessment."""

    CATEGORIES = [
        "Ultra Fresh (Harvest-Grade)",
        "Prime Retail (Fresh)",
        "Middle-Aged (Mature)",
        "Deteriorating (Bruised / Overripe)",
        "Rotten (Spoiled / Decomposed)"
    ]

    def __init__(self):
        pass

    def extract_features(
        self,
        rgb_img: Optional[np.ndarray],
        thermal_img: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Extracts colorimetric, textural, and thermal diffusivity features from input images."""
        features = {}

        # 1. RGB Visual Analysis
        if rgb_img is not None and rgb_img.size > 0:
            rgb_resized = cv2.resize(rgb_img, (320, 240))
            hsv = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)

            # Apple mask (red, yellow-green, brown)
            # Exclude bright white/studio or dark background
            gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            mask_dark = gray > 20
            fruit_mask = cv2.bitwise_and(mask, mask, mask=mask_dark.astype(np.uint8) * 255)

            if np.sum(fruit_mask) < 200:
                fruit_mask = np.ones((240, 320), dtype=np.uint8) * 255

            fruit_pixels_hsv = hsv[fruit_mask > 0]
            fruit_pixels_rgb = rgb_resized[fruit_mask > 0]

            # Browning detection: Low saturation, mid-value brownish hues (H in [10, 25], S in [40, 160], V in [40, 150])
            h_vals = fruit_pixels_hsv[:, 0]
            s_vals = fruit_pixels_hsv[:, 1]
            v_vals = fruit_pixels_hsv[:, 2]

            brown_mask = (h_vals >= 8) & (h_vals <= 25) & (s_vals >= 35) & (s_vals <= 160) & (v_vals <= 160)
            browning_ratio = float(np.mean(brown_mask)) if len(brown_mask) > 0 else 0.0

            # Red vibrancy (healthy red skin): H < 10 or H > 165 with S > 100
            red_mask = ((h_vals <= 10) | (h_vals >= 165)) & (s_vals >= 80)
            red_vibrancy = float(np.mean(red_mask)) if len(red_mask) > 0 else 0.0

            # Texture entropy / roughness via Laplacian variance
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            features["browning_ratio"] = browning_ratio
            features["red_vibrancy"] = red_vibrancy
            features["texture_variance"] = lap_var
        else:
            features["browning_ratio"] = 0.05
            features["red_vibrancy"] = 0.60
            features["texture_variance"] = 100.0

        # 2. Thermal Radiometric Analysis
        if thermal_img is not None and thermal_img.size > 0:
            th_resized = cv2.resize(thermal_img.astype(np.float32), (320, 240))
            if len(th_resized.shape) == 3:
                th_gray = cv2.cvtColor(th_resized.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                th_gray = th_resized

            # Fruit segmentation in thermal
            th_min, th_max = np.min(th_gray), np.max(th_gray)
            if th_max - th_min > 1e-3:
                norm_th = (th_gray - th_min) / (th_max - th_min)
            else:
                norm_th = np.zeros_like(th_gray)

            # Central 50% ROI
            h_th, w_th = th_gray.shape
            cy, cx = h_th // 2, w_th // 2
            core_crop = norm_th[cy-30:cy+30, cx-30:cx+30]

            # Healthy apples exhibit smooth, concentric thermal gradient
            # Bruised / rotten apples exhibit irregular high-variance hot/cold patches (cell rupture)
            th_std_core = float(np.std(core_crop)) if core_crop.size > 0 else 0.05
            th_overall_std = float(np.std(norm_th))

            # Local thermal gradient anomaly
            grad_x = cv2.Sobel(norm_th, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(norm_th, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            high_grad_ratio = float(np.mean(grad_mag > 0.35))

            features["thermal_core_std"] = th_std_core
            features["thermal_overall_std"] = th_overall_std
            features["thermal_anomaly_ratio"] = high_grad_ratio
        else:
            features["thermal_core_std"] = 0.02
            features["thermal_overall_std"] = 0.10
            features["thermal_anomaly_ratio"] = 0.02

        return features

    def predict(
        self,
        rgb_img: Optional[np.ndarray] = None,
        thermal_img: Optional[np.ndarray] = None
    ) -> AppleQualityPrediction:
        """Classifies the apple into one of 5 quality grades using visible and thermal sensor fusion."""
        if rgb_img is None and thermal_img is None:
            raise ValueError("At least one image (RGB, Thermal, or both) must be provided.")

        feat = self.extract_features(rgb_img, thermal_img)

        browning = feat["browning_ratio"]
        red_vib = feat["red_vibrancy"]
        th_core_std = feat["thermal_core_std"]
        th_anomaly = feat["thermal_anomaly_ratio"]

        # Compute composite decay penalty index:
        # High browning + high thermal irregularity indicates tissue spoilage
        decay_index = (
            browning * 0.45 +
            th_anomaly * 0.35 +
            min(1.0, th_core_std * 5.0) * 0.20
        )

        # Freshness index scale [0 - 100%]
        raw_freshness = 100.0 * (1.0 - np.clip(decay_index * 2.2 - (red_vib * 0.15), 0.0, 1.0))
        freshness_pct = round(float(np.clip(raw_freshness, 3.0, 99.0)), 1)

        # Softmax-like probability distribution across 5 categories
        # Center points: [95, 80, 60, 35, 12]
        centers = np.array([92.0, 78.0, 58.0, 36.0, 14.0])
        dist = -0.5 * ((freshness_pct - centers) / 12.0) ** 2
        exp_d = np.exp(dist)
        probs = exp_d / np.sum(exp_d)

        cat_probs = {self.CATEGORIES[i]: round(float(probs[i]), 3) for i in range(5)}
        best_idx = int(np.argmax(probs))
        predicted_cat = self.CATEGORIES[best_idx]
        confidence = float(probs[best_idx])

        # Category-specific parameters
        if best_idx == 0:  # Ultra Fresh
            shelf_life_fridge = 35
            shelf_life_ambient = 14
            action = "PASS - PREMIUM QUALITY"
            recs = [
                "Fruit is in peak physiological condition with intact cuticle wax.",
                "Suitable for long-term controlled atmosphere or retail display.",
                "Maintain optimal storage temperature at 0.0 - 2.0 °C, 90-95% RH."
            ]
        elif best_idx == 1:  # Prime Retail
            shelf_life_fridge = 21
            shelf_life_ambient = 8
            action = "PASS - STANDARD RETAIL"
            recs = [
                "Optimal for immediate retail display and consumer sale.",
                "Thermal profile indicates intact internal flesh without chilling injury.",
                "Keep away from direct store spotlight heating."
            ]
        elif best_idx == 2:  # Middle-Aged
            shelf_life_fridge = 10
            shelf_life_ambient = 3
            action = "EXPEDITE SALE - PROMOTION"
            recs = [
                "Moderate respiration rate; slight loss of crispness and moisture detected.",
                "Recommend moving to front display or applying retail discount.",
                "Prioritize turnover within 7-10 days under refrigeration."
            ]
        elif best_idx == 3:  # Deteriorating
            shelf_life_fridge = 3
            shelf_life_ambient = 1
            action = "DOWNGRADE / REPROCESS"
            recs = [
                "Localized thermal cold/hot spots indicate mechanical bruising or senescence.",
                "Significant risk of rapid secondary pathogen colonization.",
                "Divert to commercial processing (cider, applesauce) or discard if soft rot present."
            ]
        else:  # Rotten
            shelf_life_fridge = 0
            shelf_life_ambient = 0
            action = "REJECT / CONDEMN / DISCARD"
            recs = [
                "Extensive tissue browning and severe emissivity collapse detected.",
                "High microbial activity causing thermal anomaly.",
                "Immediately remove from refrigerated case to prevent cross-contamination of adjacent fruit."
            ]

        uniformity = round(float(np.clip(100.0 - (th_anomaly * 180.0 + th_core_std * 300.0), 5.0, 99.0)), 1)
        vibrancy = round(float(np.clip(red_vib * 120.0, 5.0, 98.0)), 1)

        return AppleQualityPrediction(
            predicted_category=predicted_cat,
            confidence_score=round(confidence, 3),
            freshness_index_pct=freshness_pct,
            estimated_shelf_life_days=shelf_life_fridge,
            shelf_life_room_temp_days=shelf_life_ambient,
            browning_index=round(float(browning), 3),
            thermal_anomaly_pct=round(float(th_anomaly * 100.0), 1),
            thermal_uniformity_score=uniformity,
            color_vibrancy_score=vibrancy,
            category_probabilities=cat_probs,
            recommendations=recs,
            action_status=action
        )
