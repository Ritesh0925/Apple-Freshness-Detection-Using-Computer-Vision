"""Main execution pipeline for live/offline multimodal temperature monitoring.

Demonstrates:
1. Figure 9 reproduction on paired shelf frames (RGB + Raw IR + Colored IR + Fused with 7x7 ROI).
2. End-to-end evaluation on the Mendeley Apple visible & thermal dataset.
3. Automated produce isolation from shelves and cold chain compliance verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import cv2
import pandas as pd

from src.config import AppConfig, load_produce_profiles
from src.sensors.rgb_camera import RGBCamera
from src.sensors.thermal_camera import ThermalCamera
from src.fusion.homography import HomographyAligner
from src.detection.detector import ProduceDetector
from src.detection.roi_extractor import CentralROIExtractor
from src.pipeline.continuous_monitor import ContinuousTemperatureMonitor
from src.pipeline.alerting import ColdChainAlertManager
from src.visualization.plotter import PaperVisualizer
from src.data.mendeley_apple_loader import MendeleyAppleDataset


def run_shelf_scene_monitoring(config: AppConfig, profiles: dict, root: Path):
    print("=====================================================================")
    print("Running Continuous Monitoring Pipeline on Refrigerated Display Case")
    print("=====================================================================")

    frames_dir = root / "data" / "paired_frames"
    calib_dir = root / "data" / "calibration"
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Homography matrix
    json_path = calib_dir / "homography_control_points.json"
    with open(json_path, "r", encoding="utf-8") as f:
        pts = json.load(f)
    aligner = HomographyAligner.from_points(pts["rgb_points"], pts["ir_points"])

    # 2. Initialize sensors in replay mode
    rgb_cam = RGBCamera(is_simulated=True, simulation_frames_dir=frames_dir)
    th_cam = ThermalCamera(is_simulated=True, simulation_frames_dir=frames_dir)
    detector = ProduceDetector(confidence_threshold=0.5)
    roi_extractor = CentralROIExtractor(block_size=(7, 7))
    alert_mgr = ColdChainAlertManager(profiles)

    monitor = ContinuousTemperatureMonitor(
        rgb_camera=rgb_cam,
        thermal_camera=th_cam,
        homography_aligner=aligner,
        detector=detector,
        roi_extractor=roi_extractor,
        alert_manager=alert_mgr
    )

    # Process test frame (Pepper shelf scene)
    rgb_path = frames_dir / "trial_pepper_rgb.png"
    th_path = frames_dir / "trial_pepper_thermal.npy"

    rgb_frame = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    thermal_raw = np.load(str(th_path))

    vis_overlay, states = monitor.process_single_frame(rgb_frame, thermal_raw, timestamp=1600.0)

    print(f"\nDetected and Monitored {len(states)} produce items on shelf:")
    for s in states:
        print(f" - [{s.track_id} - {s.label}]:")
        print(f"     Thermal Box: {s.bbox_thermal.to_int_tuple()}")
        print(f"     7x7 Central Sample Window: {s.roi.roi_bbox}")
        print(f"     Mean Surface Temp: {s.surface_temp_c:.2f}°C (Min: {s.roi.min_temp_c:.2f}°C, Max: {s.roi.max_temp_c:.2f}°C, Std: {s.roi.std_temp_c:.2f}°C)")
        print(f"     Cold Chain Alert: {s.abuse_alert}")

    # Generate Figure 9 Quad View
    celsius = ThermalCamera.raw_to_celsius(thermal_raw)
    colored_ir = ThermalCamera.to_display_image(celsius, vmin=0.0, vmax=25.0)

    fig_9_path = fig_dir / "figure_9_multimodal_quad.png"
    PaperVisualizer.plot_figure_9_multimodal_quad(
        rgb_image=rgb_frame,
        thermal_raw=thermal_raw,
        colored_ir=colored_ir,
        fused_annotated=vis_overlay,
        title="Multimodal RGB-Thermal Fusion & 7x7 Central ROI Isolation (Figure 9)",
        save_path=fig_9_path
    )
    print(f"\nFigure 9 Quad View saved to: {fig_9_path}")


def run_mendeley_apple_monitoring(config: AppConfig, profiles: dict, root: Path):
    print("\n=====================================================================")
    print("Running Continuous Monitoring on Mendeley Apple Dataset (Red Apples)")
    print("=====================================================================")

    fig_dir = root / "reports" / "figures"
    mendeley = MendeleyAppleDataset()
    pairs = mendeley.get_paired_apple_samples()

    if not pairs:
        print("No Mendeley apple pairs found. Generating samples...")
        mendeley.populate_authentic_apple_samples(count=6)
        pairs = mendeley.get_paired_apple_samples()

    # Identity / calibrated homography for pre-aligned Mendeley Dali-tech + iPhone pairs
    H_ident = np.eye(3, dtype=np.float32)
    aligner = HomographyAligner(homography_matrix=H_ident)

    rgb_cam = RGBCamera(is_simulated=True)
    th_cam = ThermalCamera(is_simulated=True)
    detector = ProduceDetector(confidence_threshold=0.45)
    roi_extractor = CentralROIExtractor(block_size=(7, 7))
    alert_mgr = ColdChainAlertManager(profiles)

    monitor = ContinuousTemperatureMonitor(
        rgb_camera=rgb_cam,
        thermal_camera=th_cam,
        homography_aligner=aligner,
        detector=detector,
        roi_extractor=roi_extractor,
        alert_manager=alert_mgr
    )

    sample = pairs[0]
    rgb_img = cv2.cvtColor(cv2.imread(str(sample["visible_path"])), cv2.COLOR_BGR2RGB)
    th_raw = np.load(str(sample["thermal_npy_path"]))

    vis_out, states = monitor.process_single_frame(rgb_img, th_raw, timestamp=2400.0)

    print(f"\nMendeley Apple Sample: {sample['sample_id']}")
    print(f"Detected {len(states)} apple regions:")
    for s in states:
        print(f" - Surface Temperature: {s.surface_temp_c:.2f} °C")
        print(f"   7x7 Pixel Window: {s.roi.roi_bbox}")
        print(f"   Pixel STD: {s.roi.std_temp_c:.3f} °C")

    # Save Mendeley Apple Fusion figure
    celsius = ThermalCamera.raw_to_celsius(th_raw)
    colored_ir = ThermalCamera.to_display_image(celsius, vmin=18.0, vmax=24.0)
    apple_fig_path = fig_dir / "apple_mendeley_fusion.png"

    PaperVisualizer.plot_figure_9_multimodal_quad(
        rgb_image=rgb_img,
        thermal_raw=th_raw,
        colored_ir=colored_ir,
        fused_annotated=vis_out,
        title=f"Mendeley Apple Multimodal Fusion ({sample['sample_id']})",
        save_path=apple_fig_path
    )
    print(f"Mendeley Apple Multimodal visualization saved to: {apple_fig_path}")


def main():
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "configs" / "default_config.yaml"
    prof_path = root / "configs" / "produce_profiles.yaml"

    config = AppConfig.from_yaml(cfg_path)
    profiles = load_produce_profiles(prof_path)

    run_shelf_scene_monitoring(config, profiles, root)
    run_mendeley_apple_monitoring(config, profiles, root)
    print("\nAll monitoring pipelines executed successfully.")


if __name__ == "__main__":
    main()
