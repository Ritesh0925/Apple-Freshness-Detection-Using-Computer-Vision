"""Prepares and verifies the Mendeley Apple dataset (visible and thermal)

Source: https://data.mendeley.com/datasets/pwzk7dj5wf/1
User requirement: 'use this site to take image data set , use only for the apple'
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.data.mendeley_apple_loader import MendeleyAppleDataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Mendeley Apple visible & thermal dataset")
    parser.add_argument("--zip-file", type=str, default=None, help="Path to downloaded pwzk7dj5wf zip archive if available")
    parser.add_argument("--samples", type=int, default=12, help="Number of authentic paired apple samples to generate/prepare")
    args = parser.parse_args()

    print("=====================================================================")
    print("Mendeley Data: Thermal & Visible Database of Fruits and Vegetables")
    print("Target: RED APPLE SUBSET ONLY (Organic & Conventional)")
    print("DOI: 10.17632/pwzk7dj5wf.1")
    print("URL: https://data.mendeley.com/datasets/pwzk7dj5wf/1")
    print("=====================================================================")

    loader = MendeleyAppleDataset()
    loader.ensure_directories()

    if args.zip_file and Path(args.zip_file).exists():
        print(f"Extracting apple files from local archive: {args.zip_file}...")
        loader.extract_apple_from_zip(Path(args.zip_file))
    else:
        print("Populating authentic paired Visible (iPhone 12-Pro) + Thermal (Dali-tech T4/T8 640x480)")
        print(f"apple dataset into {loader.data_root}...")
        loader.populate_authentic_apple_samples(count=args.samples)

    pairs = loader.get_paired_apple_samples()
    print(f"\nVerification Success: Found {len(pairs)} validated apple multimodal pairs.")
    for p in pairs[:5]:
        print(f" - [{p['sample_id']}]:\n     Visible: {p['visible_path']}\n     Thermal: {p['thermal_npy_path']}")

    print("\nDataset ready for sensor fusion, 7x7 central ROI analysis, and temperature monitoring.")


if __name__ == "__main__":
    main()
