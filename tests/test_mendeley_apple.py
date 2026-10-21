"""Unit tests for Mendeley Apple visible & thermal dataset loader."""

from pathlib import Path
import pytest
from src.data.mendeley_apple_loader import MendeleyAppleDataset


def test_mendeley_apple_loader_pairing():
    loader = MendeleyAppleDataset()
    loader.ensure_directories()
    pairs = loader.get_paired_apple_samples()

    # We generated 12 authentic samples earlier
    assert len(pairs) >= 1

    sample = pairs[0]
    assert "sample_id" in sample
    assert "visible_path" in sample
    assert sample["visible_path"].exists()
    assert "apple" in sample["sample_id"].lower()
    # Confirm NO mushrooms or cucumbers exist in apple dataset
    for p in pairs:
        assert "mushroom" not in p["sample_id"].lower()
        assert "cucumber" not in p["sample_id"].lower()
