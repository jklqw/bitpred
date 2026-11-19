"""Unit tests for X-day period feature engineering, modeling, and real-time action engine."""

import os
import sys
from pathlib import Path

# Automatically ensure project root and .venv site-packages are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SITE_PACKAGES = _PROJECT_ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if _SITE_PACKAGES.is_dir() and str(_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_SITE_PACKAGES))

import unittest
import numpy as np
import pandas as pd
from features import build_x_period_dataset, extract_x_period_features_for_window
from train import train_x_period_pipeline, get_x_period_artifact_path
from signal_engine import generate_x_period_action



class TestXPeriodPipeline(unittest.TestCase):
    """Test suite ensuring correctness, non-leaky timing, and formula accuracy."""

    def test_x_day_dataset_dimensions_and_target(self):
        """Verify X-day period dataset construction and target definition."""
        x_days = 4.0
        dataset, meta = build_x_period_dataset(x_days=x_days, step_hours=12)

        self.assertGreater(len(dataset), 50)
        self.assertEqual(meta["x_days"], 4.0)
        self.assertEqual(meta["total_period_hours"], 96)
        self.assertEqual(meta["half_period_hours"], 48)
        self.assertEqual(meta["prev_week_offset_hours"], 168)

        # Check target columns
        self.assertIn("delta_p", dataset.columns)
        self.assertIn("delta_p_pct", dataset.columns)
        self.assertIn("target_dir", dataset.columns)

        # Check target formula consistency: delta_p == p_end - p_start
        expected_delta = dataset["p_end"] - dataset["p_start"]
        np.testing.assert_allclose(dataset["delta_p"].values, expected_delta.values, rtol=1e-5)

        # Direction should be 1 if delta_p > 0 else 0
        expected_dir = (dataset["delta_p"] > 0).astype(int)
        np.testing.assert_array_equal(dataset["target_dir"].values, expected_dir.values)

    def test_x_period_features_presence(self):
        """Verify that Window 1, Window 2, and comparative features are present."""
        dataset, _ = build_x_period_dataset(x_days=4.0, step_hours=24)

        # Window 1 features
        self.assertIn("ret_curr", dataset.columns)
        self.assertIn("range_curr", dataset.columns)
        self.assertIn("vol_curr", dataset.columns)
        self.assertIn("vwap_ratio_curr", dataset.columns)
        self.assertIn("accel_curr", dataset.columns)

        # Window 2 features (previous week)
        self.assertIn("ret_prev", dataset.columns)
        self.assertIn("range_prev", dataset.columns)
        self.assertIn("vol_prev", dataset.columns)

        # Comparative & differential features
        self.assertIn("ret_diff", dataset.columns)
        self.assertIn("vol_ratio", dataset.columns)
        self.assertIn("level_ratio_1w", dataset.columns)

        # News features
        self.assertIn("news_tone_diff", dataset.columns)
        self.assertIn("news_vol_ratio", dataset.columns)

    def test_live_action_generation(self):
        """Verify real-time action recommendation output structure and valid decisions."""
        action_data = generate_x_period_action(x_days=4.0)

        self.assertIn("action", action_data)
        self.assertTrue("BUY" in action_data["action"] or "SELL" in action_data["action"])
        self.assertIn("current_live_price", action_data)
        self.assertIn("expected_target_price", action_data)
        self.assertIn("stop_loss", action_data)
        self.assertIn("take_profit", action_data)
        self.assertIn("prob_up", action_data)
        self.assertIn("prob_down", action_data)

        # Up and Down probabilities should sum to approximately 100%
        self.assertAlmostEqual(action_data["prob_up"] + action_data["prob_down"], 100.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
