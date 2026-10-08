import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calibration_config import (
    DEFAULT_PIXELS_PER_METER,
    load_pixels_per_meter,
    save_pixels_per_meter,
)
from kinematics import KinematicsTracker


def make_detection(x1, y1, x2, y2):
    return {
        "bbox": (x1, y1, x2, y2),
        "name": "person",
        "score": 1.0,
        "recognized": True,
    }


class CalibrationConfigTests(unittest.TestCase):
    def test_missing_file_uses_legacy_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.json"
            self.assertEqual(load_pixels_per_meter(path), DEFAULT_PIXELS_PER_METER)

    def test_saved_calibration_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            save_pixels_per_meter(237.5, path)
            self.assertEqual(load_pixels_per_meter(path), 237.5)

    def test_invalid_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            path.write_text(json.dumps({"pixels_per_meter": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive finite"):
                load_pixels_per_meter(path)


class KinematicsDisplacementTests(unittest.TestCase):
    def test_net_displacement_differs_from_cumulative_path_distance(self):
        tracker = KinematicsTracker(pixels_per_meter=100)
        tracker.update([make_detection(0, 0, 10, 10)])
        tracker.update([make_detection(25, 0, 35, 10)])
        result = tracker.update([make_detection(25, 40, 35, 50)])[0]

        self.assertAlmostEqual(result["displacement"], math.hypot(25, 40) / 100)
        self.assertAlmostEqual(result["total_distance"], (25 + 40) / 100)

    def test_tracker_loads_saved_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            save_pixels_per_meter(100, path)
            with patch(
                "kinematics.load_pixels_per_meter",
                side_effect=lambda: load_pixels_per_meter(path),
            ):
                tracker = KinematicsTracker()

        tracker.update([make_detection(0, 0, 10, 10)])
        result = tracker.update([make_detection(20, 0, 30, 10)])[0]
        self.assertAlmostEqual(result["displacement"], 0.2)


if __name__ == "__main__":
    unittest.main()
