import unittest

import numpy as np

from appearance_cues import (
    POSSIBLE_DARK_CLOTHING,
    POSSIBLE_FACE_COVERING,
    detect_appearance_cues,
)
from dashboard import DashboardState
from kinematics import KinematicsTracker


class AppearanceCueTests(unittest.TestCase):
    def test_dark_region_below_face_is_flagged_as_possible_clothing(self):
        frame = np.full((120, 100, 3), 220, dtype=np.uint8)
        frame[44:100, 35:65] = 15

        cues = detect_appearance_cues(frame, (40, 20, 60, 40))

        self.assertEqual(cues, [POSSIBLE_DARK_CLOTHING])

    def test_uniform_distinct_lower_face_region_is_flagged_as_possible_covering(self):
        frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        rng = np.random.default_rng(7)
        frame[10:70, 20:80] = rng.integers(80, 180, size=(60, 60, 3), dtype=np.uint8)
        frame[44:65, 29:71] = (20, 20, 220)

        cues = detect_appearance_cues(frame, (20, 10, 80, 70))

        self.assertIn(POSSIBLE_FACE_COVERING, cues)

    def test_bright_uniform_regions_do_not_trigger_cues(self):
        frame = np.full((120, 100, 3), 220, dtype=np.uint8)

        self.assertEqual(detect_appearance_cues(frame, (40, 20, 60, 40)), [])

    def test_invalid_face_box_returns_no_cues(self):
        frame = np.full((100, 100, 3), 0, dtype=np.uint8)

        self.assertEqual(detect_appearance_cues(frame, (30, 30, 20, 20)), [])

    def test_kinematics_preserves_appearance_cues(self):
        tracker = KinematicsTracker(pixels_per_meter=100)
        detection = {
            "bbox": (10, 10, 30, 30),
            "name": "Alex",
            "score": 0.9,
            "recognized": True,
            "appearance_cues": [POSSIBLE_DARK_CLOTHING],
        }

        result = tracker.update([detection])[0]

        self.assertEqual(result["appearance_cues"], [POSSIBLE_DARK_CLOTHING])

    def test_dashboard_records_neutral_appearance_event(self):
        state = DashboardState("unused")
        detection = {
            "track_id": 1,
            "name": "Alex",
            "recognized": True,
            "score": 0.9,
            "speed": 0.0,
            "status": "Stationary",
            "appearance_cues": [POSSIBLE_DARK_CLOTHING],
        }

        state.update([detection], ["Alex"], 30.0)
        snapshot = state.snapshot()

        self.assertEqual(snapshot["active_alert_count"], 1)
        self.assertEqual(snapshot["detections"][0]["alert"], "appearance")
        self.assertEqual(snapshot["detections"][0]["appearance_cues"], [POSSIBLE_DARK_CLOTHING])
        self.assertEqual(snapshot["events"][0]["title"], "Possible dark clothing (visual cue)")


if __name__ == "__main__":
    unittest.main()
