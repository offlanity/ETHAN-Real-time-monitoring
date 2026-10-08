"""
kinematics.py — Kinematics and motion physics tracking module.

Calculates:
  - Displacement (Δd) from origin in meters
  - Instantaneous/Windowed Speed (v = Δd / Δt) in m/s
  - Cumulative path distance (s) in meters
  - Running / Rapid movement alerts (threshold detection)

Pixel-to-meter conversion is a single calibrated scale and is only approximate
when subjects move across different depths or perspective-distorted planes.
The position anchor is the bottom-center of the face detection box, not a
ground-plane foot position.
"""

import math
import time
import numpy as np
from collections import deque
from typing import List, Dict, Optional

from calibration_config import (
    DEFAULT_PIXELS_PER_METER,
    load_pixels_per_meter,
    validate_pixels_per_meter,
)


# ── Physics Constants ─────────────────────────────────────────────────────────
# Calibrate: pixels per 1 real-world meter.
# For a typical phone camera / DroidCam at 1.5 - 2m distance, ~200 px ≈ 1 meter.
PIXELS_PER_METER = DEFAULT_PIXELS_PER_METER

# Walking speed is ~1.0–1.4 m/s. Anything above 2.2 m/s triggers an alert.
RUNNING_SPEED_THRESHOLD = 2.2  

# Time window (seconds) to compute velocity, smoothing out bounding box jitter.
VELOCITY_WINDOW_SEC = 0.5     


class KinematicsTracker:
    def __init__(
        self,
        max_disappeared: int = 10,
        pixels_per_meter: Optional[float] = None,
    ) -> None:
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.pixels_per_meter = (
            load_pixels_per_meter()
            if pixels_per_meter is None
            else validate_pixels_per_meter(pixels_per_meter)
        )

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Takes raw detections from FaceRecognizer, associates persistent IDs,
        and enriches each detection with physics telemetry.
        """
        now = time.perf_counter()
        input_data = []

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            # Bottom-center anchor point (chin/neck level)
            cx = int((x1 + x2) / 2.0)
            cy = int(y2)
            input_data.append((cx, cy, d))

        # First frame registration
        if len(self.tracks) == 0:
            for cx, cy, d in input_data:
                self._register(cx, cy, d, now)
            return self._build_results(now)

        track_ids = list(self.tracks.keys())
        track_positions = [self.tracks[tid]["pos"] for tid in track_ids]

        if len(input_data) > 0:
            # Pair detections to tracks via Euclidean distance
            D = np.linalg.norm(
                np.array(track_positions)[:, np.newaxis] - np.array([c[:2] for c in input_data]),
                axis=2,
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols or D[row][col] > 140:
                    continue

                tid = track_ids[row]
                cx, cy, d = input_data[col]

                # Accumulate step distance: Δs = √(Δx² + Δy²) / PPM
                prev_x, prev_y = self.tracks[tid]["pos"]
                step_px = math.hypot(cx - prev_x, cy - prev_y)
                self.tracks[tid]["total_distance"] += (
                    step_px / self.pixels_per_meter
                )

                self.tracks[tid]["pos"] = (cx, cy)
                self.tracks[tid]["detection"] = d
                self.tracks[tid]["history"].append((cx, cy, now))
                self.disappeared[tid] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Register brand new targets
            for col in range(len(input_data)):
                if col not in used_cols:
                    cx, cy, d = input_data[col]
                    self._register(cx, cy, d, now)

            # Flag lost targets
            for row in range(len(track_ids)):
                if row not in used_rows:
                    tid = track_ids[row]
                    self.disappeared[tid] += 1
                    if self.disappeared[tid] > self.max_disappeared:
                        self._deregister(tid)
        else:
            for tid in list(self.tracks.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        return self._build_results(now)

    def _register(self, cx: int, cy: int, detection: Dict, timestamp: float) -> None:
        self.tracks[self.next_id] = {
            "pos": (cx, cy),
            "start_pos": (cx, cy),
            "detection": detection,
            "total_distance": 0.0,
            "history": deque([(cx, cy, timestamp)], maxlen=45),
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, tid: int) -> None:
        del self.tracks[tid]
        del self.disappeared[tid]

    def _build_results(self, now: float) -> List[Dict]:
        results = []
        for tid, tdata in self.tracks.items():
            if self.disappeared[tid] > 0:
                continue

            hist = tdata["history"]
            d = tdata["detection"]

            # ── PHYSICS: VELOCITY CALCULATION (v = Δd / Δt) ─────────────
            # Search backwards for a timestamp window of ~VELOCITY_WINDOW_SEC
            idx = 0
            for i, (_, _, t) in enumerate(hist):
                if now - t <= VELOCITY_WINDOW_SEC:
                    idx = i
                    break

            old_x, old_y, old_t = hist[idx]
            cur_x, cur_y, cur_t = hist[-1]
            dt = cur_t - old_t

            if dt > 0.05:
                dx_m = (cur_x - old_x) / self.pixels_per_meter
                dy_m = (cur_y - old_y) / self.pixels_per_meter
                speed = math.hypot(dx_m, dy_m) / dt
            else:
                speed = 0.0

            # Suppress micro-jitter (staying seated/breathing)
            if speed < 0.20:
                speed = 0.0

            # ── PHYSICS: NET DISPLACEMENT (Δd = ||r_final - r_initial||) ─
            start_x, start_y = tdata["start_pos"]
            disp_px = math.hypot(cur_x - start_x, cur_y - start_y)
            displacement = disp_px / self.pixels_per_meter

            # ── BEHAVIORAL STATUS ─────────────────────────────────────────
            if speed >= RUNNING_SPEED_THRESHOLD:
                status = "RUNNING (ALERT!)"
            elif speed >= 0.35:
                status = "Walking"
            else:
                status = "Stationary"

            results.append({
                "track_id": tid,
                "bbox": d["bbox"],
                "name": d["name"],
                "score": d["score"],
                "recognized": d["recognized"],
                "appearance_cues": d.get("appearance_cues", []),
                "speed": speed,
                "displacement": displacement,
                "total_distance": tdata["total_distance"],
                "status": status,
                "history": list(hist),
            })

        return results