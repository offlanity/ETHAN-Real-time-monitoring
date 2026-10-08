"""
utils.py — Drawing utilities, Kinematics HUD, and FPS counter.
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import Tuple, List, Dict


# ── Color palette (BGR) ───────────────────────────────────────────────────────
GREEN   = (0, 220, 0)
RED     = (0, 0, 230)
ORANGE  = (0, 140, 255)
CYAN    = (255, 220, 0)
WHITE   = (255, 255, 255)
GRAY    = (190, 190, 190)
DARK_BG = (25, 25, 25)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
FONT_THICK  = 2


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0.0 else 0.0


def draw_kinematics_face(frame: np.ndarray, item: Dict) -> None:
    """
    Renders bounding box, motion breadcrumb trails, and physics metrics:
    Speed (v), Net Displacement (Δd), and Total Distance (s).
    """
    x1, y1, x2, y2 = item["bbox"]
    name = item["name"]
    score = item["score"]
    recognized = item["recognized"]
    speed = item["speed"]
    disp = item["displacement"]
    dist = item["total_distance"]
    status = item["status"]
    history = item["history"]
    appearance_cues = item.get("appearance_cues", [])

    # Visual priority: Panic/Running alert > Unrecognized > Authorized
    if status == "RUNNING (ALERT!)":
        box_color = RED
    elif not recognized:
        box_color = ORANGE
    else:
        box_color = GREEN

    # 1. Motion Trail (cyan line showing student's path)
    if len(history) > 1:
        for i in range(1, len(history)):
            pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
            pt2 = (int(history[i][0]), int(history[i][1]))
            cv2.line(frame, pt1, pt2, CYAN, 2, cv2.LINE_AA)

    # 2. Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

    # 3. Physics Telemetry HUD above the head
    line1 = f"{name} [{score:.2f}]" if recognized else name
    line2 = f"v: {speed:.2f} m/s | {status}"
    line3 = f"Disp (Δd): {disp:.2f}m | Dist: {dist:.2f}m"

    hud_y = max(y1 - 50, 55)
    # Background card for readability
    hud_bottom = hud_y + (62 if appearance_cues else 44)
    cv2.rectangle(frame, (x1 - 2, hud_y - 14), (x1 + 300, hud_bottom), DARK_BG, cv2.FILLED)
    cv2.rectangle(frame, (x1 - 2, hud_y - 14), (x1 + 300, hud_bottom), box_color, 1)

    cv2.putText(frame, line1, (x1 + 4, hud_y), FONT, 0.50, box_color, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (x1 + 4, hud_y + 18), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, line3, (x1 + 4, hud_y + 36), FONT, 0.40, GRAY, 1, cv2.LINE_AA)
    if appearance_cues:
        cue_labels = {
            "possible_dark_clothing": "possible dark clothing",
            "possible_face_covering": "possible face covering",
        }
        cue_text = "Possible cue: " + ", ".join(
            cue_labels[cue] for cue in appearance_cues if cue in cue_labels
        )
        cv2.putText(frame, cue_text, (x1 + 4, hud_y + 55), FONT, 0.38, ORANGE, 1, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (14, 30), FONT, 0.75, CYAN, 2, cv2.LINE_AA)


def draw_add_prompt(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    text = "Unregistered student — press ENTER to save"
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.65, 2)
    x, y = (w - tw) // 2, 50

    cv2.rectangle(frame, (x - 12, y - th - 10), (x + tw + 12, y + baseline + 6), ORANGE, cv2.FILLED)
    cv2.putText(frame, text, (x, y), FONT, 0.65, WHITE, 2, cv2.LINE_AA)


def draw_name_input(frame: np.ndarray, name_buffer: str) -> None:
    h, w = frame.shape[:2]
    box_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - box_h // 2), (w, h // 2 + box_h // 2), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    prompt = "Type student name, ENTER to confirm, ESC to cancel:"
    cv2.putText(frame, prompt, (40, h // 2 - 16), FONT, 0.65, GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, name_buffer + "|", (40, h // 2 + 28), FONT, 0.9, CYAN, 2, cv2.LINE_AA)


def draw_info_bar(frame: np.ndarray, num_registered: int) -> None:
    h = frame.shape[0]
    cv2.putText(
        frame,
        f"CLAIVE Engine | Registered: {num_registered} | ENTER: add face | Q: quit",
        (12, h - 14), FONT, 0.48, GRAY, 1, cv2.LINE_AA,
    )