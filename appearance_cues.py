"""Lightweight, uncertain visual-cue heuristics for detected faces."""

import cv2
import numpy as np


POSSIBLE_DARK_CLOTHING = "possible_dark_clothing"
POSSIBLE_FACE_COVERING = "possible_face_covering"


def detect_appearance_cues(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> list[str]:
    """Estimate dark clothing and lower-face obstruction from nearby pixels.

    The body and face regions are approximated from the face bounding box.
    These simple pixel heuristics are not classifiers and can produce false
    positives or miss cues; they do not indicate a person's intent.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    face_width, face_height = x2 - x1, y2 - y1
    if face_width <= 0 or face_height <= 0:
        return []

    cues = []
    center_x = (x1 + x2) // 2
    clothing_x1 = max(center_x - face_width, 0)
    clothing_x2 = min(center_x + face_width, frame_width)
    clothing_y1 = max(y2 + int(face_height * 0.2), 0)
    clothing_y2 = min(y2 + int(face_height * 2.5), frame_height)
    clothing_region = frame[clothing_y1:clothing_y2, clothing_x1:clothing_x2]
    if clothing_region.size >= 300:
        value_channel = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2HSV)[:, :, 2]
        if float(np.mean(value_channel < 65)) >= 0.65:
            cues.append(POSSIBLE_DARK_CLOTHING)

    face_x1 = max(x1, 0)
    face_y1 = max(y1, 0)
    face_x2 = min(x2, frame_width)
    face_y2 = min(y2, frame_height)
    face_region = frame[face_y1:face_y2, face_x1:face_x2]
    if face_region.shape[0] >= 24 and face_region.shape[1] >= 24:
        region_height, region_width = face_region.shape[:2]
        ref = face_region[
            int(region_height * 0.30):int(region_height * 0.52),
            int(region_width * 0.15):int(region_width * 0.85),
        ]
        lower = face_region[
            int(region_height * 0.58):int(region_height * 0.92),
            int(region_width * 0.15):int(region_width * 0.85),
        ]
        if ref.size and lower.size:
            color_delta = float(
                np.linalg.norm(
                    np.mean(ref, axis=(0, 1)).astype(np.float32)
                    - np.mean(lower, axis=(0, 1)).astype(np.float32)
                )
            )
            lower_texture = float(
                np.std(cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY))
            )
            ref_texture = float(np.std(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)))
            if color_delta >= 45.0 and lower_texture <= 20.0 and ref_texture >= 10.0:
                cues.append(POSSIBLE_FACE_COVERING)

    return cues
