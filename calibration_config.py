"""Load and save the shared pixel-to-meter calibration."""

import json
import math
from pathlib import Path
from typing import Union


DEFAULT_PIXELS_PER_METER = 200.0
CALIBRATION_FILE = Path(__file__).with_name("calibration.json")
PathLike = Union[str, Path]


def validate_pixels_per_meter(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("pixels_per_meter must be a positive finite number")

    pixels_per_meter = float(value)
    if not math.isfinite(pixels_per_meter) or pixels_per_meter <= 0:
        raise ValueError("pixels_per_meter must be a positive finite number")
    return pixels_per_meter


def load_pixels_per_meter(path: PathLike = CALIBRATION_FILE) -> float:
    calibration_path = Path(path)
    try:
        with calibration_path.open("r", encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
    except FileNotFoundError:
        return DEFAULT_PIXELS_PER_METER
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid calibration JSON in {calibration_path}") from error

    if not isinstance(calibration, dict) or "pixels_per_meter" not in calibration:
        raise ValueError(
            f"Calibration file {calibration_path} must contain 'pixels_per_meter'"
        )
    return validate_pixels_per_meter(calibration["pixels_per_meter"])


def save_pixels_per_meter(
    value: object, path: PathLike = CALIBRATION_FILE
) -> float:
    pixels_per_meter = validate_pixels_per_meter(value)
    calibration_path = Path(path)
    with calibration_path.open("w", encoding="utf-8") as calibration_file:
        json.dump(
            {"pixels_per_meter": pixels_per_meter},
            calibration_file,
            indent=2,
        )
        calibration_file.write("\n")
    return pixels_per_meter
