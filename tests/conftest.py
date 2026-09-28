"""Shared fixtures: make ``app`` importable and build synthetic detections."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light"}
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def make_detections(
    boxes: list[list[float]],
    class_ids: list[int],
    confidences: list[float] | None = None,
    tracker_ids: list[int] | None = None,
) -> sv.Detections:
    """Build ``sv.Detections`` by hand (no model involved)."""
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
        class_id=np.asarray(class_ids, dtype=int),
        confidence=None if confidences is None else np.asarray(confidences, dtype=float),
        tracker_id=None if tracker_ids is None else np.asarray(tracker_ids, dtype=int),
    )


def box_at(x_center: float, bottom: float, width: float = 20, height: float = 40) -> list[float]:
    """Box whose bottom-center anchor is ``(x_center, bottom)``."""
    return [x_center - width / 2, bottom - height, x_center + width / 2, bottom]


@pytest.fixture
def square_perspective() -> np.ndarray:
    """100×100 px square mapped to a 100×100 m road (1 px = 1 m)."""
    return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
