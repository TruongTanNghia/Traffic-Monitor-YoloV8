"""Coordinate helpers for detections."""

from __future__ import annotations

import numpy as np
import supervision as sv

SPEED_ANCHOR = sv.Position.BOTTOM_CENTER


def bottom_center_anchors(detections: sv.Detections) -> np.ndarray:
    """Bottom-center points (N×2) — approximate road contact points."""
    if len(detections) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return detections.get_anchors_coordinates(anchor=SPEED_ANCHOR).astype(np.float32)


def ratio_to_pixel(ratio: float, length: int) -> int:
    """Convert a 0..1 ratio into a pixel coordinate."""
    return int(round(ratio * length))
