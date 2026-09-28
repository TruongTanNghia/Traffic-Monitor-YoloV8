"""Perspective (homography) transform from image pixels to road metres."""

from __future__ import annotations

import cv2
import numpy as np


def build_target(road_width_m: float, road_length_m: float) -> np.ndarray:
    """Rectangle in metres matching the order of the source quadrilateral:
    top-left, top-right, bottom-right, bottom-left."""
    width, length = float(road_width_m), float(road_length_m)
    return np.array([[0, 0], [width, 0], [width, length], [0, length]], dtype=np.float32)


class ViewTransformer:
    """Map pixel coordinates onto a metric bird's-eye plane."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self._matrix = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform N×2 pixel points into N×2 metric points."""
        if points.size == 0:
            return np.asarray(points, dtype=np.float32).reshape(-1, 2)
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(reshaped, self._matrix).reshape(-1, 2)
