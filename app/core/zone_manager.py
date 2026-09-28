"""Geometry of the counting line and of the speed measurement zone."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from app.utils.logger import get_logger
from app.vision.coordinates import bottom_center_anchors
from app.vision.geometry import (
    horizontal_line_y,
    points_in_polygon,
    scale_points,
    validate_quadrilateral,
)

logger = get_logger(__name__)


class ZoneManager:
    """Compute line/zone coordinates for a given frame size."""

    def __init__(
        self,
        frame_size: tuple[int, int],
        line_y_ratio: float,
        perspective_source: Sequence[Sequence[float]],
        reference_size: Sequence[float],
    ) -> None:
        """
        Raises:
            PerspectiveError: when the perspective points are invalid.
        """
        self._frame_size = frame_size
        self._line_y_ratio = line_y_ratio
        validated = validate_quadrilateral(perspective_source)
        if tuple(reference_size) != tuple(frame_size):
            logger.info("Scaling perspective points from %s to %s", tuple(reference_size), frame_size)
            validated = scale_points(validated, reference_size, frame_size)
        self._polygon = validated

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` of processed frames."""
        return self._frame_size

    @property
    def line_y(self) -> int:
        """Pixel row of the counting line."""
        return horizontal_line_y(self._frame_size[1], self._line_y_ratio)

    def line_points(self) -> tuple[sv.Point, sv.Point]:
        """Start/end of the counting line.

        Start is on the right so that Supervision's LineZone counts vehicles
        moving downwards (towards the camera) as IN.
        """
        width = self._frame_size[0]
        return sv.Point(width, self.line_y), sv.Point(0, self.line_y)

    @property
    def speed_polygon(self) -> np.ndarray:
        """Perspective quadrilateral (4×2, pixels) in frame coordinates."""
        return self._polygon.copy()

    def in_speed_zone(self, detections: sv.Detections) -> np.ndarray:
        """Mask of detections whose bottom-center lies inside the speed zone."""
        return points_in_polygon(bottom_center_anchors(detections), self._polygon)
