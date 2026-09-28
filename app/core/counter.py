"""IN/OUT counting with Supervision LineZone (no double counting per ID)."""

from __future__ import annotations

from dataclasses import dataclass, field

import supervision as sv

from app.utils.constants import VEHICLE_CLASSES
from app.utils.logger import get_logger
from app.vision.geometry import horizontal_line_y

logger = get_logger(__name__)


@dataclass
class CrossingResult:
    """Vehicles that crossed the line in the current frame: ``(tracker_id, type)``."""

    crossed_in: list[tuple[int, str]] = field(default_factory=list)
    crossed_out: list[tuple[int, str]] = field(default_factory=list)


class TrafficCounter:
    """Count vehicles crossing a horizontal line, per direction and class."""

    def __init__(self, class_names: dict[int, str]) -> None:
        self._class_names = dict(class_names)
        self._line_zone: sv.LineZone | None = None
        self._line_y = 0
        self._frame_width = 0
        self._counted_ids: set[int] = set()
        self._class_counts = self._empty_counts()

    @staticmethod
    def _empty_counts() -> dict[str, dict[str, int]]:
        return {name: {"in": 0, "out": 0} for name in VEHICLE_CLASSES}

    def initialize_line(self, frame_width: int, frame_height: int, line_y_ratio: float) -> None:
        """Create the LineZone at ``line_y_ratio`` of the frame height.

        Start is on the right, end on the left so that downward motion
        (towards the camera) is counted as IN.
        """
        self._frame_width = frame_width
        self._line_y = horizontal_line_y(frame_height, line_y_ratio)
        self._line_zone = sv.LineZone(
            start=sv.Point(frame_width, self._line_y),
            end=sv.Point(0, self._line_y),
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
        )

    def process(self, detections: sv.Detections) -> CrossingResult:
        """Update counts with tracked detections of the current frame."""
        result = CrossingResult()
        if self._line_zone is None or len(detections) == 0 or detections.tracker_id is None:
            return result
        crossed_in, crossed_out = self._line_zone.trigger(detections)
        for index, tracker_id in enumerate(detections.tracker_id):
            direction = "in" if crossed_in[index] else "out" if crossed_out[index] else None
            if direction is None or int(tracker_id) in self._counted_ids:
                continue
            vehicle_type = self._class_names.get(int(detections.class_id[index]), "unknown")
            self._counted_ids.add(int(tracker_id))
            self._class_counts.setdefault(vehicle_type, {"in": 0, "out": 0})[direction] += 1
            target = result.crossed_in if direction == "in" else result.crossed_out
            target.append((int(tracker_id), vehicle_type))
        return result

    def get_total_in(self) -> int:
        """Vehicles counted IN in this session."""
        return sum(c["in"] for c in self._class_counts.values())

    def get_total_out(self) -> int:
        """Vehicles counted OUT in this session."""
        return sum(c["out"] for c in self._class_counts.values())

    def get_class_counts(self) -> dict[str, dict[str, int]]:
        """``{"car": {"in": n, "out": m}, ...}`` for every vehicle class."""
        return {name: dict(counts) for name, counts in self._class_counts.items()}

    @property
    def line_y(self) -> int:
        """Pixel row of the counting line."""
        return self._line_y

    @property
    def line_endpoints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Left and right end of the line (for drawing)."""
        return (0, self._line_y), (self._frame_width, self._line_y)

    def reset(self) -> None:
        """Clear counts and recreate the LineZone."""
        self._counted_ids.clear()
        self._class_counts = self._empty_counts()
        if self._line_zone is not None:
            self._line_zone = sv.LineZone(
                start=self._line_zone.vector.start,
                end=self._line_zone.vector.end,
                triggering_anchors=[sv.Position.BOTTOM_CENTER],
            )
