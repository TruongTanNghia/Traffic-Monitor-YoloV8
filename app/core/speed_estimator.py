"""Vehicle speed estimation with a perspective transform.

For each tracker id we keep a one-second deque of ``(time_s, x_m, y_m)``
road-plane positions (bottom-center anchor, transformed to metres).
Speed = magnitude of the least-squares velocity fitted over that window.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import supervision as sv

from app.utils.constants import KMH_PER_MPS
from app.utils.logger import get_logger
from app.vision.coordinates import bottom_center_anchors
from app.vision.geometry import occluded_bottom_mask, validate_quadrilateral
from app.vision.transformer import ViewTransformer, build_target

logger = get_logger(__name__)

Sample = tuple[float, float, float]


class SpeedEstimator:
    """Estimate km/h per tracker id."""

    STALE_TRACK_SECONDS = 3.0
    MIN_SAMPLES = 3
    MAX_PLAUSIBLE_KMH = 250.0

    def __init__(self, fps: float, min_track_seconds: float = 0.5) -> None:
        self._fps = fps if fps > 0 else 25.0
        self._min_track_seconds = min_track_seconds
        self._history_length = max(2, int(round(self._fps)))
        self._transformer: ViewTransformer | None = None
        self._history: dict[int, deque[Sample]] = {}
        self._speeds: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    def initialize_transformer(
        self, source: np.ndarray, road_width_m: float, road_length_m: float
    ) -> None:
        """Build the pixel→metre transformer.

        Raises:
            PerspectiveError: invalid source quadrilateral.
        """
        points = validate_quadrilateral(source)
        self._transformer = ViewTransformer(points, build_target(road_width_m, road_length_m))
        logger.info("Speed transformer ready: road %.1f m x %.1f m", road_width_m, road_length_m)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Pixel points (N×2) → road-plane metres (N×2)."""
        if self._transformer is None:
            raise RuntimeError("initialize_transformer() must be called first")
        return self._transformer.transform_points(points)

    def update_track_position(self, tracker_id: int, point_m: np.ndarray, time_s: float) -> None:
        """Append a metric position of ``tracker_id`` observed at ``time_s``."""
        history = self._history.setdefault(tracker_id, deque(maxlen=self._history_length))
        history.append((float(time_s), float(point_m[0]), float(point_m[1])))
        self._last_seen[tracker_id] = float(time_s)

    def calculate_speed(self, tracker_id: int) -> float | None:
        """Compute speed in km/h, or ``None`` when not enough data yet.

        No speed on the first sample; at least ``min_track_seconds`` of history
        is required.
        """
        history = self._history.get(tracker_id)
        if not history or len(history) < self.MIN_SAMPLES:
            return None
        samples = np.asarray(history, dtype=np.float64)
        times = samples[:, 0] - samples[0, 0]
        elapsed = times[-1]
        if elapsed < self._min_track_seconds or elapsed <= 0:
            return None
        # Least-squares velocity over the whole window: robust to single
        # anchor jumps (box resizing, partial occlusion) unlike end-point diff.
        velocity_x = np.polyfit(times, samples[:, 1], 1)[0]
        velocity_y = np.polyfit(times, samples[:, 2], 1)[0]
        speed = math.hypot(velocity_x, velocity_y) * KMH_PER_MPS
        if speed > self.MAX_PLAUSIBLE_KMH:
            logger.debug("Discarding implausible speed %.0f km/h for id %d", speed, tracker_id)
            return self._speeds.get(tracker_id)
        self._speeds[tracker_id] = speed
        return speed

    def update(
        self, detections: sv.Detections, time_s: float, in_zone: np.ndarray | None = None
    ) -> dict[int, float]:
        """Update all tracked detections and return current speeds.

        Args:
            detections: tracked detections of the frame.
            time_s: timestamp of the frame in seconds.
            in_zone: optional mask; only detections inside the measured road
                area contribute samples.
        """
        if len(detections) == 0 or detections.tracker_id is None:
            self._prune(time_s)
            return {}
        anchors = bottom_center_anchors(detections)
        mask = np.ones(len(detections), dtype=bool) if in_zone is None else in_zone.copy()
        occluded = occluded_bottom_mask(detections.xyxy)
        for tracker_id in detections.tracker_id[occluded]:
            # Anchor jumps while the vehicle emerges; restart its measurement window.
            self._history.pop(int(tracker_id), None)
        mask &= ~occluded
        if mask.any():
            metric = self.transform_points(anchors[mask])
            for tracker_id, point in zip(detections.tracker_id[mask], metric):
                self.update_track_position(int(tracker_id), point, time_s)
                self.calculate_speed(int(tracker_id))
        self._prune(time_s)
        return {
            int(tid): self._speeds[int(tid)]
            for tid in detections.tracker_id
            if int(tid) in self._speeds
        }

    def get_speed(self, tracker_id: int) -> float | None:
        """Latest computed speed of ``tracker_id``."""
        return self._speeds.get(tracker_id)

    def get_speed_violations(self, speed_limit_kmh: float) -> list[int]:
        """Tracker ids whose latest speed exceeds ``speed_limit_kmh``."""
        return [tid for tid, speed in self._speeds.items() if speed > speed_limit_kmh]

    def _prune(self, now_s: float) -> None:
        """Forget tracks not seen for a while (bounded memory)."""
        stale = [tid for tid, seen in self._last_seen.items() if now_s - seen > self.STALE_TRACK_SECONDS]
        for tid in stale:
            self._history.pop(tid, None)
            self._speeds.pop(tid, None)
            self._last_seen.pop(tid, None)

    def reset(self) -> None:
        """Clear all history and cached speeds."""
        self._history.clear()
        self._speeds.clear()
        self._last_seen.clear()
