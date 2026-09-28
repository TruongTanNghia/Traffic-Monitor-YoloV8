"""Multi-object tracking with Supervision ByteTrack + DetectionsSmoother."""

from __future__ import annotations

import numpy as np
import supervision as sv

from app.utils.logger import get_logger

logger = get_logger(__name__)


class VehicleTracker:
    """Assign a stable ``tracker_id`` to each vehicle across frames."""

    SMOOTHING_LENGTH = 5

    def __init__(self, frame_rate: float = 30.0, track_activation_threshold: float = 0.25) -> None:
        self._frame_rate = frame_rate
        self._activation_threshold = track_activation_threshold
        self._tracker: sv.ByteTrack | None = None
        self._smoother: sv.DetectionsSmoother | None = None
        self._active_ids: list[int] = []
        self.initialize(frame_rate)

    def initialize(self, frame_rate: float | None = None) -> None:
        """Create a fresh ByteTrack and smoother.

        Args:
            frame_rate: FPS of the source; controls how long lost tracks are kept.
        """
        if frame_rate is not None:
            self._frame_rate = frame_rate
        rate = max(1, int(round(self._frame_rate)))
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self._activation_threshold,
            lost_track_buffer=rate,
            frame_rate=rate,
        )
        self._smoother = sv.DetectionsSmoother(length=self.SMOOTHING_LENGTH)
        self._active_ids = []
        logger.debug("Tracker initialised (frame_rate=%d)", rate)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Associate detections with tracks; returned detections have ``tracker_id``."""
        tracked = self._tracker.update_with_detections(detections)  # type: ignore[union-attr]
        if tracked.tracker_id is None:
            tracked.tracker_id = np.empty(0, dtype=int)
        self._active_ids = [int(tid) for tid in tracked.tracker_id]
        return tracked

    def smooth(self, detections: sv.Detections) -> sv.Detections:
        """Reduce bounding-box jitter using the recent history of each track."""
        if len(detections) == 0:
            return detections
        smoothed = self._smoother.update_with_detections(detections)  # type: ignore[union-attr]
        return smoothed if len(smoothed) == len(detections) else detections

    def get_track_ids(self) -> list[int]:
        """Tracker ids present in the latest frame."""
        return list(self._active_ids)

    def reset(self) -> None:
        """Forget all tracks (new session)."""
        self.initialize()
