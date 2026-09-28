"""Run the full pipeline on one frame.

detect → filter → track → count → speed → alert → annotate → dashboard (statistics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import supervision as sv

from app.core.alert_manager import TrafficEvent
from app.core.pipeline import PipelineComponents
from app.vision.annotations import OverlayStats, build_label


@dataclass
class MaxSpeed:
    """Highest speed measured in the session."""

    speed_kmh: float = 0.0
    tracker_id: int | None = None
    vehicle_type: str | None = None

    def to_dict(self) -> dict[str, Any] | None:
        """Dictionary or ``None`` when nothing was measured."""
        if self.tracker_id is None:
            return None
        return {"speed_kmh": round(self.speed_kmh, 1), "tracker_id": self.tracker_id,
                "vehicle_type": self.vehicle_type}


@dataclass(frozen=True)
class FrameResult:
    """Output of :meth:`FrameProcessor.process_frame`."""

    annotated_frame: np.ndarray
    detections: sv.Detections
    statistics: dict[str, Any]
    new_events: list[TrafficEvent] = field(default_factory=list)
    frame_index: int = 0


class FrameProcessor:
    """Stateful per-frame processor of one session."""

    def __init__(self, components: PipelineComponents) -> None:
        self._c = components
        self._max_speed = MaxSpeed()
        self._frame_index = -1
        self._processing_fps = 0.0

    @property
    def components(self) -> PipelineComponents:
        """Underlying pipeline components."""
        return self._c

    def set_processing_fps(self, fps: float) -> None:
        """FPS measured by the caller, shown in the overlay."""
        self._processing_fps = fps

    def process_frame(
        self, frame: np.ndarray, frame_index: int | None = None, timestamp_s: float | None = None
    ) -> FrameResult:
        """Process one BGR frame.

        Raises:
            InferenceError: if the detector fails.
        """
        self._frame_index = self._frame_index + 1 if frame_index is None else frame_index
        time_s = timestamp_s if timestamp_s is not None else self._frame_index / 25.0
        c = self._c

        detections = c.detector.filter_detections(c.detector.detect(frame))
        detections = c.tracker.smooth(c.tracker.update(detections))
        crossings = c.counter.process(detections)
        speeds = c.speed_estimator.update(detections, time_s, c.zone_manager.in_speed_zone(detections))

        events = self._crossing_events(crossings, speeds, time_s)
        events += self._speed_alerts(detections, speeds, time_s)
        violation_mask, labels = self._labels(detections, speeds)

        annotated = c.renderer.render(
            frame, detections, labels, violation_mask, c.counter.line_endpoints,
            c.zone_manager.speed_polygon, self._overlay_stats(),
        )
        return FrameResult(annotated, detections, self.statistics(len(detections)), events, self._frame_index)

    # ----------------------------------------------------------------- steps
    def _crossing_events(self, crossings: Any, speeds: dict[int, float], time_s: float) -> list[TrafficEvent]:
        alert_manager = self._c.alert_manager
        events = []
        for direction, items in (("in", crossings.crossed_in), ("out", crossings.crossed_out)):
            for tracker_id, vehicle_type in items:
                events.append(alert_manager.create_crossing_event(
                    tracker_id, vehicle_type, direction, speeds.get(tracker_id), self._frame_index, time_s))
        return events

    def _speed_alerts(self, detections: sv.Detections, speeds: dict[int, float], time_s: float) -> list[TrafficEvent]:
        alerts = []
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            tracker_id = int(tracker_id)
            speed = speeds.get(tracker_id)
            if speed is None:
                continue
            vehicle_type = self._c.detector.class_name(int(class_id))
            if speed > self._max_speed.speed_kmh:
                self._max_speed = MaxSpeed(speed, tracker_id, vehicle_type)
            alert = self._c.alert_manager.check_speed(tracker_id, vehicle_type, speed, self._frame_index, time_s)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _labels(self, detections: sv.Detections, speeds: dict[int, float]) -> tuple[np.ndarray, list[str]]:
        violated_ids = self._c.alert_manager.violated_ids
        mask = np.array([int(t) in violated_ids for t in detections.tracker_id], dtype=bool)
        confidences = detections.confidence if detections.confidence is not None else [None] * len(detections)
        labels = [
            build_label(int(tid), self._c.detector.class_name(int(cid)),
                        None if conf is None else float(conf), speeds.get(int(tid)), bool(violated))
            for tid, cid, conf, violated in zip(detections.tracker_id, detections.class_id, confidences, mask)
        ]
        return mask, labels

    def _overlay_stats(self) -> OverlayStats:
        counter = self._c.counter
        return OverlayStats(counter.get_total_in(), counter.get_total_out(),
                            len(self._c.alert_manager.get_alert_history()), self._processing_fps)

    # ------------------------------------------------------------- dashboard
    def statistics(self, active_tracks: int = 0) -> dict[str, Any]:
        """Snapshot of the session statistics (emitted to the dashboard)."""
        counter = self._c.counter
        total_in, total_out = counter.get_total_in(), counter.get_total_out()
        class_counts = counter.get_class_counts()
        return {
            "frame_index": self._frame_index,
            "total_in": total_in,
            "total_out": total_out,
            "total_vehicles": total_in + total_out,
            "class_counts": class_counts,
            "class_totals": {name: c["in"] + c["out"] for name, c in class_counts.items()},
            "speed_violations": len(self._c.alert_manager.get_alert_history()),
            "maximum_speed": self._max_speed.to_dict(),
            "active_tracks": active_tracks,
            "fps": round(self._processing_fps, 1),
        }
