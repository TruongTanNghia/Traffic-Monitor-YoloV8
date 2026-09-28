"""Traffic events and speed-violation alerts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.utils.constants import EventType
from app.utils.logger import get_logger
from app.utils.time_utils import now_iso

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrafficEvent:
    """One event of a session (vehicle counted or speed violation)."""

    timestamp: str
    frame_index: int
    tracker_id: int
    vehicle_type: str
    speed_kmh: float | None
    speed_limit_kmh: float
    event_type: str
    video_time_s: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON/CSV friendly dictionary."""
        data = asdict(self)
        if self.speed_kmh is not None:
            data["speed_kmh"] = round(self.speed_kmh, 1)
        data["video_time_s"] = round(self.video_time_s, 3)
        return data

    @property
    def speed(self) -> float | None:
        """Alias required by the specification."""
        return self.speed_kmh

    @property
    def speed_limit(self) -> float:
        """Alias required by the specification."""
        return self.speed_limit_kmh


class AlertManager:
    """Create SPEED_VIOLATION alerts and keep the session event history."""

    def __init__(self, speed_limit_kmh: float = 80.0, active_window_s: float = 5.0) -> None:
        self.speed_limit_kmh = float(speed_limit_kmh)
        self._active_window_s = active_window_s
        self._alerts: list[TrafficEvent] = []
        self._events: list[TrafficEvent] = []
        self._violated_ids: set[int] = set()
        self._now_s = 0.0

    # ---------------------------------------------------------------- alerts
    def check_speed(
        self,
        tracker_id: int,
        vehicle_type: str,
        speed_kmh: float | None,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> TrafficEvent | None:
        """Create and register an alert when ``speed > speed_limit``.

        Each tracker id produces at most one SPEED_VIOLATION per session.
        """
        self._now_s = max(self._now_s, time_s)
        if speed_kmh is None or speed_kmh <= self.speed_limit_kmh:
            return None
        if tracker_id in self._violated_ids:
            return None
        alert = self.create_speed_alert(tracker_id, vehicle_type, speed_kmh, frame_index, time_s)
        self.register_alert(alert)
        return alert

    def create_speed_alert(
        self,
        tracker_id: int,
        vehicle_type: str,
        speed_kmh: float,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> TrafficEvent:
        """Build a SPEED_VIOLATION event."""
        return TrafficEvent(
            timestamp=now_iso(),
            frame_index=frame_index,
            tracker_id=int(tracker_id),
            vehicle_type=vehicle_type,
            speed_kmh=float(speed_kmh),
            speed_limit_kmh=self.speed_limit_kmh,
            event_type=EventType.SPEED_VIOLATION.value,
            video_time_s=time_s,
        )

    def register_alert(self, alert: TrafficEvent) -> None:
        """Store an alert in the active list and the history."""
        self._alerts.append(alert)
        self._violated_ids.add(alert.tracker_id)
        self.record_event(alert)
        logger.warning(
            "SPEED_VIOLATION tracker_id=%d %s %.1f km/h > %.0f",
            alert.tracker_id, alert.vehicle_type, alert.speed_kmh or 0.0, alert.speed_limit_kmh,
        )

    def get_active_alerts(self) -> list[TrafficEvent]:
        """Alerts raised within the last ``active_window_s`` seconds of video."""
        return [a for a in self._alerts if self._now_s - a.video_time_s <= self._active_window_s]

    def get_alert_history(self) -> list[TrafficEvent]:
        """All alerts of the session."""
        return list(self._alerts)

    def clear_alerts(self) -> None:
        """Forget alerts and events (new session)."""
        self._alerts.clear()
        self._events.clear()
        self._violated_ids.clear()
        self._now_s = 0.0

    @property
    def violated_ids(self) -> set[int]:
        """Tracker ids that violated the limit in this session."""
        return set(self._violated_ids)

    # ---------------------------------------------------------------- events
    def create_crossing_event(
        self,
        tracker_id: int,
        vehicle_type: str,
        direction: str,
        speed_kmh: float | None,
        frame_index: int,
        time_s: float,
    ) -> TrafficEvent:
        """Build and record a VEHICLE_IN / VEHICLE_OUT event."""
        event_type = EventType.VEHICLE_IN if direction == "in" else EventType.VEHICLE_OUT
        event = TrafficEvent(
            timestamp=now_iso(),
            frame_index=frame_index,
            tracker_id=int(tracker_id),
            vehicle_type=vehicle_type,
            speed_kmh=speed_kmh,
            speed_limit_kmh=self.speed_limit_kmh,
            event_type=event_type.value,
            video_time_s=time_s,
        )
        self.record_event(event)
        return event

    def record_event(self, event: TrafficEvent) -> None:
        """Append an event to the session history."""
        self._events.append(event)

    def get_event_history(self) -> list[TrafficEvent]:
        """All events of the session in chronological order."""
        return list(self._events)
