"""Build and save session reports (JSON summary + CSV event list)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.reports.csv_exporter import CsvExporter
from app.reports.json_exporter import JsonExporter
from app.utils.constants import APP_VERSION, REPORTS_SUBDIR, EventType
from app.utils.file_utils import timestamped_filename

REPORT_PREFIX = "traffic_report"
CONFIG_KEYS_IN_REPORT = (
    "model_path", "confidence_threshold", "speed_limit_kmh", "line_y_ratio",
    "road_width_m", "road_length_m", "perspective_source", "perspective_reference_size",
    "min_track_seconds", "device",
)


class ReportGenerator:
    """Create the report dictionary from real session data and export it."""

    def __init__(
        self,
        json_exporter: JsonExporter | None = None,
        csv_exporter: CsvExporter | None = None,
        app_version: str = APP_VERSION,
    ) -> None:
        self._json = json_exporter or JsonExporter()
        self._csv = csv_exporter or CsvExporter()
        self._app_version = app_version

    def generate(
        self,
        session: dict[str, Any],
        source: dict[str, Any],
        config: dict[str, Any],
        statistics: dict[str, Any],
        events: list[dict[str, Any]],
        outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the report.

        Args:
            session: ``session_id``, ``started_at``, ``ended_at``, ``duration_s``,
                ``frames_processed``, ``device``.
            source: source metadata (``SourceInfo.to_dict()``).
            config: ``AppConfig.to_dict()``.
            statistics: last ``FrameProcessor.statistics()`` snapshot.
            events: event dictionaries in chronological order.
            outputs: paths of produced files.
        """
        violations = [e for e in events if e.get("event_type") == EventType.SPEED_VIOLATION.value]
        class_counts = statistics.get("class_counts", {})
        return {
            "session": {**session, "app_version": self._app_version},
            "source": source,
            "config": {key: config.get(key) for key in CONFIG_KEYS_IN_REPORT},
            "total_in": int(statistics.get("total_in", 0)),
            "total_out": int(statistics.get("total_out", 0)),
            "total_vehicles": int(statistics.get("total_vehicles", 0)),
            "vehicle_counts": {
                name: {"in": c.get("in", 0), "out": c.get("out", 0), "total": c.get("in", 0) + c.get("out", 0)}
                for name, c in class_counts.items()
            },
            "speed_violations": {"count": len(violations), "items": violations},
            "maximum_speed": statistics.get("maximum_speed"),
            "events": events,
            "outputs": outputs or {},
        }

    def save_json(self, report: dict[str, Any], output_dir: Path) -> Path:
        """Write ``traffic_report_<session_id>.json`` into ``output_dir/reports``."""
        session_id = report["session"]["session_id"]
        return self._json.export(report, self._reports_dir(output_dir) / timestamped_filename(
            REPORT_PREFIX, session_id, "json"))

    def save_csv(self, events: list[dict[str, Any]], output_dir: Path, session_id: str) -> Path:
        """Write ``traffic_report_<session_id>.csv`` (event list) into ``output_dir/reports``."""
        return self._csv.export(events, self._reports_dir(output_dir) / timestamped_filename(
            REPORT_PREFIX, session_id, "csv"))

    @staticmethod
    def _reports_dir(output_dir: Path) -> Path:
        return Path(output_dir) / REPORTS_SUBDIR
