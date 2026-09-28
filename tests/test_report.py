"""Unit tests for ReportGenerator (JSON + CSV)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from app.config.settings import AppConfig
from app.reports.csv_exporter import CSV_COLUMNS
from app.reports.report_generator import ReportGenerator

SESSION = {"session_id": "20260928_123055", "started_at": "2026-09-28T12:30:55",
           "ended_at": "2026-09-28T12:34:00", "duration_s": 185.0, "frames_processed": 5526}
EVENTS = [
    {"timestamp": "2026-09-28T12:31:02", "frame_index": 1843, "video_time_s": 73.72, "tracker_id": 21,
     "vehicle_type": "car", "speed_kmh": 92.4, "speed_limit_kmh": 80.0, "event_type": "SPEED_VIOLATION"},
    {"timestamp": "2026-09-28T12:31:03", "frame_index": 1850, "video_time_s": 74.0, "tracker_id": 3,
     "vehicle_type": "truck", "speed_kmh": None, "speed_limit_kmh": 80.0, "event_type": "VEHICLE_IN"},
]
STATS = {
    "total_in": 1, "total_out": 0, "total_vehicles": 1,
    "class_counts": {"car": {"in": 0, "out": 0}, "motorcycle": {"in": 0, "out": 0},
                     "bus": {"in": 0, "out": 0}, "truck": {"in": 1, "out": 0}},
    "maximum_speed": {"speed_kmh": 92.4, "tracker_id": 21, "vehicle_type": "car"},
}


def build_report() -> dict:
    return ReportGenerator().generate(SESSION, {"type": "file", "uri": "x.mp4"}, AppConfig().to_dict(), STATS, EVENTS)


def test_report_contains_required_sections() -> None:
    report = build_report()
    for key in ("session", "source", "total_in", "total_out", "vehicle_counts",
                "speed_violations", "maximum_speed", "events"):
        assert key in report
    assert report["vehicle_counts"]["truck"] == {"in": 1, "out": 0, "total": 1}
    assert report["speed_violations"]["count"] == 1
    assert report["maximum_speed"]["speed_kmh"] == 92.4
    assert report["session"]["app_version"]


def test_save_json_roundtrip(tmp_path: Path) -> None:
    path = ReportGenerator().save_json(build_report(), tmp_path)
    assert path.parent == tmp_path / "reports"
    assert re.fullmatch(r"traffic_report_\d{8}_\d{6}\.json", path.name)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["total_in"] == 1 and len(data["events"]) == 2


def test_save_csv_header_and_rows(tmp_path: Path) -> None:
    path = ReportGenerator().save_csv(EVENTS, tmp_path, SESSION["session_id"])
    assert path.name == "traffic_report_20260928_123055.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(CSV_COLUMNS)
    assert len(rows) == 2
    assert rows[0]["event_type"] == "SPEED_VIOLATION" and rows[1]["speed_kmh"] == ""


def test_empty_session_report(tmp_path: Path) -> None:
    report = ReportGenerator().generate(SESSION, {}, {}, {}, [])
    assert report["total_vehicles"] == 0 and report["events"] == []
    assert ReportGenerator().save_csv([], tmp_path, "20260101_000000").read_text(encoding="utf-8-sig").strip() \
        == ",".join(CSV_COLUMNS)
