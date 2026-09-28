"""Unit tests for TrafficCounter (LineZone, IN/OUT, no double counting)."""

from __future__ import annotations

import supervision as sv

from app.core.counter import TrafficCounter
from tests.conftest import VEHICLE_IDS, box_at, make_detections

LINE_Y = 500


def counter() -> TrafficCounter:
    result = TrafficCounter(VEHICLE_IDS)
    result.initialize_line(1000, 1000, LINE_Y / 1000)
    return result


def drive(target: TrafficCounter, tracker_id: int, class_id: int, bottoms: list[float], x: float = 100) -> list:
    results = []
    for bottom in bottoms:
        detections = make_detections([box_at(x, bottom)], [class_id], [0.9], [tracker_id])
        results.append(target.process(detections))
    return results


def test_moving_down_counts_in() -> None:
    target = counter()
    results = drive(target, 1, 2, list(range(400, 620, 20)))
    assert target.get_total_in() == 1 and target.get_total_out() == 0
    assert [r.crossed_in for r in results if r.crossed_in] == [[(1, "car")]]


def test_moving_up_counts_out() -> None:
    target = counter()
    drive(target, 2, 7, list(range(620, 400, -20)))
    assert target.get_total_out() == 1 and target.get_total_in() == 0
    assert target.get_class_counts()["truck"] == {"in": 0, "out": 1}


def test_oscillating_vehicle_is_counted_once() -> None:
    target = counter()
    path = list(range(440, 580, 20)) + list(range(560, 420, -20)) + list(range(440, 600, 20))
    drive(target, 3, 2, path)
    assert target.get_total_in() + target.get_total_out() == 1


def test_counts_by_class_include_all_vehicle_types() -> None:
    target = counter()
    drive(target, 1, 2, [460, 480, 520, 540], x=100)
    drive(target, 2, 3, [460, 480, 520, 540], x=300)
    drive(target, 3, 5, [540, 520, 480, 460], x=500)
    counts = target.get_class_counts()
    assert set(counts) == {"car", "motorcycle", "bus", "truck"}
    assert counts["car"]["in"] == 1 and counts["motorcycle"]["in"] == 1 and counts["bus"]["out"] == 1
    assert counts["truck"] == {"in": 0, "out": 0}


def test_detections_without_tracker_id_are_ignored() -> None:
    target = counter()
    result = target.process(make_detections([box_at(100, 520)], [2], [0.9]))
    assert result.crossed_in == [] and target.get_total_in() == 0
    assert target.process(sv.Detections.empty()).crossed_out == []


def test_reset_clears_counts() -> None:
    target = counter()
    drive(target, 1, 2, [460, 480, 520, 540])
    target.reset()
    assert target.get_total_in() == 0 and target.get_total_out() == 0
    drive(target, 1, 2, [460, 480, 520, 540])
    assert target.get_total_in() == 1


def test_line_position_follows_ratio() -> None:
    target = TrafficCounter(VEHICLE_IDS)
    target.initialize_line(1920, 1080, 0.25)
    assert target.line_y == 270
    assert target.line_endpoints == ((0, 270), (1920, 270))
