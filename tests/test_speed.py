"""Unit tests for SpeedEstimator, geometry validation and AlertManager."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.alert_manager import AlertManager
from app.core.speed_estimator import SpeedEstimator
from app.utils.constants import EventType
from app.utils.exceptions import PerspectiveError
from app.vision.geometry import occluded_bottom_mask, validate_quadrilateral
from tests.conftest import box_at, make_detections

FPS = 25


def estimator(square: np.ndarray, min_seconds: float = 0.5) -> SpeedEstimator:
    result = SpeedEstimator(fps=FPS, min_track_seconds=min_seconds)
    result.initialize_transformer(square, 100, 100)
    return result


def run_constant_speed(target: SpeedEstimator, metres_per_second: float, frames: int, tracker_id: int = 5) -> list:
    speeds = []
    for frame in range(frames):
        bottom = 10 + metres_per_second / FPS * frame
        detections = make_detections([box_at(50, bottom, width=10, height=5)], [2], [0.9], [tracker_id])
        speeds.append(target.update(detections, frame / FPS).get(tracker_id))
    return speeds


def test_no_speed_on_first_frame(square_perspective: np.ndarray) -> None:
    speeds = run_constant_speed(estimator(square_perspective), 20, 1)
    assert speeds == [None]


def test_no_speed_before_min_track_seconds(square_perspective: np.ndarray) -> None:
    speeds = run_constant_speed(estimator(square_perspective, min_seconds=0.5), 20, 30)
    first_index = next(i for i, s in enumerate(speeds) if s is not None)
    assert first_index / FPS >= 0.5
    assert all(s is None for s in speeds[:first_index])


def test_constant_motion_gives_expected_kmh(square_perspective: np.ndarray) -> None:
    speeds = run_constant_speed(estimator(square_perspective), 20, 40)
    assert speeds[-1] == pytest.approx(72.0, abs=1.0)


def test_single_anchor_jump_does_not_explode_speed(square_perspective: np.ndarray) -> None:
    target = estimator(square_perspective)
    for frame in range(25):
        bottom = 10 + 0.8 * frame + (6 if frame == 24 else 0)  # sudden 6 m jump on last frame
        target.update(make_detections([box_at(50, bottom, 10, 5)], [2], [0.9], [1]), frame / FPS)
    assert target.get_speed(1) < 110  # endpoint difference would give ~94 km/h + jump ≈ 180


def test_occluded_vehicle_is_not_measured(square_perspective: np.ndarray) -> None:
    target = estimator(square_perspective)
    for frame in range(30):
        far = box_at(50, 40 + 0.8 * frame, 10, 10)
        front = [40, 38 + 0.8 * frame, 60, 70 + 0.8 * frame]  # covers the bottom of the far car
        target.update(make_detections([far, front], [2, 2], [0.9, 0.9], [1, 2]), frame / FPS)
    assert target.get_speed(1) is None
    assert target.get_speed(2) == pytest.approx(72.0, abs=1.0)


def test_occlusion_mask() -> None:
    boxes = np.array([[40, 20, 60, 50], [38, 45, 62, 80], [200, 20, 220, 50]], dtype=float)
    assert occluded_bottom_mask(boxes).tolist() == [True, False, False]


def test_get_speed_violations(square_perspective: np.ndarray) -> None:
    target = estimator(square_perspective)
    run_constant_speed(target, 25, 40, tracker_id=7)  # 90 km/h
    assert target.get_speed_violations(80) == [7]
    assert target.get_speed_violations(100) == []


def test_reset_clears_history(square_perspective: np.ndarray) -> None:
    target = estimator(square_perspective)
    run_constant_speed(target, 20, 40)
    target.reset()
    assert target.get_speed(5) is None


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [10, 0], [10, 10]],  # 3 points
        [[0, 0], [100, 0], [0, 100], [100, 100]],  # self-intersecting
        [[0, 0], [50, 0], [100, 0], [0, 100]],  # collinear
        [[0, 0], ["a", 0], [1, 1], [0, 1]],  # not numeric
    ],
)
def test_invalid_perspective_points_raise(points: list) -> None:
    with pytest.raises(PerspectiveError):
        validate_quadrilateral(points)


def test_points_outside_frame_are_valid() -> None:
    assert validate_quadrilateral([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]]).shape == (4, 2)


def test_alert_manager_creates_one_violation_per_vehicle() -> None:
    alerts = AlertManager(speed_limit_kmh=80)
    assert alerts.check_speed(1, "car", 80.0) is None
    alert = alerts.check_speed(1, "car", 92.4, frame_index=40, time_s=1.6)
    assert alert is not None and alert.event_type == EventType.SPEED_VIOLATION.value
    data = alert.to_dict()
    for key in ("timestamp", "tracker_id", "vehicle_type", "speed_kmh", "speed_limit_kmh", "event_type"):
        assert key in data
    assert alerts.check_speed(1, "car", 120.0) is None
    assert len(alerts.get_alert_history()) == 1
    alerts.clear_alerts()
    assert alerts.get_alert_history() == [] and alerts.get_event_history() == []
