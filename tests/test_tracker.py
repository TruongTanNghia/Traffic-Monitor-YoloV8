"""Unit tests for VehicleTracker (ByteTrack + DetectionsSmoother)."""

from __future__ import annotations

import supervision as sv

from app.core.tracker import VehicleTracker
from tests.conftest import box_at, make_detections


def moving_car(step: int, x: float = 100) -> sv.Detections:
    return make_detections([box_at(x, 100 + step * 3)], [2], [0.9])


def test_update_assigns_tracker_ids() -> None:
    tracker = VehicleTracker(frame_rate=25)
    tracked = tracker.update(make_detections([box_at(50, 100), box_at(300, 100)], [2, 7], [0.9, 0.8]))
    assert tracked.tracker_id is not None
    assert len(set(tracked.tracker_id.tolist())) == 2


def test_same_vehicle_keeps_its_id_across_frames() -> None:
    tracker = VehicleTracker(frame_rate=25)
    ids = {int(tracker.update(moving_car(step)).tracker_id[0]) for step in range(10)}
    assert len(ids) == 1


def test_get_track_ids_reflects_last_frame() -> None:
    tracker = VehicleTracker(frame_rate=25)
    tracked = tracker.update(make_detections([box_at(50, 100), box_at(300, 100)], [2, 2], [0.9, 0.9]))
    assert sorted(tracker.get_track_ids()) == sorted(tracked.tracker_id.tolist())


def test_reset_clears_tracks_and_restarts_ids() -> None:
    tracker = VehicleTracker(frame_rate=25)
    first_id = int(tracker.update(moving_car(0)).tracker_id[0])
    tracker.reset()
    assert tracker.get_track_ids() == []
    assert int(tracker.update(moving_car(0)).tracker_id[0]) == first_id


def test_smooth_keeps_detection_count() -> None:
    tracker = VehicleTracker(frame_rate=25)
    for step in range(6):
        tracked = tracker.update(make_detections([box_at(50, 100 + step), box_at(300, 200 + step)], [2, 3], [0.9, 0.9]))
        smoothed = tracker.smooth(tracked)
        assert len(smoothed) == len(tracked)
        assert smoothed.tracker_id is not None


def test_empty_input() -> None:
    tracker = VehicleTracker(frame_rate=25)
    assert len(tracker.update(sv.Detections.empty())) == 0
    assert len(tracker.smooth(sv.Detections.empty())) == 0
    assert tracker.get_track_ids() == []
