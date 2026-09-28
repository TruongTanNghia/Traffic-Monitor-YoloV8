"""Default configuration values and validation ranges."""

from __future__ import annotations

# Perspective points calibrated for data/samples/vehicles.mp4 (3840x2160).
# Order: top-left, top-right, bottom-right, bottom-left of the measured road area.
DEFAULT_PERSPECTIVE_SOURCE: list[list[int]] = [
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159],
]
DEFAULT_PERSPECTIVE_REFERENCE_SIZE: list[int] = [3840, 2160]

DEFAULTS: dict[str, object] = {
    "model_path": "models/yolov8n.pt",
    "confidence_threshold": 0.30,
    "speed_limit_kmh": 80.0,
    "line_y_ratio": 0.55,
    "road_width_m": 25.0,
    "road_length_m": 250.0,
    "perspective_source": DEFAULT_PERSPECTIVE_SOURCE,
    "perspective_reference_size": DEFAULT_PERSPECTIVE_REFERENCE_SIZE,
    "show_preview": True,
    "save_output_video": True,
    "save_report": True,
    "save_snapshots": True,
    "source_type": "file",
    "source_uri": "data/samples/vehicles.mp4",
    "output_dir": "data/output",
    "min_track_seconds": 0.5,
    "device": "auto",
    "rtsp_timeout_ms": 5000,
}

RANGES: dict[str, tuple[float, float]] = {
    "confidence_threshold": (0.05, 0.95),
    "speed_limit_kmh": (10.0, 250.0),
    "line_y_ratio": (0.10, 0.90),
    "road_width_m": (0.5, 200.0),
    "road_length_m": (1.0, 1000.0),
    "min_track_seconds": (0.2, 3.0),
    "rtsp_timeout_ms": (1000, 60000),
}
