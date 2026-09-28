"""Pick the 4 perspective points on the first frame and save them to config.yaml.

This standalone tool uses an OpenCV window (it is not part of the GUI). The same
feature is available in the application: Configuration → *Calibrate from video…*.

Usage:
    python scripts/calibrate_perspective.py --video data/samples/vehicles.mp4 --width 25 --length 250

Click A (top-left), B (top-right), C (bottom-right), D (bottom-left) of a road
segment whose real size is width × length metres. Keys: r = reset, s/Enter =
save, q/Esc = quit without saving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import AppConfig, load_config_safely  # noqa: E402
from app.utils.constants import CONFIG_FILE  # noqa: E402
from app.utils.exceptions import PerspectiveError, TrafficMonitorError  # noqa: E402
from app.video.video_source import read_first_frame  # noqa: E402
from app.vision.geometry import validate_quadrilateral  # noqa: E402

WINDOW = "Calibrate perspective (A-B-C-D, r=reset, s=save, q=quit)"
DISPLAY_WIDTH = 1280
MARGIN = 0.2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="Video file (mặc định: source_uri trong config)")
    parser.add_argument("--width", type=float, help="Road width in metres")
    parser.add_argument("--length", type=float, help="Road length in metres")
    args = parser.parse_args()

    config, _ = load_config_safely(CONFIG_FILE)
    if args.video:
        config.source_type, config.source_uri = "file", args.video
    try:
        frame = read_first_frame(config)
    except TrafficMonitorError as exc:
        print(f"[FAIL] {exc}")
        return 1
    points = pick_points(frame)
    if points is None:
        print("Cancelled.")
        return 0
    return save(config, points, frame.shape, args)


def pick_points(frame: np.ndarray) -> list[list[int]] | None:
    height, width = frame.shape[:2]
    scale = DISPLAY_WIDTH / width
    pad_x, pad_y = int(width * scale * MARGIN), int(height * scale * MARGIN)
    base = cv2.copyMakeBorder(cv2.resize(frame, None, fx=scale, fy=scale), pad_y, pad_y, pad_x, pad_x,
                              cv2.BORDER_CONSTANT, value=(20, 20, 20))
    points: list[list[int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([round((x - pad_x) / scale), round((y - pad_y) / scale)])

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        canvas = base.copy()
        shown = [(int(px * scale) + pad_x, int(py * scale) + pad_y) for px, py in points]
        if len(shown) > 1:
            cv2.polylines(canvas, [np.array(shown)], len(shown) == 4, (65, 164, 217), 2)
        for index, point in enumerate(shown):
            cv2.circle(canvas, point, 6, (65, 164, 217), -1)
            cv2.putText(canvas, "ABCD"[index], (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None
        if key == ord("r"):
            points.clear()
        if key in (ord("s"), 13) and len(points) == 4:
            cv2.destroyAllWindows()
            return points


def save(config: AppConfig, points: list[list[int]], shape: tuple[int, ...], args: argparse.Namespace) -> int:
    try:
        validate_quadrilateral(points)
    except PerspectiveError as exc:
        print(f"[FAIL] {exc}")
        return 1
    config.perspective_source = points
    config.perspective_reference_size = [int(shape[1]), int(shape[0])]
    if args.width:
        config.road_width_m = args.width
    if args.length:
        config.road_length_m = args.length
    try:
        config.save(CONFIG_FILE)
    except TrafficMonitorError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print(f"[ok] Saved to {CONFIG_FILE}: {points} (reference {config.perspective_reference_size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
