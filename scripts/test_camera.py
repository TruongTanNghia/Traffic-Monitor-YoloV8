"""Check that a webcam or RTSP stream can be opened and read.

Usage:
    python scripts/test_camera.py --camera 0
    python scripts/test_camera.py --rtsp rtsp://user:pass@192.168.1.10:554/stream1 --timeout 5000
    python scripts/test_camera.py --video data/samples/vehicles.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.exceptions import TrafficMonitorError  # noqa: E402
from app.video.video_source import (  # noqa: E402
    CameraVideoSource,
    FileVideoSource,
    RTSPVideoSource,
    VideoSource,
)


def build_source(args: argparse.Namespace) -> VideoSource:
    if args.rtsp:
        return RTSPVideoSource(args.rtsp, args.timeout)
    if args.video:
        return FileVideoSource(args.video)
    return CameraVideoSource(args.camera)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, default=0, help="Camera index (mặc định 0)")
    group.add_argument("--rtsp", help="RTSP URL")
    group.add_argument("--video", help="Video file")
    parser.add_argument("--timeout", type=int, default=5000, help="RTSP timeout (ms)")
    parser.add_argument("--frames", type=int, default=60, help="Số frame cần đọc")
    args = parser.parse_args()

    try:
        source = build_source(args)
        started = time.perf_counter()
        source.open()
    except TrafficMonitorError as exc:
        print(f"[FAIL] {exc.title}: {exc}")
        return 1
    info = source.info()
    print(f"[ok] Opened {info.description} in {time.perf_counter() - started:.2f}s "
          f"— {info.width}x{info.height} @ {info.fps:.1f} fps")
    read, started = 0, time.perf_counter()
    try:
        while read < args.frames:
            ok, _ = source.read()
            if not ok:
                break
            read += 1
    finally:
        source.release()
    elapsed = time.perf_counter() - started
    if read == 0:
        print("[FAIL] Không đọc được frame nào.")
        return 1
    print(f"[ok] Read {read} frames in {elapsed:.2f}s ({read / elapsed:.1f} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
