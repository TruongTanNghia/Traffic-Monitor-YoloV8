"""Download the default YOLOv8n model (and optionally the sample video).

Usage:
    python scripts/download_model.py            # models/yolov8n.pt
    python scripts/download_model.py --sample   # + data/samples/vehicles.mp4
    python scripts/download_model.py --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.constants import MODELS_DIR, SAMPLES_DIR  # noqa: E402

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
SAMPLE_URL = "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"


def download(url: str, destination: Path, force: bool = False) -> Path:
    """Download ``url`` to ``destination`` with a simple progress indicator."""
    if destination.is_file() and not force:
        print(f"[skip] {destination} đã tồn tại")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            percent = min(100, blocks * block_size * 100 // total)
            print(f"\r  {destination.name}: {percent:3d}%", end="", flush=True)

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, temp, progress)  # noqa: S310 - fixed https URLs
    temp.replace(destination)
    print(f"\n[ok] {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8n.pt", help="Tên model Ultralytics (mặc định yolov8n.pt)")
    parser.add_argument("--sample", action="store_true", help="Tải thêm video mẫu vehicles.mp4")
    parser.add_argument("--force", action="store_true", help="Tải lại dù file đã tồn tại")
    args = parser.parse_args()
    try:
        download(MODEL_URL.format(name=args.model), MODELS_DIR / args.model, args.force)
        if args.sample:
            download(SAMPLE_URL, SAMPLES_DIR / "vehicles.mp4", args.force)
    except OSError as exc:
        print(f"\n[error] Tải thất bại: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
