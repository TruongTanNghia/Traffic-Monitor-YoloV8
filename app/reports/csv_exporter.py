"""Write the event list of a session as CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.utils.file_utils import ensure_dir
from app.utils.logger import get_logger

logger = get_logger(__name__)

CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "frame_index",
    "video_time_s",
    "tracker_id",
    "vehicle_type",
    "speed_kmh",
    "speed_limit_kmh",
    "event_type",
)


class CsvExporter:
    """Serialise events to CSV (UTF-8 with BOM so Excel shows accents)."""

    def export(self, events: Iterable[dict[str, Any]], destination: Path) -> Path:
        """Write ``events`` to ``destination`` and return the path.

        Raises:
            OSError: when the file cannot be written.
        """
        ensure_dir(destination.parent)
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for event in events:
                writer.writerow({key: "" if event.get(key) is None else event.get(key) for key in CSV_COLUMNS})
        logger.info("Event CSV saved: %s", destination)
        return destination
