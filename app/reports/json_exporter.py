"""Write session reports as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.utils.file_utils import ensure_dir
from app.utils.logger import get_logger

logger = get_logger(__name__)


class JsonExporter:
    """Serialise a report dictionary to a UTF-8 JSON file."""

    def export(self, report: dict[str, Any], destination: Path) -> Path:
        """Write ``report`` to ``destination`` and return the path.

        Raises:
            OSError: when the file cannot be written.
        """
        ensure_dir(destination.parent)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Report saved: %s", destination)
        return destination
