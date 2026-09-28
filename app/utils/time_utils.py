"""Time helpers: session ids, ISO timestamps, FPS measurement."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime


def make_session_id(moment: datetime | None = None) -> str:
    """Return ``YYYYMMDD_HHMMSS`` for ``moment`` (default: now)."""
    return (moment or datetime.now()).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    """Return the current local time as ISO 8601 without microseconds."""
    return datetime.now().replace(microsecond=0).isoformat()


def iso_to_clock(value: str) -> str:
    """Convert an ISO timestamp to ``HH:MM:SS`` for display."""
    try:
        return datetime.fromisoformat(value).strftime("%H:%M:%S")
    except ValueError:
        return value


class FpsMeter:
    """Moving-window frames-per-second meter."""

    def __init__(self, window: int = 30) -> None:
        self._stamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Register a frame and return the current FPS estimate."""
        self._stamps.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        """Current FPS (0 until two frames were registered)."""
        if len(self._stamps) < 2:
            return 0.0
        elapsed = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        """Forget all registered frames."""
        self._stamps.clear()
