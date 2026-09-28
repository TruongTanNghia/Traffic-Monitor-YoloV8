"""QObject worker running a TrafficMonitor session inside a QThread."""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from app.config.settings import AppConfig
from app.core.traffic_monitor import TrafficMonitor
from app.gui.ui_helpers import bgr_to_qimage
from app.utils.constants import PREVIEW_MAX_WIDTH, STATISTICS_EMIT_INTERVAL_S
from app.utils.exceptions import TrafficMonitorError
from app.utils.logger import get_logger
from app.video.frame_processor import FrameResult

logger = get_logger(__name__)


class ProcessingWorker(QObject):
    """Owns the processing loop. Communicates with the GUI through signals only."""

    frame_ready = Signal(QImage)
    statistics_updated = Signal(dict)
    event_created = Signal(dict)
    processing_started = Signal(dict)
    processing_finished = Signal(dict)
    error_occurred = Signal(str, str, bool)  # title, message, critical

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config.copy()
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._frame_consumed = threading.Event()
        self._frame_consumed.set()
        self._last_statistics_emit = 0.0

    # ------------------------------------------------------ control (GUI thread)
    def request_stop(self) -> None:
        """Ask the loop to finish; outputs are still written."""
        self._stop_event.set()
        self._resume_event.set()

    def pause(self) -> None:
        """Pause processing after the current frame."""
        self._resume_event.clear()

    def resume(self) -> None:
        """Resume processing."""
        self._resume_event.set()

    @property
    def is_paused(self) -> bool:
        """Whether processing is paused."""
        return not self._resume_event.is_set()

    def mark_frame_consumed(self) -> None:
        """Called by the GUI after painting a frame (back-pressure for previews)."""
        self._frame_consumed.set()

    # ------------------------------------------------------- loop (worker thread)
    @Slot()
    def run(self) -> None:
        """Process the whole session. Never raises."""
        monitor = TrafficMonitor(self._config, self._stop_event, on_warning=self._emit_warning)
        status = "completed"
        try:
            monitor.open()
            self.processing_started.emit(monitor.session_info())
            for result in monitor.iter_results():
                self._emit_result(result)
                if not self._wait_while_paused():
                    break
            if self._stop_event.is_set():
                status = "stopped"
        except TrafficMonitorError as exc:
            status = "error"
            logger.error("Processing failed: %s", exc)
            self.error_occurred.emit(exc.title, str(exc), exc.critical)
        except Exception as exc:  # noqa: BLE001 - the GUI must never crash
            status = "error"
            logger.exception("Unexpected processing error")
            self.error_occurred.emit("Lỗi không mong đợi", f"{type(exc).__name__}: {exc}", True)
        finally:
            self.processing_finished.emit(self._close(monitor, status))

    def _close(self, monitor: TrafficMonitor, status: str) -> dict:
        try:
            return monitor.close(status)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to close session")
            self.error_occurred.emit("Lỗi khi kết thúc phiên", str(exc), True)
            return {"session_id": monitor.session_id, "status": "error", "frames_processed": 0,
                    "statistics": {}, "outputs": {}, "report": None, "output_dir": str(monitor.output_dir)}

    def _emit_result(self, result: FrameResult) -> None:
        for event in result.new_events:
            self.event_created.emit(event.to_dict())
        now = time.monotonic()
        if now - self._last_statistics_emit >= STATISTICS_EMIT_INTERVAL_S:
            self._last_statistics_emit = now
            self.statistics_updated.emit(result.statistics)
        if self._config.show_preview and self._frame_consumed.is_set():
            self._frame_consumed.clear()
            self.frame_ready.emit(bgr_to_qimage(result.annotated_frame, PREVIEW_MAX_WIDTH))

    def _wait_while_paused(self) -> bool:
        """Block while paused. Returns ``False`` when a stop was requested."""
        while not self._resume_event.wait(0.1):
            if self._stop_event.is_set():
                return False
        return not self._stop_event.is_set()

    def _emit_warning(self, exc: TrafficMonitorError) -> None:
        self.error_occurred.emit(exc.title, str(exc), False)
