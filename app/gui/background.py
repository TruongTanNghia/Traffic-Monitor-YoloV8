"""Run short blocking jobs (probing a source, reading a frame) off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app.utils.exceptions import TrafficMonitorError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TaskSignals(QObject):
    """Signals of a :class:`BackgroundTask` (delivered in the GUI thread)."""

    succeeded = Signal(object)
    failed = Signal(str, str, bool)  # title, message, critical


class BackgroundTask(QRunnable):
    """Execute ``function`` in the global thread pool."""

    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self._function = function
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            result = self._function()
        except TrafficMonitorError as exc:
            self.signals.failed.emit(exc.title, str(exc), exc.critical)
        except Exception as exc:  # noqa: BLE001 - never crash a pool thread
            logger.exception("Background task failed")
            self.signals.failed.emit("Lỗi không mong đợi", f"{type(exc).__name__}: {exc}", True)
        else:
            self.signals.succeeded.emit(result)


class TaskRunner:
    """Keep references to running tasks so their signals are not garbage collected."""

    def __init__(self) -> None:
        self._tasks: set[BackgroundTask] = set()

    def start(
        self,
        function: Callable[[], Any],
        on_success: Callable[[Any], None],
        on_failure: Callable[[str, str, bool], None],
    ) -> BackgroundTask:
        """Run ``function`` and dispatch its result to the callbacks."""
        task = BackgroundTask(function)
        task.setAutoDelete(False)
        self._tasks.add(task)
        task.signals.succeeded.connect(on_success)
        task.signals.failed.connect(on_failure)
        task.signals.succeeded.connect(lambda _=None, t=task: self._tasks.discard(t))
        task.signals.failed.connect(lambda *_, t=task: self._tasks.discard(t))
        QThreadPool.globalInstance().start(task)
        return task

    @property
    def busy(self) -> bool:
        """Whether a task is still running."""
        return bool(self._tasks)
