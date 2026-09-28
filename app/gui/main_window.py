"""Main window: navigation, pages, session lifecycle and configuration state."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppConfig
from app.core.detector import describe_device, resolve_device
from app.gui.background import TaskRunner
from app.gui.dashboard import DashboardPage
from app.gui.log_panel import LogPanel
from app.gui.processing_worker import ProcessingWorker
from app.gui.report_panel import ReportPanel
from app.gui.settings_panel import SettingsPanel
from app.gui.source_panel import SourcePanel, describe_uri
from app.gui.ui_helpers import confirm, show_info, show_message
from app.utils.constants import APP_NAME, APP_VERSION, CONFIG_FILE, EventType, SessionState
from app.utils.exceptions import ConfigError
from app.utils.logger import get_logger

logger = get_logger(__name__)

PAGES: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "Dashboard", "MONITOR"),
    ("source", "Video Source", "MONITOR"),
    ("config", "Configuration", "MONITOR"),
    ("reports", "Reports", "DATA"),
    ("logs", "Event Logs", "DATA"),
)
WORKER_SHUTDOWN_TIMEOUT_MS = 20000


class MainWindow(QMainWindow):
    """Top-level window. Owns the configuration and the processing thread."""

    def __init__(self, config: AppConfig, config_error: str | None = None) -> None:
        super().__init__()
        self.config = config
        self._runner = TaskRunner()
        self._thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self._state = SessionState.IDLE.value
        self._session_info: dict[str, Any] = {}
        self._unseen_violations = 0
        self._last_error: tuple[str, str] | None = None
        self._closing = False

        self.dashboard = DashboardPage()
        self.source_panel = SourcePanel(config, self._runner)
        self.settings_panel = SettingsPanel(config, self._runner)
        self.report_panel = ReportPanel(config.output_path)
        self.log_panel = LogPanel()
        self._pages = {"dashboard": self.dashboard, "source": self.source_panel, "config": self.settings_panel,
                       "reports": self.report_panel, "logs": self.log_panel}
        self._nav_buttons: dict[str, QPushButton] = {}
        self._stack = QStackedWidget()
        self._status_state = QLabel()
        self._status_session = QLabel()
        self._status_device = QLabel()

        self._build()
        self._connect()
        self._apply_config_to_ui()
        self._set_state(SessionState.IDLE.value)
        self.setWindowTitle(f"{APP_NAME} — v{APP_VERSION}")
        self.resize(1480, 900)
        self.setMinimumSize(1200, 760)
        if config_error:
            show_message(self, "Cấu hình không hợp lệ",
                         f"{config_error}\n\nỨng dụng dùng cấu hình mặc định.", critical=False)

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_nav())
        for page in self._pages.values():
            self._stack.addWidget(page)
        root.addWidget(self._stack, 1)
        self.setCentralWidget(central)
        status = self.statusBar()
        status.addWidget(self._status_state)
        status.addWidget(self._status_session)
        status.addPermanentWidget(self._status_device)

    def _build_nav(self) -> QFrame:
        nav = QFrame()
        nav.setObjectName("NavBar")
        nav.setFixedWidth(190)
        layout = QVBoxLayout(nav)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(1)
        group = QButtonGroup(self)
        group.setExclusive(True)
        current_group = ""
        for index, (key, title, group_name) in enumerate(PAGES):
            if group_name != current_group:
                label = QLabel(group_name)
                label.setObjectName("NavGroup")
                layout.addWidget(label)
                current_group = group_name
            button = QPushButton(title)
            button.setCheckable(True)
            button.setToolTip(f"Ctrl+{index + 1}")
            group.addButton(button)
            button.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self._nav_buttons[key] = button
            layout.addWidget(button)
            QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self, activated=lambda k=key: self.show_page(k))
        layout.addStretch(1)
        self._nav_footer = QLabel()
        self._nav_footer.setObjectName("NavFooter")
        self._nav_footer.setWordWrap(True)
        layout.addWidget(self._nav_footer)
        self._nav_buttons["dashboard"].setChecked(True)
        return nav

    def _connect(self) -> None:
        panel = self.dashboard.video_panel
        panel.start_requested.connect(self.start_processing)
        panel.pause_toggled.connect(self.toggle_pause)
        panel.stop_requested.connect(self.stop_processing)
        self.source_panel.source_applied.connect(self._on_source_applied)
        self.settings_panel.config_saved.connect(self._on_config_saved)

    def show_page(self, key: str) -> None:
        """Switch to a page by key."""
        self._stack.setCurrentWidget(self._pages[key])
        self._nav_buttons[key].setChecked(True)
        if key == "logs":
            self._unseen_violations = 0
            self._refresh_log_badge()

    # ----------------------------------------------------------------- config
    def _apply_config(self, config: AppConfig) -> None:
        self.config = config
        self.source_panel.set_config(config)
        self.settings_panel.set_config(config)
        self.report_panel.set_output_dir(config.output_path)
        self._apply_config_to_ui()

    def _apply_config_to_ui(self) -> None:
        config = self.config
        source_name = describe_uri(config)
        device = describe_device(resolve_device(config.device))
        self.dashboard.system_panel.update_many({
            "source": f"{config.source_type} · {source_name}",
            "model": config.model_file.name,
            "device": device,
            "confidence": f"{config.confidence_threshold:.2f}",
            "speed_limit": f"{config.speed_limit_kmh:.0f} km/h",
        })
        self.dashboard.source_label.setText(f"{config.source_type.upper()} · {source_name}")
        self._nav_footer.setText(f"MODEL  {config.model_file.name}\nDEVICE {config.device} → {resolve_device(config.device)}")
        self._status_device.setText(device)

    def _on_config_saved(self, config: AppConfig) -> None:
        self._apply_config(config)
        self.statusBar().showMessage(f"Đã lưu {CONFIG_FILE.name}", 5000)
        logger.info("Configuration updated from Settings page")

    def _on_source_applied(self, source_type: str, uri: str, timeout_ms: int) -> None:
        config = self.config.copy()
        config.source_type, config.source_uri, config.rtsp_timeout_ms = source_type, uri, timeout_ms
        try:
            config.save(CONFIG_FILE)
        except (ConfigError, OSError) as exc:
            show_message(self, "Không lưu được nguồn video", str(exc), critical=False)
            return
        self._apply_config(config)
        self.statusBar().showMessage(f"Nguồn video: {source_type} · {describe_uri(config)}", 5000)

    # ---------------------------------------------------------------- session
    def start_processing(self) -> None:
        """START: validate config and launch the worker thread."""
        if self._thread is not None:
            return
        if self.settings_panel.is_dirty and not confirm(
                self, "Cấu hình chưa lưu",
                "Trang Configuration có thay đổi chưa lưu và sẽ KHÔNG được dùng.\nTiếp tục START với cấu hình đã lưu?"):
            return
        try:
            self.config.validate()
        except ConfigError as exc:
            show_message(self, exc.title, str(exc), critical=False)
            return
        self._last_error = None
        self._unseen_violations = 0
        self.dashboard.reset_session()
        self.log_panel.clear()
        self._refresh_log_badge()
        self._launch_worker(self.config)
        self._set_state(SessionState.STARTING.value)

    def _launch_worker(self, config: AppConfig) -> None:
        thread = QThread(self)
        worker = ProcessingWorker(config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.processing_finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.processing_started.connect(self._on_started)
        worker.frame_ready.connect(self._on_frame)
        worker.statistics_updated.connect(self._on_statistics)
        worker.event_created.connect(self._on_event)
        worker.error_occurred.connect(self._on_error)
        worker.processing_finished.connect(self._on_finished)
        thread.finished.connect(self._on_thread_finished)
        self._thread, self._worker = thread, worker
        thread.start()
        logger.info("Processing thread started")

    def toggle_pause(self) -> None:
        """PAUSE / RESUME."""
        if self._worker is None:
            return
        if self._worker.is_paused:
            self._worker.resume()
            self._set_state(SessionState.RUNNING.value)
        else:
            self._worker.pause()
            self._set_state(SessionState.PAUSED.value)

    def stop_processing(self) -> None:
        """STOP: finish the session; video and reports are still written."""
        if self._worker is None:
            return
        self._worker.request_stop()
        self._set_state(SessionState.STOPPING.value)

    # ---------------------------------------------------------- worker slots
    def _on_started(self, info: dict[str, Any]) -> None:
        self._session_info = info
        self._set_state(SessionState.RUNNING.value)
        source = info.get("source", {})
        self.dashboard.system_panel.update_many({
            "device": info.get("device", "—"),
            "session": info.get("session_id", "—"),
            "source": f"{source.get('type', '')} · {source.get('description', '')} "
                      f"({source.get('resolution', ['?', '?'])[0]}×{source.get('resolution', ['?', '?'])[1]})",
        })
        self.dashboard.video_panel.set_progress(-1, source.get("frame_count"))
        self.source_panel.show_info(source)
        self.report_panel.session_started(info)
        self._status_session.setText(f"session {info.get('session_id', '')}")
        self._status_device.setText(str(info.get("device", "")))

    def _on_frame(self, image: QImage) -> None:
        self.dashboard.video_panel.show_frame(image)
        if self._worker is not None:
            self._worker.mark_frame_consumed()

    def _on_statistics(self, stats: dict[str, Any]) -> None:
        self.dashboard.cards.update_statistics(stats)
        self.dashboard.system_panel.update_many({
            "fps": f"{stats.get('fps', 0):.1f}",
            "active_tracks": stats.get("active_tracks", 0),
        })
        frame_count = self._session_info.get("source", {}).get("frame_count")
        self.dashboard.video_panel.set_progress(int(stats.get("frame_index", 0)), frame_count)
        self.report_panel.update_live(stats)

    def _on_event(self, event: dict[str, Any]) -> None:
        self.dashboard.add_event(event)
        self.log_panel.add_event(event)
        if event.get("event_type") == EventType.SPEED_VIOLATION.value and self._stack.currentWidget() is not self.log_panel:
            self._unseen_violations += 1
            self._refresh_log_badge()

    def _on_error(self, title: str, message: str, critical: bool) -> None:
        self._last_error = (title, message)
        if self._closing:
            return
        show_message(self, title, message, critical)

    def _on_finished(self, summary: dict[str, Any]) -> None:
        failed = summary.get("status") == "error"
        self._set_state(SessionState.ERROR.value if failed else SessionState.FINISHED.value)
        if summary.get("statistics"):
            self.dashboard.cards.update_statistics(summary["statistics"])
        self.report_panel.set_summary(summary)
        self.dashboard.video_panel.video_view.set_placeholder("Phiên xử lý đã kết thúc — nhấn START để chạy lại")
        if not failed and not self._closing and summary.get("frames_processed"):
            self._show_finished_message(summary)

    def _show_finished_message(self, summary: dict[str, Any]) -> None:
        outputs = summary.get("outputs", {})
        stats = summary.get("statistics", {})
        lines = [
            f"Trạng thái: {summary.get('status')} · {summary.get('frames_processed')} frames",
            f"IN {stats.get('total_in', 0)} · OUT {stats.get('total_out', 0)} · "
            f"Vi phạm {stats.get('speed_violations', 0)}",
            "",
            f"Video:  {outputs.get('video') or '(không lưu)'}",
            f"JSON:   {outputs.get('json') or '(không lưu)'}",
            f"CSV:    {outputs.get('csv') or '(không lưu)'}",
        ]
        self.statusBar().showMessage("Phiên xử lý hoàn tất — xem trang Reports", 8000)
        show_info(self, "Hoàn tất xử lý", "\n".join(lines))

    def _on_thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker, self._thread = None, None
        logger.info("Processing thread finished")

    # ------------------------------------------------------------------ state
    def _set_state(self, state: str) -> None:
        self._state = state
        self.dashboard.video_panel.set_state(state)
        busy = state in ("STARTING", "RUNNING", "PAUSED", "STOPPING")
        self.settings_panel.set_locked(busy)
        self.source_panel.set_locked(busy)
        status_text = {
            "IDLE": "Idle", "STARTING": "Loading model…", "RUNNING": "Processing", "PAUSED": "Paused",
            "STOPPING": "Stopping…", "FINISHED": "Finished", "ERROR": "Error",
        }[state]
        self.dashboard.system_panel.set_value("status", status_text)
        if state != "RUNNING":
            self.dashboard.system_panel.set_value("fps", "—" if state != "PAUSED" else "0.0")
        self._status_state.setText(f"● {status_text}")

    @property
    def state(self) -> str:
        """Current session state (IDLE, STARTING, RUNNING, PAUSED, STOPPING, FINISHED, ERROR)."""
        return self._state

    def _refresh_log_badge(self) -> None:
        count = self._unseen_violations
        self._nav_buttons["logs"].setText(f"Event Logs  ({count} ⚠)" if count else "Event Logs")

    # ------------------------------------------------------------------ close
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        """Stop the worker safely before closing."""
        if self._thread is not None and self._worker is not None:
            if not confirm(self, "Đang xử lý", "Phiên xử lý đang chạy. Dừng và thoát?\n(Video/báo cáo đã xử lý sẽ được lưu.)"):
                event.ignore()
                return
            self._closing = True
            self._worker.request_stop()
            self._thread.quit()
            if not self._thread.wait(WORKER_SHUTDOWN_TIMEOUT_MS):
                logger.error("Worker did not stop in time; terminating")
                self._thread.terminate()
                self._thread.wait()
        logger.info("Application closed")
        event.accept()
