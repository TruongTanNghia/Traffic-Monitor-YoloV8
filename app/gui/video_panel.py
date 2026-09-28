"""Live video area with START / PAUSE / STOP controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets.video_view import VideoView

STATE_TEXT = {
    "IDLE": "IDLE",
    "STARTING": "LOADING…",
    "RUNNING": "RUNNING",
    "PAUSED": "PAUSED",
    "STOPPING": "STOPPING…",
    "FINISHED": "FINISHED",
    "ERROR": "ERROR",
}


class VideoPanel(QWidget):
    """Video view + control buttons. Emits requests; the MainWindow decides."""

    start_requested = Signal()
    pause_toggled = Signal()
    stop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_view = VideoView()
        self.start_button = QPushButton("▶  START")
        self.start_button.setObjectName("PrimaryButton")
        self.pause_button = QPushButton("❚❚  PAUSE")
        self.stop_button = QPushButton("■  STOP")
        self.stop_button.setObjectName("DangerButton")
        self.state_pill = QLabel("IDLE")
        self.state_pill.setObjectName("StatePill")
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(8)
        self.progress.setTextVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Hint")

        self.start_button.clicked.connect(self.start_requested)
        self.pause_button.clicked.connect(self.pause_toggled)
        self.stop_button.clicked.connect(self.stop_requested)

        controls = QHBoxLayout()
        controls.addWidget(self.state_pill)
        controls.addStretch(1)
        for button in (self.start_button, self.pause_button, self.stop_button):
            button.setMinimumWidth(96)
            controls.addWidget(button)

        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.progress_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(controls)
        layout.addWidget(self.video_view, 1)
        layout.addLayout(progress_row)
        self.set_state("IDLE")

    def set_state(self, state: str) -> None:
        """Enable/disable buttons according to the session state."""
        self.state_pill.setText(STATE_TEXT.get(state, state))
        self.state_pill.setProperty("state", state)
        self.state_pill.style().unpolish(self.state_pill)
        self.state_pill.style().polish(self.state_pill)
        active = state in ("RUNNING", "PAUSED")
        self.start_button.setEnabled(state in ("IDLE", "FINISHED", "ERROR"))
        self.pause_button.setEnabled(active)
        self.stop_button.setEnabled(active or state == "STARTING")
        self.pause_button.setText("▶  RESUME" if state == "PAUSED" else "❚❚  PAUSE")
        if state == "STARTING":
            self.video_view.clear("Đang load model và mở nguồn video…")

    def show_frame(self, image: QImage) -> None:
        """Display a processed frame."""
        self.video_view.show_image(image)

    def set_progress(self, frame_index: int, frame_count: int | None) -> None:
        """Show progress for file sources; hide it for live sources."""
        if not frame_count:
            self.progress.setRange(0, 0 if frame_index >= 0 else 1)
            self.progress_label.setText(f"frame {frame_index + 1}" if frame_index >= 0 else "")
            return
        self.progress.setRange(0, frame_count)
        self.progress.setValue(min(frame_index + 1, frame_count))
        self.progress_label.setText(f"{frame_index + 1} / {frame_count}")

    def reset_progress(self) -> None:
        """Clear the progress bar."""
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText("")
