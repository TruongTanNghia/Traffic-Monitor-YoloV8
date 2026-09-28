"""Event Logs page: realtime event table with filters."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.ui_helpers import page_header
from app.gui.widgets.log_view import LogView
from app.utils.constants import EventType


class LogPanel(QWidget):
    """Full event log of the current session."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.log_view = LogView()
        self.type_filter = QComboBox()
        self.type_filter.addItem("All events", "")
        for event_type in EventType:
            self.type_filter.addItem(event_type.value, event_type.value)
        self.id_filter = QLineEdit()
        self.id_filter.setPlaceholderText("Track ID")
        self.id_filter.setMaximumWidth(110)
        self.auto_scroll = QCheckBox("Auto-scroll")
        self.auto_scroll.setChecked(True)
        self.clear_button = QPushButton("Clear view")
        self.count_label = QLabel("0 events")
        self.count_label.setObjectName("Hint")
        self.count_label.setMinimumWidth(140)
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.type_filter)
        toolbar.addWidget(self.id_filter)
        toolbar.addWidget(self.auto_scroll)
        toolbar.addStretch(1)
        toolbar.addWidget(self.count_label)
        toolbar.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(page_header("Event Logs", "REALTIME · event_created"))
        layout.addLayout(toolbar)
        layout.addWidget(self.log_view, 1)

        self.type_filter.currentIndexChanged.connect(self._apply_filters)
        self.id_filter.textChanged.connect(self._apply_filters)
        self.auto_scroll.toggled.connect(self._set_auto_scroll)
        self.clear_button.clicked.connect(self.clear)

    def add_event(self, event: dict[str, Any]) -> None:
        """Append one event."""
        self.log_view.add_event(event)
        self._update_count()

    def clear(self) -> None:
        """Clear the table (session data and reports are not affected)."""
        self.log_view.clear()
        self._update_count()

    def _apply_filters(self) -> None:
        self.log_view.set_event_type_filter(self.type_filter.currentData() or "")
        self.log_view.set_tracker_filter(self.id_filter.text())
        self._update_count()

    def _set_auto_scroll(self, enabled: bool) -> None:
        self.log_view.auto_scroll = enabled
        if enabled:
            self.log_view.scrollToBottom()

    def _update_count(self) -> None:
        total, visible = self.log_view.total_count(), self.log_view.visible_count()
        self.count_label.setText(f"{visible} / {total} events" if visible != total else f"{total} events")
