"""Dashboard page: live video, controls, statistics, system info, live events."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.gui.statistics_panel import StatisticsCards, SystemInfoPanel
from app.gui.ui_helpers import make_panel, page_header
from app.gui.video_panel import VideoPanel
from app.gui.widgets.log_view import LogView

RECENT_EVENTS_LIMIT = 300


class DashboardPage(QWidget):
    """Main monitoring page."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_panel = VideoPanel()
        self.system_panel = SystemInfoPanel()
        self.cards = StatisticsCards()
        self.recent_events = LogView(max_rows=RECENT_EVENTS_LIMIT, compact=True)
        self.source_label = QLabel("")
        self.source_label.setObjectName("PageSubtitle")

        events_panel, events_body = make_panel("Live events")
        events_body.setContentsMargins(0, 0, 0, 0)
        events_body.addWidget(self.recent_events)

        side = QVBoxLayout()
        side.setSpacing(10)
        side.addWidget(self.system_panel)
        side.addWidget(events_panel, 1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(340)
        side.setContentsMargins(0, 0, 0, 0)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.video_panel, 1)
        body.addWidget(side_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(page_header("Dashboard", "LIVE VIDEO", [self.source_label]))
        layout.addLayout(body, 1)
        layout.addWidget(self.cards)

    def add_event(self, event: dict[str, Any]) -> None:
        """Append a live event."""
        self.recent_events.add_event(event)

    def reset_session(self) -> None:
        """Clear per-session widgets before a new START."""
        self.recent_events.clear()
        self.cards.reset()
        self.video_panel.reset_progress()
