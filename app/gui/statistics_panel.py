"""Eight live statistic cards + system information panel."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QFrame, QGridLayout, QWidget

from app.gui.ui_helpers import key_label, key_value_label, make_panel
from app.gui.widgets.stat_card import StatCard

CARD_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("total_vehicles", "Total Vehicles", "total"),
    ("total_in", "Vehicles IN", "in"),
    ("total_out", "Vehicles OUT", "out"),
    ("car", "Cars", ""),
    ("motorcycle", "Motorcycles", ""),
    ("bus", "Buses", ""),
    ("truck", "Trucks", ""),
    ("speed_violations", "Speed Violations", "violation"),
)

SYSTEM_FIELDS: tuple[tuple[str, str], ...] = (
    ("fps", "FPS"),
    ("status", "Processing status"),
    ("source", "Current source"),
    ("model", "Model"),
    ("device", "Device"),
    ("confidence", "Confidence threshold"),
    ("speed_limit", "Speed limit"),
    ("active_tracks", "Active tracks"),
    ("session", "Session"),
)


class StatisticsCards(QWidget):
    """Row of the eight required stat cards."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cards: dict[str, StatCard] = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        for column, (key, title, accent) in enumerate(CARD_DEFINITIONS):
            card = StatCard(title, accent)
            self.cards[key] = card
            grid.addWidget(card, 0, column)

    def update_statistics(self, stats: dict[str, Any]) -> None:
        """Apply a ``statistics`` snapshot from the pipeline."""
        totals = stats.get("class_totals", {})
        for key, card in self.cards.items():
            value = stats.get(key, totals.get(key, 0))
            card.set_value(value)
        self.cards["speed_violations"].set_hot(int(stats.get("speed_violations", 0)) > 0)

    def reset(self) -> None:
        """Show zeros."""
        for card in self.cards.values():
            card.set_value(0)
        self.cards["speed_violations"].set_hot(False)


class SystemInfoPanel(QFrame):
    """Key/value information about the running system."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        panel, body = make_panel("System")
        self.values = {key: key_value_label() for key, _ in SYSTEM_FIELDS}
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        for row, (key, label) in enumerate(SYSTEM_FIELDS):
            grid.addWidget(key_label(label), row, 0)
            grid.addWidget(self.values[key], row, 1)
        body.addLayout(grid)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(panel)

    def set_value(self, key: str, value: object) -> None:
        """Update one field."""
        label = self.values[key]
        text = str(value)
        label.setText(text)
        label.setToolTip(text)

    def update_many(self, values: dict[str, object]) -> None:
        """Update several fields."""
        for key, value in values.items():
            if key in self.values:
                self.set_value(key, value)
