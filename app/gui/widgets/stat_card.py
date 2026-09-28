"""Compact statistic card (title + large monospace value)."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class StatCard(QFrame):
    """Card showing one live number. ``accent`` selects the top border colour."""

    def __init__(self, title: str, accent: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setProperty("accent", accent)
        self._title = QLabel(title.upper())
        self._title.setObjectName("StatTitle")
        self._value = QLabel("0")
        self._value.setObjectName("StatValue")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, value: object) -> None:
        """Display ``value``."""
        self._value.setText(str(value))

    def value_text(self) -> str:
        """Currently displayed text."""
        return self._value.text()

    def set_hot(self, hot: bool) -> None:
        """Highlight the value (used for violations)."""
        if self.property("hot") == hot:
            return
        self.setProperty("hot", hot)
        self.style().unpolish(self)
        self.style().polish(self)
