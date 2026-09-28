"""Realtime event table (model/view) with filtering by type and tracker id."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QWidget

from app.utils.constants import LOG_VIEW_MAX_ROWS, EventType
from app.utils.time_utils import iso_to_clock

Index = QModelIndex | QPersistentModelIndex

COLUMNS: tuple[tuple[str, str], ...] = (
    ("Time", "timestamp"),
    ("Frame", "frame_index"),
    ("Track ID", "tracker_id"),
    ("Vehicle", "vehicle_type"),
    ("Speed", "speed_kmh"),
    ("Limit", "speed_limit_kmh"),
    ("Event", "event_type"),
)
EVENT_COLORS = {
    EventType.SPEED_VIOLATION.value: QColor("#FFB3B3"),
    EventType.VEHICLE_IN.value: QColor("#A9E3C2"),
    EventType.VEHICLE_OUT.value: QColor("#B5D3F0"),
}
VIOLATION_BACKGROUND = QBrush(QColor("#241617"))
COMPACT_EVENT_NAMES = {
    EventType.SPEED_VIOLATION.value: "OVER",
    EventType.VEHICLE_IN.value: "IN",
    EventType.VEHICLE_OUT.value: "OUT",
}


class EventTableModel(QAbstractTableModel):
    """Holds event dictionaries (newest appended last)."""

    def __init__(self, max_rows: int = LOG_VIEW_MAX_ROWS, compact: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._events: list[dict[str, Any]] = []
        self._max_rows = max_rows
        self._compact = compact

    def rowCount(self, parent: Index = QModelIndex()) -> int:  # noqa: N802,B008 - Qt API
        return 0 if parent.isValid() else len(self._events)

    def columnCount(self, parent: Index = QModelIndex()) -> int:  # noqa: N802,B008
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section][0]
        return None

    def data(self, index: Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        event = self._events[index.row()]
        key = COLUMNS[index.column()][1]
        if role == Qt.ItemDataRole.DisplayRole:
            if self._compact:
                return self._format_compact(key, event.get(key))
            return self._format(key, event.get(key))
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._format(key, event.get(key))
        if role == Qt.ItemDataRole.ForegroundRole and key == "event_type":
            return EVENT_COLORS.get(event.get("event_type", ""))
        if role == Qt.ItemDataRole.BackgroundRole and event.get("event_type") == EventType.SPEED_VIOLATION.value:
            return VIOLATION_BACKGROUND
        if role == Qt.ItemDataRole.UserRole:
            return event
        return None

    @staticmethod
    def _format(key: str, value: Any) -> str:
        if value is None or value == "":
            return "—"
        if key == "timestamp":
            return iso_to_clock(str(value))
        if key == "tracker_id":
            return f"Vehicle #{value}"
        if key == "speed_kmh":
            return f"{float(value):.1f} km/h"
        if key == "speed_limit_kmh":
            return f"{float(value):.0f}"
        return str(value)

    @classmethod
    def _format_compact(cls, key: str, value: Any) -> str:
        """Short form for the narrow dashboard panel."""
        if value is None or value == "":
            return "—"
        if key == "tracker_id":
            return f"#{value}"
        if key == "speed_kmh":
            return f"{float(value):.0f}"
        if key == "event_type":
            return COMPACT_EVENT_NAMES.get(str(value), str(value))
        return cls._format(key, value)

    def append_event(self, event: dict[str, Any]) -> None:
        """Append one event, dropping the oldest when over capacity."""
        if len(self._events) >= self._max_rows:
            self.beginRemoveRows(QModelIndex(), 0, 0)
            self._events.pop(0)
            self.endRemoveRows()
        row = len(self._events)
        self.beginInsertRows(QModelIndex(), row, row)
        self._events.append(event)
        self.endInsertRows()

    def clear(self) -> None:
        """Remove every row."""
        self.beginResetModel()
        self._events.clear()
        self.endResetModel()

    def events(self) -> list[dict[str, Any]]:
        """Copy of all stored events."""
        return list(self._events)


class EventFilterProxy(QSortFilterProxyModel):
    """Filter rows by event type and exact tracker id."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._event_type = ""
        self._tracker_id: int | None = None

    def set_event_type(self, event_type: str) -> None:
        """Show only ``event_type`` (empty string = all)."""
        self._event_type = event_type
        self.invalidateFilter()

    def set_tracker_id(self, text: str) -> None:
        """Show only one tracker id (empty = all)."""
        digits = text.strip().lstrip("#")
        self._tracker_id = int(digits) if digits.isdigit() else None
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: Index) -> bool:  # noqa: N802
        model = self.sourceModel()
        event = model.data(model.index(source_row, 0, source_parent), Qt.ItemDataRole.UserRole) or {}
        if self._event_type and event.get("event_type") != self._event_type:
            return False
        if self._tracker_id is not None and event.get("tracker_id") != self._tracker_id:
            return False
        return True


class LogView(QTableView):
    """Table view of events with optional auto-scroll."""

    def __init__(self, max_rows: int = LOG_VIEW_MAX_ROWS, compact: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = EventTableModel(max_rows, compact, self)
        self._proxy = EventFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self.setModel(self._proxy)
        self.auto_scroll = True
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(24)
        self.setShowGrid(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        if compact:
            for column in (1, 5):  # frame, limit
                self.setColumnHidden(column, True)
            for column in (0, 2, 4):  # time, id, speed
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

    def add_event(self, event: dict[str, Any]) -> None:
        """Append an event and scroll to it when auto-scroll is on."""
        self._model.append_event(event)
        if self.auto_scroll:
            self.scrollToBottom()

    def clear(self) -> None:
        """Remove all rows."""
        self._model.clear()

    def set_event_type_filter(self, event_type: str) -> None:
        """Filter by event type (empty = all)."""
        self._proxy.set_event_type(event_type)

    def set_tracker_filter(self, text: str) -> None:
        """Filter by tracker id (empty = all)."""
        self._proxy.set_tracker_id(text)

    def total_count(self) -> int:
        """Rows stored (before filtering)."""
        return self._model.rowCount()

    def visible_count(self) -> int:
        """Rows visible after filtering."""
        return self._proxy.rowCount()

    def events(self) -> list[dict[str, Any]]:
        """All stored events."""
        return self._model.events()
