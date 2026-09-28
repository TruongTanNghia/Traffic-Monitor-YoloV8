"""Reports page: session summary, class breakdown, exports and output folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.ui_helpers import key_label, key_value_label, make_panel, page_header, show_info, show_message
from app.gui.widgets.stat_card import StatCard
from app.reports.report_generator import ReportGenerator
from app.utils.constants import REPORTS_SUBDIR, VEHICLE_CLASSES
from app.utils.file_utils import ensure_dir, open_in_file_manager, resolve_path, to_project_relative

METRICS: tuple[tuple[str, str, str], ...] = (
    ("total_vehicles", "Total vehicles", "total"),
    ("total_in", "IN", "in"),
    ("total_out", "OUT", "out"),
    ("car", "Car", ""),
    ("motorcycle", "Motorcycle", ""),
    ("bus", "Bus", ""),
    ("truck", "Truck", ""),
    ("speed_violations", "Speed violations", "violation"),
    ("maximum_speed", "Maximum speed", ""),
)
SESSION_FIELDS = ("session_id", "status", "source", "started_at", "duration_s", "frames_processed", "device")


class ReportPanel(QWidget):
    """Show real session numbers and export them."""

    def __init__(self, output_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir = output_dir
        self._summary: dict[str, Any] | None = None
        self._generator = ReportGenerator()
        self.cards = {key: StatCard(title, accent) for key, title, accent in METRICS}
        self.session_values = {key: key_value_label() for key in SESSION_FIELDS}
        self.output_values = {key: key_value_label() for key in ("video", "json", "csv", "snapshots")}
        self.subtitle = QLabel("Chưa có phiên xử lý")
        self.subtitle.setObjectName("PageSubtitle")
        self.export_json_button = QPushButton("Export JSON")
        self.export_csv_button = QPushButton("Export CSV")
        self.open_folder_button = QPushButton("Open Output Folder")
        self.class_table = self._create_class_table()
        self._build()
        self.export_json_button.clicked.connect(self.export_json)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.open_folder_button.clicked.connect(self.open_output_folder)
        self._update_buttons()

    def _create_class_table(self) -> QTableWidget:
        table = QTableWidget(len(VEHICLE_CLASSES), 3)
        table.setHorizontalHeaderLabels(["IN", "OUT", "TOTAL"])
        table.setVerticalHeaderLabels([name.capitalize() for name in VEHICLE_CLASSES])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setDefaultSectionSize(28)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setFixedHeight(32 + 28 * len(VEHICLE_CLASSES) + 4)
        return table

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(page_header("Reports", "", [self.subtitle, self.export_json_button,
                                                     self.export_csv_button, self.open_folder_button]))
        grid = QGridLayout()
        grid.setSpacing(8)
        for index, (key, _, _) in enumerate(METRICS):
            grid.addWidget(self.cards[key], index // 5, index % 5)
        layout.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(12)
        session_panel, session_body = make_panel("Session")
        session_body.addLayout(self._kv_grid(self.session_values))
        classes_panel, classes_body = make_panel("Counts by class")
        classes_body.addWidget(self.class_table)
        row.addWidget(session_panel, 1)
        row.addWidget(classes_panel, 1)
        layout.addLayout(row)

        outputs_panel, outputs_body = make_panel("Output files")
        outputs_body.addLayout(self._kv_grid(self.output_values))
        layout.addWidget(outputs_panel)
        layout.addStretch(1)

    @staticmethod
    def _kv_grid(values: dict[str, QLabel]) -> QGridLayout:
        grid = QGridLayout()
        for row, (key, label) in enumerate(values.items()):
            grid.addWidget(key_label(key), row, 0)
            grid.addWidget(label, row, 1)
        grid.setColumnStretch(1, 1)
        return grid

    # --------------------------------------------------------------- update
    def set_output_dir(self, output_dir: Path) -> None:
        """Folder opened by *Open Output Folder* and used by exports."""
        self._output_dir = output_dir

    def update_live(self, stats: dict[str, Any]) -> None:
        """Live numbers while a session is running."""
        totals = stats.get("class_totals", {})
        for key, card in self.cards.items():
            if key == "maximum_speed":
                maximum = stats.get("maximum_speed")
                card.set_value(f"{maximum['speed_kmh']:.1f} km/h" if maximum else "—")
            else:
                card.set_value(stats.get(key, totals.get(key, 0)))
        self.cards["speed_violations"].set_hot(int(stats.get("speed_violations", 0)) > 0)
        for row, name in enumerate(VEHICLE_CLASSES):
            counts = stats.get("class_counts", {}).get(name, {"in": 0, "out": 0})
            values = (counts["in"], counts["out"], counts["in"] + counts["out"])
            for column, value in enumerate(values):
                self.class_table.setItem(row, column, QTableWidgetItem(str(value)))

    def session_started(self, info: dict[str, Any]) -> None:
        """Reset for a new session."""
        self._summary = None
        self.update_live({})
        self.subtitle.setText(f"SESSION {info.get('session_id', '')} · RUNNING")
        self.session_values["session_id"].setText(str(info.get("session_id", "—")))
        self.session_values["status"].setText("running")
        self.session_values["source"].setText(str(info.get("source", {}).get("description", "—")))
        self.session_values["device"].setText(str(info.get("device", "—")))
        for key in ("started_at", "duration_s", "frames_processed"):
            self.session_values[key].setText("—")
        for label in self.output_values.values():
            label.setText("—")
        self._update_buttons()

    def set_summary(self, summary: dict[str, Any]) -> None:
        """Show the final result of a session."""
        self._summary = summary
        if summary.get("output_dir"):
            self._output_dir = Path(summary["output_dir"])
        if summary.get("statistics"):
            self.update_live(summary["statistics"])
        report = summary.get("report") or {}
        session = report.get("session", {})
        self.subtitle.setText(f"SESSION {summary.get('session_id', '')} · {str(summary.get('status', '')).upper()}")
        values = {
            "session_id": summary.get("session_id"),
            "status": summary.get("status"),
            "source": report.get("source", {}).get("description", self.session_values["source"].text()),
            "started_at": session.get("started_at"),
            "duration_s": f"{session['duration_s']:.1f} s" if session.get("duration_s") is not None else None,
            "frames_processed": summary.get("frames_processed"),
            "device": session.get("device", self.session_values["device"].text()),
        }
        for key, value in values.items():
            self.session_values[key].setText("—" if value in (None, "") else str(value))
        self._show_outputs(summary.get("outputs", {}))
        self._update_buttons()

    def _show_outputs(self, outputs: dict[str, Any]) -> None:
        for key in ("video", "json", "csv"):
            self.output_values[key].setText(outputs.get(key) or "— (không lưu)")
        snapshots = outputs.get("snapshots") or []
        self.output_values["snapshots"].setText(f"{len(snapshots)} ảnh vi phạm" if snapshots else "—")

    def _update_buttons(self) -> None:
        has_report = bool(self._summary and self._summary.get("report"))
        self.export_json_button.setEnabled(has_report)
        self.export_csv_button.setEnabled(has_report)

    # -------------------------------------------------------------- actions
    def _report(self) -> dict[str, Any] | None:
        report = (self._summary or {}).get("report")
        if not report:
            show_message(self, "Chưa có dữ liệu", "Chưa có phiên xử lý nào hoàn tất. Hãy START ở Dashboard.", False)
        return report

    def export_json(self) -> Path | None:
        """Write the JSON report of the last session to output/reports."""
        report = self._report()
        if report is None:
            return None
        try:
            path = self._generator.save_json(report, self._output_dir)
        except OSError as exc:
            show_message(self, "Không xuất được JSON", str(exc))
            return None
        self.output_values["json"].setText(to_project_relative(path))
        show_info(self, "Export JSON", f"Đã lưu báo cáo:\n{path}")
        return path

    def export_csv(self) -> Path | None:
        """Write the CSV event list of the last session to output/reports."""
        report = self._report()
        if report is None:
            return None
        try:
            path = self._generator.save_csv(report["events"], self._output_dir, report["session"]["session_id"])
        except OSError as exc:
            show_message(self, "Không xuất được CSV", str(exc))
            return None
        self.output_values["csv"].setText(to_project_relative(path))
        show_info(self, "Export CSV", f"Đã lưu {len(report['events'])} event:\n{path}")
        return path

    def open_output_folder(self) -> None:
        """Open the output directory in the file manager."""
        folder = resolve_path(self._output_dir)
        try:
            ensure_dir(folder / REPORTS_SUBDIR)
            open_in_file_manager(folder)
        except OSError as exc:
            show_message(self, "Không mở được thư mục", str(exc))
