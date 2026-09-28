"""Configuration page: edit, validate, save and reset AppConfig."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config.default_config import RANGES
from app.config.settings import AppConfig
from app.core.detector import describe_device, resolve_device
from app.gui.background import TaskRunner
from app.gui.calibration_dialog import CalibrationDialog
from app.gui.source_panel import browse_video
from app.gui.ui_helpers import confirm, key_label, page_header, set_invalid, show_message
from app.utils.constants import CONFIG_FILE, DEVICE_CHOICES, MODELS_DIR, SOURCE_FILE
from app.utils.exceptions import PerspectiveError
from app.utils.file_utils import resolve_path, to_project_relative
from app.video.video_source import read_first_frame
from app.vision.geometry import validate_quadrilateral

COORD_LIMIT = 20000


def _double_spin(key: str, step: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    low, high = RANGES[key]
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(suffix)
    return spin


class SettingsPanel(QWidget):
    """Edit every configurable parameter. ``config_saved`` carries the new AppConfig."""

    config_saved = Signal(object)

    def __init__(self, config: AppConfig, runner: TaskRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._runner = runner
        self._loading = False
        self._create_widgets()
        self._build()
        self._connect()
        self.set_config(config)

    # ------------------------------------------------------------ widgets
    def _create_widgets(self) -> None:
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_browse = QPushButton("Browse…")
        self.device_combo = QComboBox()
        self.device_combo.addItems(list(DEVICE_CHOICES))
        self.device_hint = QLabel(f"auto → {describe_device(resolve_device('auto'))}")
        self.device_hint.setObjectName("Hint")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_spin = _double_spin("confidence_threshold", 0.01, 2)
        self.limit_spin = _double_spin("speed_limit_kmh", 5, 0, " km/h")
        self.line_slider = QSlider(Qt.Orientation.Horizontal)
        self.line_slider.setRange(10, 90)
        self.line_spin = _double_spin("line_y_ratio", 0.01, 2)
        self.width_spin = _double_spin("road_width_m", 0.5, 1, " m")
        self.length_spin = _double_spin("road_length_m", 1, 1, " m")
        self.min_track_spin = _double_spin("min_track_seconds", 0.1, 1, " s")
        self.point_spins = [[self._coord_spin(), self._coord_spin()] for _ in range(4)]
        self.ref_w_spin, self.ref_h_spin = self._coord_spin(1), self._coord_spin(1)
        self.calibrate_button = QPushButton("Calibrate from video…")
        self.video_edit = QLineEdit()
        self.video_edit.setReadOnly(True)
        self.video_browse = QPushButton("Browse…")
        self.output_edit = QLineEdit()
        self.output_browse = QPushButton("Browse…")
        self.save_video_check = QCheckBox("save_output_video — lưu video đã annotation")
        self.save_report_check = QCheckBox("save_report — tự lưu JSON + CSV khi kết thúc")
        self.save_snapshots_check = QCheckBox("save_snapshots — lưu ảnh mỗi vi phạm")
        self.preview_check = QCheckBox("show_preview — hiển thị video trên GUI")
        self.dirty_label = QLabel("")
        self.dirty_label.setObjectName("DirtyHint")
        self.reset_button = QPushButton("RESET DEFAULT")
        self.save_button = QPushButton("SAVE CONFIG")
        self.save_button.setObjectName("PrimaryButton")
        self._field_widgets: dict[str, list[QWidget]] = {
            "model_path": [self.model_combo],
            "device": [self.device_combo],
            "confidence_threshold": [self.conf_spin],
            "speed_limit_kmh": [self.limit_spin],
            "line_y_ratio": [self.line_spin],
            "road_width_m": [self.width_spin],
            "road_length_m": [self.length_spin],
            "min_track_seconds": [self.min_track_spin],
            "perspective_source": [spin for pair in self.point_spins for spin in pair],
            "perspective_reference_size": [self.ref_w_spin, self.ref_h_spin],
            "output_dir": [self.output_edit],
            "source_uri": [self.video_edit],
        }

    @staticmethod
    def _coord_spin(minimum: int = -COORD_LIMIT) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, COORD_LIMIT)
        return spin

    def _build(self) -> None:
        content = QWidget()
        columns = QHBoxLayout(content)
        columns.setSpacing(14)
        left, right = QVBoxLayout(), QVBoxLayout()
        left.addWidget(self._model_group())
        left.addWidget(self._counting_group())
        left.addStretch(1)
        right.addWidget(self._calibration_group())
        right.addWidget(self._io_group())
        right.addStretch(1)
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.output_edit.setMinimumWidth(120)
        self.video_edit.setMinimumWidth(120)
        self.model_combo.setMinimumContentsLength(12)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(page_header("Configuration", f"AppConfig → {CONFIG_FILE.name}",
                                     [self.dirty_label, self.reset_button, self.save_button]))
        layout.addWidget(scroll, 1)

    def _model_group(self) -> QGroupBox:
        group = QGroupBox("MODEL && DETECTION")
        grid = QGridLayout(group)
        grid.addWidget(key_label("Model"), 0, 0)
        grid.addWidget(self.model_combo, 0, 1)
        grid.addWidget(self.model_browse, 0, 2)
        grid.addWidget(key_label("Device"), 1, 0)
        grid.addWidget(self.device_combo, 1, 1)
        grid.addWidget(self.device_hint, 2, 1, 1, 2)
        grid.addWidget(key_label("Confidence"), 3, 0)
        grid.addWidget(self.conf_slider, 3, 1)
        grid.addWidget(self.conf_spin, 3, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _counting_group(self) -> QGroupBox:
        group = QGroupBox("COUNTING && SPEED")
        grid = QGridLayout(group)
        grid.addWidget(key_label("Speed limit"), 0, 0)
        grid.addWidget(self.limit_spin, 0, 1, 1, 2)
        grid.addWidget(key_label("Line position (y)"), 1, 0)
        grid.addWidget(self.line_slider, 1, 1)
        grid.addWidget(self.line_spin, 1, 2)
        grid.addWidget(key_label("Min track time"), 2, 0)
        grid.addWidget(self.min_track_spin, 2, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _calibration_group(self) -> QGroupBox:
        group = QGroupBox("SPEED CALIBRATION (PERSPECTIVE)")
        grid = QGridLayout(group)
        grid.addWidget(key_label("Road width"), 0, 0)
        grid.addWidget(self.width_spin, 0, 1, 1, 2)
        grid.addWidget(key_label("Road length"), 1, 0)
        grid.addWidget(self.length_spin, 1, 1, 1, 2)
        for index, (name, (x_spin, y_spin)) in enumerate(zip("ABCD", self.point_spins)):
            grid.addWidget(key_label(f"Point {name} (x, y)"), 2 + index, 0)
            grid.addWidget(x_spin, 2 + index, 1)
            grid.addWidget(y_spin, 2 + index, 2)
        grid.addWidget(key_label("Reference size"), 6, 0)
        grid.addWidget(self.ref_w_spin, 6, 1)
        grid.addWidget(self.ref_h_spin, 6, 2)
        grid.addWidget(self.calibrate_button, 7, 1, 1, 2)
        hint = QLabel("A trên-trái, B trên-phải, C dưới-phải, D dưới-trái của đoạn đường W × L mét. "
                      "Toạ độ theo reference size; tự scale nếu video có độ phân giải khác.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        grid.addWidget(hint, 8, 0, 1, 3)
        return group

    def _io_group(self) -> QGroupBox:
        group = QGroupBox("INPUT && OUTPUT")
        grid = QGridLayout(group)
        grid.addWidget(key_label("Video"), 0, 0)
        grid.addWidget(self.video_edit, 0, 1)
        grid.addWidget(self.video_browse, 0, 2)
        grid.addWidget(key_label("Output directory"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(self.output_browse, 1, 2)
        for row, check in enumerate((self.save_video_check, self.save_report_check,
                                     self.save_snapshots_check, self.preview_check), start=2):
            grid.addWidget(check, row, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        return group

    def _connect(self) -> None:
        self.conf_slider.valueChanged.connect(lambda v: self._sync(self.conf_spin, v / 100))
        self.conf_spin.valueChanged.connect(lambda v: self._sync(self.conf_slider, round(v * 100)))
        self.line_slider.valueChanged.connect(lambda v: self._sync(self.line_spin, v / 100))
        self.line_spin.valueChanged.connect(lambda v: self._sync(self.line_slider, round(v * 100)))
        editors = [self.conf_spin, self.limit_spin, self.line_spin, self.width_spin, self.length_spin,
                   self.min_track_spin, self.ref_w_spin, self.ref_h_spin,
                   *[spin for pair in self.point_spins for spin in pair]]
        for spin in editors:
            spin.valueChanged.connect(self._mark_dirty)
        self.model_combo.currentTextChanged.connect(self._mark_dirty)
        self.device_combo.currentTextChanged.connect(self._mark_dirty)
        self.output_edit.textChanged.connect(self._mark_dirty)
        self.video_edit.textChanged.connect(self._mark_dirty)
        for check in (self.save_video_check, self.save_report_check, self.save_snapshots_check, self.preview_check):
            check.toggled.connect(self._mark_dirty)
        self.model_browse.clicked.connect(self._browse_model)
        self.video_browse.clicked.connect(self._browse_video)
        self.output_browse.clicked.connect(self._browse_output)
        self.calibrate_button.clicked.connect(self._calibrate)
        self.save_button.clicked.connect(self._save)
        self.reset_button.clicked.connect(self._reset_default)

    @staticmethod
    def _sync(widget: QWidget, value: float) -> None:
        if abs(widget.value() - value) > 1e-9:  # type: ignore[attr-defined]
            widget.setValue(value)  # type: ignore[attr-defined]

    # --------------------------------------------------------------- state
    def set_config(self, config: AppConfig) -> None:
        """Fill the form with ``config`` (the applied configuration)."""
        self._config = config
        self._loading = True
        try:
            self._fill(config)
        finally:
            self._loading = False
        self._clear_invalid()
        self._set_dirty(False)

    def _fill(self, config: AppConfig) -> None:
        self._populate_models(config.model_path)
        self.device_combo.setCurrentText(config.device)
        self.conf_spin.setValue(config.confidence_threshold)
        self.limit_spin.setValue(config.speed_limit_kmh)
        self.line_spin.setValue(config.line_y_ratio)
        self.width_spin.setValue(config.road_width_m)
        self.length_spin.setValue(config.road_length_m)
        self.min_track_spin.setValue(config.min_track_seconds)
        for (x_spin, y_spin), point in zip(self.point_spins, config.perspective_source):
            x_spin.setValue(int(point[0]))
            y_spin.setValue(int(point[1]))
        self.ref_w_spin.setValue(int(config.perspective_reference_size[0]))
        self.ref_h_spin.setValue(int(config.perspective_reference_size[1]))
        self.video_edit.setText(config.source_uri if config.source_type == SOURCE_FILE else "")
        self.video_edit.setPlaceholderText("" if config.source_type == SOURCE_FILE
                                           else f"(nguồn hiện tại: {config.source_type})")
        self.output_edit.setText(config.output_dir)
        self.save_video_check.setChecked(config.save_output_video)
        self.save_report_check.setChecked(config.save_report)
        self.save_snapshots_check.setChecked(config.save_snapshots)
        self.preview_check.setChecked(config.show_preview)

    def _populate_models(self, current: str) -> None:
        models = sorted(to_project_relative(p) for p in MODELS_DIR.glob("*.pt"))
        if current and current not in models:
            models.append(current)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(current)

    def collect(self) -> AppConfig:
        """Build an AppConfig from the form (not yet validated)."""
        config = self._config.copy()
        config.model_path = self.model_combo.currentText().strip()
        config.device = self.device_combo.currentText()
        config.confidence_threshold = round(self.conf_spin.value(), 2)
        config.speed_limit_kmh = self.limit_spin.value()
        config.line_y_ratio = round(self.line_spin.value(), 2)
        config.road_width_m = self.width_spin.value()
        config.road_length_m = self.length_spin.value()
        config.min_track_seconds = round(self.min_track_spin.value(), 2)
        config.perspective_source = [[x.value(), y.value()] for x, y in self.point_spins]
        config.perspective_reference_size = [self.ref_w_spin.value(), self.ref_h_spin.value()]
        video = self.video_edit.text().strip()
        if video and (config.source_type != SOURCE_FILE or video != config.source_uri):
            config.source_type, config.source_uri = SOURCE_FILE, video
        config.output_dir = self.output_edit.text().strip()
        config.save_output_video = self.save_video_check.isChecked()
        config.save_report = self.save_report_check.isChecked()
        config.save_snapshots = self.save_snapshots_check.isChecked()
        config.show_preview = self.preview_check.isChecked()
        return config

    def set_locked(self, locked: bool) -> None:
        """Disable editing while a session runs (config applies on next START)."""
        for widgets in self._field_widgets.values():
            for widget in widgets:
                widget.setEnabled(not locked)
        for widget in (self.conf_slider, self.line_slider, self.model_browse, self.video_browse,
                       self.output_browse, self.calibrate_button, self.save_button, self.reset_button,
                       self.save_video_check, self.save_report_check, self.save_snapshots_check,
                       self.preview_check):
            widget.setEnabled(not locked)

    @property
    def is_dirty(self) -> bool:
        """Whether the form has unsaved changes."""
        return bool(self.dirty_label.text())

    def _mark_dirty(self, *_: object) -> None:
        if not self._loading:
            self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self.dirty_label.setText("● Có thay đổi chưa lưu" if dirty else "")

    # ----------------------------------------------------------- validation
    def validation_errors(self, config: AppConfig) -> list[tuple[str, str]]:
        """Field errors of ``config`` including model/video existence and geometry."""
        errors = config.collect_errors()
        fields_with_error = {name for name, _ in errors}
        if "model_path" not in fields_with_error and not config.model_file.is_file():
            errors.append(("model_path", f"không tìm thấy file {config.model_file}"))
        if "perspective_source" not in fields_with_error:
            try:
                validate_quadrilateral(config.perspective_source)
            except PerspectiveError as exc:
                errors.append(("perspective_source", str(exc)))
        if config.source_type == SOURCE_FILE and config.source_uri and not resolve_path(config.source_uri).is_file():
            errors.append(("source_uri", f"không tìm thấy video {config.source_uri}"))
        return errors

    def _clear_invalid(self) -> None:
        for widgets in self._field_widgets.values():
            for widget in widgets:
                set_invalid(widget, False)

    # -------------------------------------------------------------- actions
    def _save(self) -> None:
        config = self.collect()
        self._clear_invalid()
        errors = self.validation_errors(config)
        if errors:
            for name, _ in errors:
                for widget in self._field_widgets.get(name, []):
                    set_invalid(widget, True)
            message = "\n".join(f"• {name}: {text}" for name, text in errors)
            show_message(self, "Cấu hình không hợp lệ", message + "\n\nCấu hình cũ được giữ nguyên.", critical=False)
            return
        try:
            config.save(CONFIG_FILE)
        except OSError as exc:
            show_message(self, "Không lưu được cấu hình", str(exc))
            return
        self._config = config
        self._set_dirty(False)
        self.config_saved.emit(config)

    def _reset_default(self) -> None:
        if not confirm(self, "Reset default",
                       "Khôi phục toàn bộ cấu hình mặc định và lưu vào config.yaml?\n"
                       "(Nguồn video hiện tại được giữ nguyên.)"):
            return
        defaults = AppConfig()
        defaults.source_type, defaults.source_uri = self._config.source_type, self._config.source_uri
        defaults.rtsp_timeout_ms = self._config.rtsp_timeout_ms
        try:
            defaults.save(CONFIG_FILE)
        except OSError as exc:
            show_message(self, "Không lưu được cấu hình", str(exc))
            return
        self.set_config(defaults)
        self.config_saved.emit(defaults)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn model YOLO", str(MODELS_DIR), "PyTorch model (*.pt)")
        if path:
            self.model_combo.setCurrentText(to_project_relative(Path(path)))

    def _browse_video(self) -> None:
        path = browse_video(self, self.video_edit.text())
        if path:
            self.video_edit.setText(path)

    def _browse_output(self) -> None:
        start = resolve_path(self.output_edit.text() or "data/output")
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục output", str(start))
        if path:
            self.output_edit.setText(to_project_relative(Path(path)))

    def _calibrate(self) -> None:
        config = self.collect()
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.setText("Đang đọc frame…")
        self._runner.start(lambda: read_first_frame(config), self._open_calibration, self._calibration_failed)

    def _calibration_failed(self, title: str, message: str, critical: bool) -> None:
        self._restore_calibrate_button()
        show_message(self, title, message, critical)

    def _restore_calibrate_button(self) -> None:
        self.calibrate_button.setEnabled(True)
        self.calibrate_button.setText("Calibrate from video…")

    def _open_calibration(self, frame: np.ndarray) -> None:
        self._restore_calibrate_button()
        current = self.collect()
        dialog = CalibrationDialog(frame, current.perspective_source, current.perspective_reference_size, self)
        if dialog.exec() != CalibrationDialog.DialogCode.Accepted:
            return
        for (x_spin, y_spin), (x, y) in zip(self.point_spins, dialog.selected_points()):
            x_spin.setValue(x)
            y_spin.setValue(y)
        self.ref_w_spin.setValue(dialog.reference_size[0])
        self.ref_h_spin.setValue(dialog.reference_size[1])
        self._set_dirty(True)
