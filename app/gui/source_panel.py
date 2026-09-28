"""Video Source page: choose file / webcam / RTSP, test it and apply it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppConfig
from app.gui.background import TaskRunner
from app.gui.ui_helpers import key_label, key_value_label, make_panel, page_header, set_invalid, show_message
from app.utils.constants import (
    INPUT_DIR,
    SAMPLES_DIR,
    SOURCE_CAMERA,
    SOURCE_FILE,
    SOURCE_RTSP,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from app.utils.file_utils import resolve_path, to_project_relative
from app.video.video_source import SourceInfo, mask_credentials, probe_source

SOURCE_LABELS = {
    SOURCE_FILE: ("Local video", "FileVideoSource"),
    SOURCE_CAMERA: ("Webcam", "CameraVideoSource"),
    SOURCE_RTSP: ("RTSP stream", "RTSPVideoSource"),
}
VIDEO_FILTER = "Video ({})".format(" ".join(f"*{ext}" for ext in SUPPORTED_VIDEO_EXTENSIONS))


def browse_video(parent: QWidget, current: str) -> str | None:
    """Open a file dialog and return a project-relative path, or ``None``."""
    start = resolve_path(current).parent if current else INPUT_DIR
    if not start.is_dir():
        start = SAMPLES_DIR if SAMPLES_DIR.is_dir() else INPUT_DIR
    path, _ = QFileDialog.getOpenFileName(parent, "Chọn video", str(start), VIDEO_FILTER)
    return to_project_relative(Path(path)) if path else None


class SourcePanel(QWidget):
    """Select and validate the video input."""

    source_applied = Signal(str, str, int)  # source_type, source_uri, rtsp_timeout_ms

    def __init__(self, config: AppConfig, runner: TaskRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._runner = runner
        self._type_group = QButtonGroup(self)
        self._panes = QStackedWidget()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("data/input/video.mp4")
        self.browse_button = QPushButton("Browse…")
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.rtsp_edit = QLineEdit()
        self.rtsp_edit.setPlaceholderText("rtsp://user:pass@192.168.1.10:554/stream1")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1000, 60000)
        self.timeout_spin.setSingleStep(500)
        self.timeout_spin.setSuffix(" ms")
        self.test_button = QPushButton("TEST SOURCE")
        self.apply_button = QPushButton("APPLY SOURCE")
        self.apply_button.setObjectName("PrimaryButton")
        self.probe_status = QLabel("Chưa kiểm tra")
        self.probe_status.setObjectName("Hint")
        self.info_values = {key: key_value_label() for key in ("type", "source", "resolution", "fps", "frames", "duration")}
        self.current_label = QLabel()
        self.current_label.setObjectName("PageSubtitle")
        self._radios: dict[str, QRadioButton] = {}
        self._build()
        self._connect()
        self.set_config(config)

    # ----------------------------------------------------------------- build
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(page_header("Video Source", "FILE · WEBCAM · RTSP — VideoSource interface", [self.current_label]))

        type_row = QHBoxLayout()
        for index, (key, (title, cls)) in enumerate(SOURCE_LABELS.items()):
            radio = QRadioButton(f"{title}   ({cls})")
            self._type_group.addButton(radio, index)
            self._radios[key] = radio
            type_row.addWidget(radio)
        type_row.addStretch(1)
        layout.addLayout(type_row)

        self._panes.addWidget(self._file_pane())
        self._panes.addWidget(self._camera_pane())
        self._panes.addWidget(self._rtsp_pane())
        settings_panel, settings_body = make_panel("Source settings")
        settings_body.addWidget(self._panes)
        buttons = QHBoxLayout()
        buttons.addWidget(self.probe_status, 1)
        buttons.addWidget(self.test_button)
        buttons.addWidget(self.apply_button)
        settings_body.addLayout(buttons)
        layout.addWidget(settings_panel)

        info_panel, info_body = make_panel("Source information")
        grid = QGridLayout()
        labels = {"type": "Type", "source": "Source", "resolution": "Resolution", "fps": "FPS",
                  "frames": "Frames", "duration": "Duration"}
        for row, (key, text) in enumerate(labels.items()):
            grid.addWidget(key_label(text), row, 0)
            grid.addWidget(self.info_values[key], row, 1)
        grid.setColumnStretch(1, 1)
        info_body.addLayout(grid)
        layout.addWidget(info_panel)
        hint = QLabel("GUI không mở cv2.VideoCapture trực tiếp: TEST SOURCE chạy VideoSource.probe trong thread nền. "
                      "APPLY SOURCE ghi source_type/source_uri vào config.yaml và dùng cho lần START tiếp theo.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

    def _file_pane(self) -> QWidget:
        pane = QWidget()
        row = QGridLayout(pane)
        row.addWidget(key_label("Video file"), 0, 0)
        row.addWidget(self.file_edit, 0, 1)
        row.addWidget(self.browse_button, 0, 2)
        row.addWidget(key_label("Hỗ trợ: " + ", ".join(SUPPORTED_VIDEO_EXTENSIONS)), 1, 1)
        row.setColumnStretch(1, 1)
        return pane

    def _camera_pane(self) -> QWidget:
        pane = QWidget()
        row = QGridLayout(pane)
        row.addWidget(key_label("Camera index"), 0, 0)
        row.addWidget(self.camera_spin, 0, 1)
        row.addWidget(key_label("Backend: CAP_DSHOW trên Windows"), 1, 1)
        row.setColumnStretch(2, 1)
        return pane

    def _rtsp_pane(self) -> QWidget:
        pane = QWidget()
        row = QGridLayout(pane)
        row.addWidget(key_label("RTSP URL"), 0, 0)
        row.addWidget(self.rtsp_edit, 0, 1)
        row.addWidget(key_label("Timeout"), 1, 0)
        row.addWidget(self.timeout_spin, 1, 1)
        row.addWidget(key_label("Mất kết nối: thử lại 3 lần (1s, 2s, 4s)"), 2, 1)
        row.setColumnStretch(1, 1)
        return pane

    def _connect(self) -> None:
        self._type_group.idToggled.connect(self._on_type_toggled)
        self.browse_button.clicked.connect(self._browse)
        for editor in (self.file_edit, self.rtsp_edit):
            editor.textEdited.connect(lambda _text: self.reset_probe())
        self.camera_spin.valueChanged.connect(lambda _value: self.reset_probe())
        self.test_button.clicked.connect(self._test_source)
        self.apply_button.clicked.connect(self._apply)

    # --------------------------------------------------------------- state
    def set_config(self, config: AppConfig) -> None:
        """Load the applied source into the form."""
        self._config = config
        self._radios.get(config.source_type, self._radios[SOURCE_FILE]).setChecked(True)
        self._panes.setCurrentIndex(list(SOURCE_LABELS).index(config.source_type)
                                    if config.source_type in SOURCE_LABELS else 0)
        if config.source_type == SOURCE_FILE:
            self.file_edit.setText(config.source_uri)
        elif config.source_type == SOURCE_CAMERA:
            self.camera_spin.setValue(int(config.source_uri or 0) if str(config.source_uri).isdigit() else 0)
        else:
            self.rtsp_edit.setText(config.source_uri)
        self.timeout_spin.setValue(config.rtsp_timeout_ms)
        self.current_label.setText(f"CURRENT: {config.source_type.upper()} · {describe_uri(config)}")

    def set_locked(self, locked: bool) -> None:
        """Disable editing while a session runs."""
        for widget in (self.apply_button, self.test_button, self.browse_button, self.file_edit,
                       self.camera_spin, self.rtsp_edit, self.timeout_spin, *self._radios.values()):
            widget.setEnabled(not locked)

    def selected_type(self) -> str:
        """Source type currently selected in the form."""
        return list(SOURCE_LABELS)[max(0, self._type_group.checkedId())]

    def form_uri(self) -> str:
        """URI currently typed in the form."""
        source_type = self.selected_type()
        if source_type == SOURCE_FILE:
            return self.file_edit.text().strip()
        if source_type == SOURCE_CAMERA:
            return str(self.camera_spin.value())
        return self.rtsp_edit.text().strip()

    def _form_config(self) -> AppConfig:
        config = self._config.copy()
        config.source_type = self.selected_type()
        config.source_uri = self.form_uri()
        config.rtsp_timeout_ms = self.timeout_spin.value()
        return config

    # ------------------------------------------------------------- actions
    def _on_type_toggled(self, button_id: int, checked: bool) -> None:
        if checked:
            self._panes.setCurrentIndex(button_id)
            self.reset_probe()

    def reset_probe(self) -> None:
        """Forget the previous probe result (it belonged to another source)."""
        self.probe_status.setText("Chưa kiểm tra")
        for label in self.info_values.values():
            label.setText("—")

    def _browse(self) -> None:
        path = browse_video(self, self.file_edit.text())
        if path:
            self.file_edit.setText(path)
            set_invalid(self.file_edit, False)
            self.reset_probe()

    def _validate_form(self) -> str | None:
        """Return an error message or ``None``."""
        source_type, uri = self.selected_type(), self.form_uri()
        if source_type == SOURCE_FILE:
            if not uri:
                return "Chưa chọn file video."
            path = resolve_path(uri)
            if not path.is_file():
                return f"Không tìm thấy file video: {path}"
            if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                return f"Định dạng {path.suffix} không được hỗ trợ."
        if source_type == SOURCE_RTSP and not uri.lower().startswith(("rtsp://", "rtsps://")):
            return "RTSP URL phải có dạng rtsp://host:port/path"
        return None

    def _test_source(self) -> None:
        error = self._validate_form()
        if error:
            self._mark_invalid(True)
            show_message(self, "Nguồn video không hợp lệ", error, critical=False)
            return
        self._mark_invalid(False)
        config = self._form_config()
        self.test_button.setEnabled(False)
        self.probe_status.setText("Đang kiểm tra nguồn…")
        self._runner.start(lambda: probe_source(config), self._on_probe_ok, self._on_probe_failed)

    def _on_probe_ok(self, info: SourceInfo) -> None:
        self.test_button.setEnabled(True)
        self.probe_status.setText("✓ Kết nối thành công")
        self.show_info(info.to_dict())

    def _on_probe_failed(self, title: str, message: str, critical: bool) -> None:
        self.test_button.setEnabled(True)
        self.probe_status.setText("✕ " + title)
        show_message(self, title, message, critical)

    def _apply(self) -> None:
        error = self._validate_form()
        if error:
            self._mark_invalid(True)
            show_message(self, "Nguồn video không hợp lệ", error, critical=False)
            return
        self._mark_invalid(False)
        self.source_applied.emit(self.selected_type(), self.form_uri(), self.timeout_spin.value())

    def _mark_invalid(self, invalid: bool) -> None:
        editor = {SOURCE_FILE: self.file_edit, SOURCE_RTSP: self.rtsp_edit}.get(self.selected_type())
        if editor is not None:
            set_invalid(editor, invalid)

    def show_info(self, info: dict[str, Any]) -> None:
        """Display source metadata (from a probe or a started session)."""
        width, height = info.get("resolution", [0, 0])
        fps = float(info.get("fps", 0) or 0)
        frames = info.get("frame_count")
        self.info_values["type"].setText(str(info.get("type", "—")).upper())
        uri = str(info.get("uri", "—"))
        if info.get("type") == SOURCE_FILE:
            uri = to_project_relative(Path(uri))
        self.info_values["source"].setText(mask_credentials(uri))
        self.info_values["resolution"].setText(f"{width} × {height}")
        self.info_values["fps"].setText(f"{fps:.2f}")
        self.info_values["frames"].setText(str(frames) if frames else "live")
        self.info_values["duration"].setText(f"{frames / fps:.1f} s" if frames and fps else "—")


def describe_uri(config: AppConfig) -> str:
    """Short display name of the configured source."""
    if config.source_type == SOURCE_FILE:
        return Path(config.source_uri).name or "—"
    if config.source_type == SOURCE_CAMERA:
        return f"Camera {config.source_uri or 0}"
    return mask_credentials(config.source_uri) or "—"
