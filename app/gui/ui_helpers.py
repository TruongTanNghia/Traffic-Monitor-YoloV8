"""Small GUI building blocks shared by the pages."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


def bgr_to_qimage(frame: np.ndarray, max_width: int | None = None) -> QImage:
    """Convert an OpenCV BGR frame to a detached QImage (safe across threads).

    Args:
        frame: BGR image.
        max_width: downscale wider frames first (keeps preview cheap).
    """
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()


def page_header(title: str, subtitle: str, right: list[QWidget] | None = None) -> QHBoxLayout:
    """Title + subtitle on the left, optional widgets on the right."""
    layout = QHBoxLayout()
    texts = QVBoxLayout()
    texts.setSpacing(0)
    title_label = QLabel(title)
    title_label.setObjectName("PageTitle")
    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("PageSubtitle")
    texts.addWidget(title_label)
    texts.addWidget(subtitle_label)
    layout.addLayout(texts)
    layout.addStretch(1)
    for widget in right or []:
        layout.addWidget(widget)
    return layout


def make_panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    """Framed panel with an uppercase header. Returns the frame and its body layout."""
    frame = QFrame()
    frame.setObjectName("Panel")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    header = QLabel(title.upper())
    header.setObjectName("PanelHeader")
    outer.addWidget(header)
    body = QVBoxLayout()
    body.setContentsMargins(10, 8, 10, 10)
    body.setSpacing(6)
    outer.addLayout(body)
    return frame, body


def key_value_label(text: str = "—") -> QLabel:
    """Monospace value label."""
    label = QLabel(text)
    label.setObjectName("ValueLabel")
    return label


def key_label(text: str) -> QLabel:
    """Muted key label."""
    label = QLabel(text)
    label.setObjectName("KeyLabel")
    return label


def set_invalid(widget: QWidget, invalid: bool) -> None:
    """Toggle the red ``invalid`` border defined in app.qss."""
    widget.setProperty("invalid", invalid)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def show_message(parent: QWidget | None, title: str, message: str, critical: bool = True) -> None:
    """Modal QMessageBox (critical or warning)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical if critical else QMessageBox.Icon.Warning)
    box.setWindowTitle(f"Traffic Monitor — {title}")
    box.setText(title)
    box.setInformativeText(message)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def show_info(parent: QWidget | None, title: str, message: str) -> None:
    """Modal information box."""
    QMessageBox.information(parent, f"Traffic Monitor — {title}", message)


def confirm(parent: QWidget | None, title: str, message: str) -> bool:
    """Yes/No question; returns ``True`` on Yes."""
    answer = QMessageBox.question(
        parent, f"Traffic Monitor — {title}", message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes
