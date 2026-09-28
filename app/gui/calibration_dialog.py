"""Dialog to pick the 4 perspective points on a real frame of the source."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.ui_helpers import bgr_to_qimage
from app.utils.exceptions import PerspectiveError
from app.vision.geometry import scale_points, validate_quadrilateral

POINT_NAMES = ("A · trên-trái", "B · trên-phải", "C · dưới-phải", "D · dưới-trái")
MARGIN_RATIO = 0.25  # clickable area around the frame (road edges may be outside)


class CalibrationCanvas(QWidget):
    """Frame with a margin; left click adds a point, right click removes the last."""

    def __init__(self, frame: np.ndarray, points: list[list[float]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap.fromImage(bgr_to_qimage(frame, max_width=1600))
        self._frame_w, self._frame_h = frame.shape[1], frame.shape[0]
        self.points: list[list[float]] = [list(p) for p in points][:4]
        self.on_change = lambda: None
        self.setMinimumSize(900, 560)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------ mapping
    def _layout(self) -> tuple[float, float, float]:
        """Scale and offset mapping frame pixels to widget pixels."""
        total_w = self._frame_w * (1 + 2 * MARGIN_RATIO)
        total_h = self._frame_h * (1 + 2 * MARGIN_RATIO)
        scale = min(self.width() / total_w, self.height() / total_h)
        offset_x = (self.width() - self._frame_w * scale) / 2
        offset_y = (self.height() - self._frame_h * scale) / 2
        return scale, offset_x, offset_y

    def _to_widget(self, point: list[float]) -> QPointF:
        scale, ox, oy = self._layout()
        return QPointF(ox + point[0] * scale, oy + point[1] * scale)

    def _to_frame(self, position: QPointF) -> list[float]:
        scale, ox, oy = self._layout()
        return [round((position.x() - ox) / scale), round((position.y() - oy) / scale)]

    # ------------------------------------------------------------- events
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and len(self.points) < 4:
            self.points.append(self._to_frame(event.position()))
        elif event.button() == Qt.MouseButton.RightButton and self.points:
            self.points.pop()
        self.update()
        self.on_change()

    def reset(self) -> None:
        """Remove all points."""
        self.points.clear()
        self.update()
        self.on_change()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0A0B0D"))
        scale, ox, oy = self._layout()
        target = QRectF(ox, oy, self._frame_w * scale, self._frame_h * scale)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setPen(QPen(QColor("#323843"), 1, Qt.PenStyle.DashLine))
        painter.drawRect(target)
        widget_points = [self._to_widget(p) for p in self.points]
        if len(widget_points) >= 2:
            painter.setPen(QPen(QColor("#D9A441"), 2))
            polygon = QPolygonF(widget_points)
            if len(widget_points) == 4:
                painter.setBrush(QColor(217, 164, 65, 50))
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)
        painter.setFont(QFont(self.font().family(), 10, QFont.Weight.Bold))
        for index, point in enumerate(widget_points):
            painter.setBrush(QColor("#D9A441"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(point, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(point + QPointF(9, -9), "ABCD"[index])


class CalibrationDialog(QDialog):
    """Pick A, B, C, D on a frame; returns points in frame pixel coordinates."""

    def __init__(
        self,
        frame: np.ndarray,
        current_points: list[list[float]],
        current_reference: list[int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Speed calibration — perspective points")
        height, width = frame.shape[:2]
        self.reference_size = [width, height]
        try:
            initial = scale_points(np.asarray(current_points, dtype=np.float32), current_reference, (width, height))
            initial_points = [[round(float(x)), round(float(y))] for x, y in initial]
        except (ValueError, ZeroDivisionError):
            initial_points = []
        self.canvas = CalibrationCanvas(frame, initial_points)
        self.canvas.on_change = self._refresh
        self.status = QLabel()
        self.status.setWordWrap(True)
        instructions = QLabel(
            "Click lần lượt 4 góc của đoạn đường đã biết kích thước thật: A trên-trái → B trên-phải → "
            "C dưới-phải → D dưới-trái. Có thể click ra ngoài khung hình. Chuột phải: xoá điểm cuối."
        )
        instructions.setWordWrap(True)
        instructions.setObjectName("Hint")
        reset_button = QPushButton("Reset points")
        reset_button.clicked.connect(self.canvas.reset)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        footer = QHBoxLayout()
        footer.addWidget(reset_button)
        footer.addWidget(self.status, 1)
        footer.addWidget(self.buttons)
        layout = QVBoxLayout(self)
        layout.addWidget(instructions)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(footer)
        self.resize(1200, 760)
        self._refresh()

    def _refresh(self) -> None:
        points = self.canvas.points
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if len(points) < 4:
            self.status.setText(f"Điểm tiếp theo: {POINT_NAMES[len(points)]}  ({len(points)}/4)")
            ok_button.setEnabled(False)
            return
        try:
            validate_quadrilateral(points)
        except PerspectiveError as exc:
            self.status.setText(f"✕ {exc}")
            ok_button.setEnabled(False)
            return
        self.status.setText("✓ " + "  ".join(f"{'ABCD'[i]}({int(x)}, {int(y)})" for i, (x, y) in enumerate(points)))
        ok_button.setEnabled(True)

    def selected_points(self) -> list[list[int]]:
        """The 4 validated points in frame pixel coordinates."""
        return [[int(x), int(y)] for x, y in self.canvas.points]
