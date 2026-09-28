"""Widget painting the latest processed frame (QImage) with aspect ratio kept."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class VideoView(QWidget):
    """Display frames produced by the worker. QPixmap is created here, in the GUI thread."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = "Chưa có video — nhấn START để bắt đầu xử lý"
        self._frames_shown = 0
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(960, 540)

    def show_image(self, image: QImage) -> None:
        """Show a new frame."""
        self._pixmap = QPixmap.fromImage(image)
        self._frames_shown += 1
        self.update()

    def clear(self, message: str | None = None) -> None:
        """Remove the frame and show a placeholder message."""
        self._pixmap = None
        if message is not None:
            self._placeholder = message
        self.update()

    def set_placeholder(self, message: str) -> None:
        """Change the message shown when there is no frame."""
        self._placeholder = message
        if self._pixmap is None:
            self.update()

    @property
    def frames_shown(self) -> int:
        """Number of frames displayed since creation."""
        return self._frames_shown

    @property
    def has_frame(self) -> bool:
        """Whether a frame is currently displayed."""
        return self._pixmap is not None

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#7F8893"))
            painter.setFont(QFont(self.font().family(), 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return
        scaled = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(
            (self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2,
            scaled.width(), scaled.height(),
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(target, self._pixmap)
