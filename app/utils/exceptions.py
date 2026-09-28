"""Exception hierarchy of the application.

Every error that can reach the user derives from :class:`TrafficMonitorError`
and carries a human readable ``title`` used by the GUI message box.
"""

from __future__ import annotations


class TrafficMonitorError(Exception):
    """Base class of all application errors."""

    title = "Lỗi"
    critical = True


class ModelLoadError(TrafficMonitorError):
    """The YOLO model file is missing or cannot be loaded."""

    title = "Không load được model"


class InferenceError(TrafficMonitorError):
    """Model inference failed on a frame."""

    title = "Lỗi inference"


class ConfigError(TrafficMonitorError):
    """Configuration values are invalid."""

    title = "Cấu hình không hợp lệ"
    critical = False


class PerspectiveError(TrafficMonitorError):
    """Perspective source points do not form a valid quadrilateral."""

    title = "Perspective không hợp lệ"
    critical = False


class VideoWriterError(TrafficMonitorError):
    """The annotated output video cannot be written."""

    title = "Không ghi được video"
    critical = False


class VideoSourceError(TrafficMonitorError):
    """Base class for video input errors."""

    title = "Lỗi nguồn video"


class VideoNotFoundError(VideoSourceError):
    """The video file does not exist."""

    title = "Video không tồn tại"


class VideoOpenError(VideoSourceError):
    """The video exists but OpenCV cannot decode it."""

    title = "Không mở được video"


class CameraConnectionError(VideoSourceError):
    """The webcam cannot be opened or disconnected."""

    title = "Camera không kết nối"
    critical = False


class RTSPTimeoutError(VideoSourceError):
    """The RTSP stream timed out or could not be reconnected."""

    title = "RTSP timeout"
    critical = False
