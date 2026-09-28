"""Unified video input interface.

``VideoSource`` hides ``cv2.VideoCapture`` from the rest of the application:
the GUI only works with these classes (through the worker) and never touches
OpenCV capture objects directly.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.utils.constants import (
    DEFAULT_FPS,
    MAX_REASONABLE_FPS,
    SOURCE_CAMERA,
    SOURCE_FILE,
    SOURCE_RTSP,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from app.utils.exceptions import (
    CameraConnectionError,
    ConfigError,
    RTSPTimeoutError,
    VideoNotFoundError,
    VideoOpenError,
    VideoSourceError,
)
from app.utils.file_utils import resolve_path
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.config.settings import AppConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class SourceInfo:
    """Metadata of an opened source."""

    source_type: str
    uri: str
    description: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    is_live: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary."""
        return {
            "type": self.source_type,
            "uri": self.uri,
            "description": self.description,
            "resolution": [self.width, self.height],
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "is_live": self.is_live,
        }


class VideoSource(ABC):
    """Abstract video input."""

    source_type: str = ""
    is_live: bool = False

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None

    # ------------------------------------------------------------ interface
    @abstractmethod
    def _create_capture(self) -> cv2.VideoCapture:
        """Create the underlying OpenCV capture."""

    @abstractmethod
    def _open_error(self) -> VideoSourceError:
        """Return the error raised when the capture cannot be opened."""

    @property
    @abstractmethod
    def uri(self) -> str:
        """Identifier of the source (path, index, URL)."""

    @property
    def description(self) -> str:
        """Short human readable name."""
        return self.uri

    # ---------------------------------------------------------------- logic
    def open(self) -> None:
        """Open the source.

        Raises:
            VideoSourceError: subclass specific error when opening fails.
        """
        self.release()
        capture = self._create_capture()
        if not capture.isOpened():
            capture.release()
            raise self._open_error()
        self._capture = capture
        logger.info("Opened %s source: %s", self.source_type, self.description)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame. Returns ``(False, None)`` at end/failure."""
        if self._capture is None:
            return False, None
        ok, frame = self._capture.read()
        return (True, frame) if ok and frame is not None else (False, None)

    def release(self) -> None:
        """Release the capture. Safe to call several times."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_opened(self) -> bool:
        """Whether the capture is open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        """Frames per second, falling back to ``DEFAULT_FPS`` when unknown."""
        value = self._get(cv2.CAP_PROP_FPS)
        if not value or math.isnan(value) or value <= 0 or value > MAX_REASONABLE_FPS:
            return DEFAULT_FPS
        return float(value)

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return int(self._get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int | None:
        """Total frames for files, ``None`` for live sources."""
        if self.is_live:
            return None
        count = int(self._get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None

    def info(self) -> SourceInfo:
        """Return metadata of the opened source."""
        width, height = self.frame_size
        return SourceInfo(self.source_type, self.uri, self.description, width, height,
                          self.fps, self.frame_count, self.is_live)

    def _get(self, prop: int) -> float:
        return float(self._capture.get(prop)) if self._capture is not None else 0.0

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class FileVideoSource(VideoSource):
    """Local video file."""

    source_type = SOURCE_FILE
    is_live = False

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = resolve_path(path)

    @property
    def uri(self) -> str:
        return str(self._path)

    @property
    def description(self) -> str:
        return self._path.name

    @property
    def path(self) -> Path:
        """Absolute file path."""
        return self._path

    def open(self) -> None:
        if not self._path.is_file():
            raise VideoNotFoundError(f"Không tìm thấy file video: {self._path}")
        if self._path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise VideoOpenError(
                f"Định dạng {self._path.suffix} không được hỗ trợ. "
                f"Hỗ trợ: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
            )
        super().open()
        ok, _ = self.read()
        if not ok:
            self.release()
            raise VideoOpenError(f"Không đọc được frame nào từ {self._path.name} (file hỏng hoặc codec không hỗ trợ).")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[union-attr]

    def _create_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(self._path))

    def _open_error(self) -> VideoSourceError:
        return VideoOpenError(
            f"OpenCV không mở được {self._path.name}. File có thể hỏng hoặc dùng codec không được hỗ trợ."
        )


class CameraVideoSource(VideoSource):
    """Local webcam (DirectShow backend on Windows for fast opening)."""

    source_type = SOURCE_CAMERA
    is_live = True

    def __init__(self, index: int = 0) -> None:
        super().__init__()
        self._index = int(index)

    @property
    def uri(self) -> str:
        return str(self._index)

    @property
    def description(self) -> str:
        return f"Camera {self._index}"

    def _create_capture(self) -> cv2.VideoCapture:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        return cv2.VideoCapture(self._index, backend)

    def _open_error(self) -> VideoSourceError:
        return CameraConnectionError(
            f"Không mở được camera index {self._index}. Kiểm tra camera đã cắm và không bị ứng dụng khác sử dụng."
        )


class RTSPVideoSource(VideoSource):
    """RTSP network stream with open/read timeouts."""

    source_type = SOURCE_RTSP
    is_live = True

    def __init__(self, url: str, timeout_ms: int = 5000) -> None:
        super().__init__()
        if not url.lower().startswith(("rtsp://", "rtsps://")):
            raise ConfigError(f"RTSP URL phải bắt đầu bằng rtsp://, nhận: {url!r}")
        self._url = url
        self._timeout_ms = int(timeout_ms)

    @property
    def uri(self) -> str:
        return self._url

    @property
    def description(self) -> str:
        return mask_credentials(self._url)

    def _create_capture(self) -> cv2.VideoCapture:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_ms,
        ]
        return cv2.VideoCapture(self._url, cv2.CAP_FFMPEG, params)

    def _open_error(self) -> VideoSourceError:
        return RTSPTimeoutError(
            f"Không kết nối được {self.description} trong {self._timeout_ms} ms."
        )


def mask_credentials(url: str) -> str:
    """Hide ``user:password@`` in a URL."""
    scheme, sep, rest = url.partition("://")
    if not sep or "@" not in rest:
        return url
    return f"{scheme}://***@{rest.split('@', 1)[1]}"


def create_video_source(config: AppConfig) -> VideoSource:
    """Factory building the source described by ``config``.

    Raises:
        ConfigError: when the source settings are invalid.
    """
    uri = config.source_uri.strip()
    if config.source_type == SOURCE_FILE:
        if not uri:
            raise VideoNotFoundError("Chưa chọn file video. Vào trang Video Source để chọn.")
        return FileVideoSource(uri)
    if config.source_type == SOURCE_CAMERA:
        try:
            return CameraVideoSource(int(uri or 0))
        except ValueError as exc:
            raise ConfigError(f"Camera index không hợp lệ: {uri!r}") from exc
    if config.source_type == SOURCE_RTSP:
        return RTSPVideoSource(uri, config.rtsp_timeout_ms)
    raise ConfigError(f"source_type không hợp lệ: {config.source_type!r}")


def probe_source(config: AppConfig) -> SourceInfo:
    """Open the configured source briefly and return its metadata.

    Used by the GUI (in a background thread) to test a source.
    """
    source = create_video_source(config)
    with source:
        ok, _ = source.read()
        if not ok:
            raise source._open_error()  # noqa: SLF001 - same module family
        return source.info()


def read_first_frame(config: AppConfig) -> np.ndarray:
    """Return the first frame of the configured source (for calibration)."""
    source = create_video_source(config)
    with source:
        ok, frame = source.read()
        if not ok or frame is None:
            raise source._open_error()  # noqa: SLF001
        return frame
