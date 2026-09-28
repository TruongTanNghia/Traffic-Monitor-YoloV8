"""Annotated video writer.

Frames are encoded in a background thread (bounded queue) so that MP4 encoding
of large frames overlaps with GPU inference instead of adding to it.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from app.utils.exceptions import VideoWriterError
from app.utils.logger import get_logger

logger = get_logger(__name__)

_STOP = object()


class VideoWriter:
    """Write BGR frames to an MP4 file using the ``mp4v`` codec."""

    FOURCC = "mp4v"
    QUEUE_SIZE = 8

    def __init__(self, output_path: Path, fps: float, asynchronous: bool = True) -> None:
        self._output_path = output_path
        self._fps = fps if fps > 0 else 25.0
        self._asynchronous = asynchronous
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._frames_written = 0
        self._queue: queue.Queue[object] = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._error: VideoWriterError | None = None

    @property
    def output_path(self) -> Path:
        """Destination file."""
        return self._output_path

    @property
    def frames_written(self) -> int:
        """Number of frames encoded so far."""
        return self._frames_written

    def write(self, frame: np.ndarray) -> None:
        """Queue (or write) one frame, opening the file on first use.

        The caller must not modify ``frame`` afterwards.

        Raises:
            VideoWriterError: when the file cannot be created or a previous
                background write failed.
        """
        if self._error is not None:
            raise self._error
        if self._writer is None:
            self._open(frame)
        if not self._asynchronous:
            self._encode(frame)
            return
        self._queue.put(frame)

    def _open(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VideoWriterError(f"Không tạo được thư mục {self._output_path.parent}: {exc}") from exc
        writer = cv2.VideoWriter(
            str(self._output_path), cv2.VideoWriter_fourcc(*self.FOURCC), self._fps, (width, height)
        )
        if not writer.isOpened():
            raise VideoWriterError(
                f"Không tạo được file {self._output_path.name}. Kiểm tra quyền ghi thư mục output."
            )
        self._writer = writer
        self._size = (width, height)
        if self._asynchronous:
            self._thread = threading.Thread(target=self._encode_loop, name="VideoEncode", daemon=True)
            self._thread.start()
        logger.info("Writing video %s (%dx%d @ %.1f fps)", self._output_path, width, height, self._fps)

    def _encode(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (width, height) != self._size:
            frame = cv2.resize(frame, self._size)  # type: ignore[arg-type]
        try:
            self._writer.write(frame)  # type: ignore[union-attr]
        except cv2.error as exc:
            raise VideoWriterError(f"Lỗi ghi frame vào {self._output_path.name}: {exc}") from exc
        self._frames_written += 1

    def _encode_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if self._error is not None:
                continue  # drain the queue after a failure
            try:
                self._encode(item)  # type: ignore[arg-type]
            except VideoWriterError as exc:
                logger.error("%s", exc)
                self._error = exc

    def release(self) -> Path | None:
        """Flush queued frames and close the file.

        Returns:
            The file path if at least one frame was written, else ``None``.
        """
        if self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join()
            self._thread = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Video saved: %s (%d frames)", self._output_path, self._frames_written)
        return self._output_path if self._frames_written > 0 else None
