"""Frame iterator on top of a VideoSource with prefetch and reconnect support."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from app.utils.constants import RTSP_RECONNECT_DELAYS_S
from app.utils.exceptions import CameraConnectionError, RTSPTimeoutError, VideoSourceError
from app.utils.logger import get_logger
from app.video.video_source import RTSPVideoSource, VideoSource

logger = get_logger(__name__)

_END = object()


@dataclass(frozen=True)
class FramePacket:
    """A frame with its index and timestamp in seconds.

    For files ``timestamp_s = frame_index / fps`` (video time, independent of
    processing speed). For live sources it is monotonic wall time since start.
    """

    frame_index: int
    timestamp_s: float
    frame: np.ndarray


class VideoReader:
    """Iterate frames of an opened :class:`VideoSource`.

    File sources are decoded in a background thread (``prefetch`` frames ahead)
    so decoding overlaps with GPU inference. Live sources are read directly and
    RTSP streams are reconnected with exponential backoff.
    """

    def __init__(
        self,
        source: VideoSource,
        stop_event: threading.Event | None = None,
        reconnect_delays_s: tuple[float, ...] = RTSP_RECONNECT_DELAYS_S,
        prefetch: int = 4,
    ) -> None:
        self._source = source
        self._stop_event = stop_event or threading.Event()
        self._reconnect_delays = reconnect_delays_s
        self._prefetch = prefetch
        self._fps = source.fps
        self._frame_index = 0

    @property
    def frames_read(self) -> int:
        """Number of frames yielded so far."""
        return self._frame_index

    def __iter__(self) -> Iterator[FramePacket]:
        if self._source.is_live or self._prefetch <= 0:
            yield from self._iter_direct()
        else:
            yield from self._iter_prefetched()

    # ----------------------------------------------------------------- files
    def _iter_prefetched(self) -> Iterator[FramePacket]:
        buffer: queue.Queue[object] = queue.Queue(maxsize=self._prefetch)
        finished = threading.Event()
        thread = threading.Thread(target=self._decode_loop, args=(buffer, finished),
                                  name="VideoDecode", daemon=True)
        thread.start()
        try:
            while not self._stop_event.is_set():
                try:
                    item = buffer.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _END:
                    logger.info("End of video after %d frames", self._frame_index)
                    return
                yield FramePacket(self._frame_index, self._frame_index / self._fps, item)  # type: ignore[arg-type]
                self._frame_index += 1
        finally:
            finished.set()
            thread.join(timeout=2.0)

    def _decode_loop(self, buffer: queue.Queue[object], finished: threading.Event) -> None:
        while not finished.is_set() and not self._stop_event.is_set():
            ok, frame = self._source.read()
            item = frame if ok and frame is not None else _END
            while not finished.is_set():
                try:
                    buffer.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if item is _END:
                return

    # ------------------------------------------------------------------ live
    def _iter_direct(self) -> Iterator[FramePacket]:
        start = time.monotonic()
        while not self._stop_event.is_set():
            ok, frame = self._source.read()
            if not ok or frame is None:
                if not self._source.is_live:
                    logger.info("End of video after %d frames", self._frame_index)
                    return
                self._recover_live_source()
                continue
            timestamp = (
                time.monotonic() - start if self._source.is_live else self._frame_index / self._fps
            )
            yield FramePacket(self._frame_index, timestamp, frame)
            self._frame_index += 1

    def _recover_live_source(self) -> None:
        """Try to reconnect a live source with exponential backoff."""
        if not isinstance(self._source, RTSPVideoSource):
            raise CameraConnectionError(f"Mất kết nối {self._source.description}.")
        for attempt, delay in enumerate(self._reconnect_delays, start=1):
            logger.warning("RTSP read failed, reconnect %d/%d in %.0fs",
                           attempt, len(self._reconnect_delays), delay)
            if self._stop_event.wait(delay):
                return
            try:
                self._source.open()
                return
            except VideoSourceError as exc:
                logger.warning("Reconnect %d failed: %s", attempt, exc)
        raise RTSPTimeoutError(
            f"Mất kết nối {self._source.description} sau {len(self._reconnect_delays)} lần thử lại."
        )
