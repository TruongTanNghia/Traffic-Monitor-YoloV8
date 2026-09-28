"""Session facade: open source → process frames → write outputs → report.

The Qt worker only talks to this class, which keeps the Computer Vision core
independent from the GUI.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from app.config.settings import AppConfig
from app.core.detector import describe_device
from app.core.pipeline import ProcessingPipeline
from app.reports.report_generator import ReportGenerator
from app.utils.constants import (
    MAX_CONSECUTIVE_INFERENCE_ERRORS,
    SNAPSHOTS_SUBDIR,
    VIDEOS_SUBDIR,
    EventType,
)
from app.utils.exceptions import InferenceError, TrafficMonitorError, VideoWriterError
from app.utils.file_utils import ensure_dir, timestamped_filename, to_project_relative
from app.utils.logger import get_logger
from app.utils.time_utils import FpsMeter, make_session_id
from app.video.frame_processor import FrameProcessor, FrameResult
from app.video.video_reader import VideoReader
from app.video.video_source import SourceInfo, VideoSource, create_video_source
from app.video.video_writer import VideoWriter

logger = get_logger(__name__)

WarningCallback = Callable[[TrafficMonitorError], None]


class TrafficMonitor:
    """Run one processing session."""

    def __init__(
        self,
        config: AppConfig,
        stop_event: threading.Event | None = None,
        on_warning: WarningCallback | None = None,
        pipeline: ProcessingPipeline | None = None,
        report_generator: ReportGenerator | None = None,
    ) -> None:
        self._config = config.copy()
        self._stop_event = stop_event or threading.Event()
        self._on_warning = on_warning or (lambda exc: None)
        self._pipeline = pipeline or ProcessingPipeline(self._config)
        self._reports = report_generator or ReportGenerator()
        self._source: VideoSource | None = None
        self._source_info: SourceInfo | None = None
        self._processor: FrameProcessor | None = None
        self._writer: VideoWriter | None = None
        self._session_id = make_session_id()
        self._started_at: datetime | None = None
        self._started_perf = 0.0
        self._frames_processed = 0
        self._events: list[dict[str, Any]] = []
        self._snapshots: list[str] = []
        self._last_statistics: dict[str, Any] = {}
        self._closed_summary: dict[str, Any] | None = None

    # ------------------------------------------------------------ lifecycle
    def open(self) -> None:
        """Validate config, open the source, load the model, prepare outputs.

        Raises:
            TrafficMonitorError: any configuration, source, model or zone error.
        """
        self._config.validate()
        source = create_video_source(self._config)
        source.open()
        self._source = source
        try:
            self._source_info = source.info()
            frame_size = (self._source_info.width, self._source_info.height)
            components = self._pipeline.build(frame_size, self._source_info.fps)
            self._processor = FrameProcessor(components)
        except TrafficMonitorError:
            source.release()
            self._source = None
            raise
        if self._config.save_output_video:
            video_dir = ensure_dir(self._config.output_path / VIDEOS_SUBDIR)
            self._writer = VideoWriter(
                video_dir / timestamped_filename("traffic_output", self._session_id, "mp4"),
                self._source_info.fps,
            )
        self._started_at = datetime.now()
        self._started_perf = time.perf_counter()
        logger.info("Session %s started: source=%s %dx%d@%.1f", self._session_id,
                    self._source_info.description, self._source_info.width,
                    self._source_info.height, self._source_info.fps)

    def iter_results(self) -> Iterator[FrameResult]:
        """Yield one :class:`FrameResult` per processed frame.

        Raises:
            InferenceError: after too many consecutive inference failures.
            VideoSourceError: when a live source is lost.
        """
        if self._source is None or self._processor is None:
            raise RuntimeError("open() must succeed before iter_results()")
        fps_meter = FpsMeter()
        consecutive_errors = 0
        for packet in VideoReader(self._source, self._stop_event):
            self._processor.set_processing_fps(fps_meter.tick())
            try:
                result = self._processor.process_frame(packet.frame, packet.frame_index, packet.timestamp_s)
            except InferenceError as exc:
                consecutive_errors += 1
                logger.warning("Frame %d skipped: %s", packet.frame_index, exc)
                if consecutive_errors >= MAX_CONSECUTIVE_INFERENCE_ERRORS:
                    raise InferenceError(
                        f"Inference thất bại {consecutive_errors} frame liên tiếp. Phiên xử lý đã dừng.\n{exc}"
                    ) from exc
                continue
            consecutive_errors = 0
            self._record(result)
            yield result

    def _record(self, result: FrameResult) -> None:
        self._frames_processed += 1
        self._last_statistics = result.statistics
        for event in result.new_events:
            self._events.append(event.to_dict())
            if event.event_type == EventType.SPEED_VIOLATION.value and self._config.save_snapshots:
                self._save_snapshot(result, event.tracker_id)
        self._write_frame(result)

    def _write_frame(self, result: FrameResult) -> None:
        if self._writer is None:
            return
        try:
            self._writer.write(result.annotated_frame)
        except VideoWriterError as exc:
            logger.error("Video writer disabled: %s", exc)
            self._writer = None
            self._on_warning(VideoWriterError(f"{exc}\nTiếp tục xử lý nhưng không lưu video."))

    def _save_snapshot(self, result: FrameResult, tracker_id: int) -> None:
        folder = ensure_dir(self._config.output_path / SNAPSHOTS_SUBDIR)
        path = folder / f"violation_{self._session_id}_id{tracker_id}_f{result.frame_index}.jpg"
        ok, buffer = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            path.write_bytes(buffer.tobytes())  # imwrite cannot handle non-ASCII paths on Windows
            self._snapshots.append(to_project_relative(path))

    def close(self, status: str = "completed") -> dict[str, Any]:
        """Release resources, save the report and return the session summary.

        Safe to call several times and after a failed :meth:`open`.
        """
        if self._closed_summary is not None:
            return self._closed_summary
        if self._source is not None:
            self._source.release()
        video_path = self._writer.release() if self._writer is not None else None
        outputs: dict[str, Any] = {
            "video": to_project_relative(video_path) if video_path else None,
            "json": None,
            "csv": None,
            "snapshots": list(self._snapshots),
        }
        report = self.build_report(outputs, status) if self._frames_processed > 0 else None
        if report is not None and self._config.save_report:
            outputs.update(self._save_report(report))
            report["outputs"] = outputs
        summary = {
            "session_id": self._session_id,
            "status": status,
            "frames_processed": self._frames_processed,
            "statistics": self._last_statistics,
            "outputs": outputs,
            "report": report,
            "output_dir": str(self._config.output_path),
        }
        self._closed_summary = summary
        logger.info("Session %s %s after %d frames", self._session_id, status, self._frames_processed)
        return summary

    def _save_report(self, report: dict[str, Any]) -> dict[str, str | None]:
        saved: dict[str, str | None] = {}
        try:
            saved["json"] = to_project_relative(self._reports.save_json(report, self._config.output_path))
            saved["csv"] = to_project_relative(
                self._reports.save_csv(report["events"], self._config.output_path, self._session_id))
        except OSError as exc:
            logger.error("Cannot save report: %s", exc)
            self._on_warning(TrafficMonitorError(f"Không lưu được báo cáo: {exc}"))
        return saved

    # --------------------------------------------------------------- queries
    def build_report(self, outputs: dict[str, Any], status: str) -> dict[str, Any]:
        """Report of the session so far."""
        ended = datetime.now()
        session = {
            "session_id": self._session_id,
            "status": status,
            "started_at": self._started_at.replace(microsecond=0).isoformat() if self._started_at else None,
            "ended_at": ended.replace(microsecond=0).isoformat(),
            "duration_s": round(time.perf_counter() - self._started_perf, 2),
            "frames_processed": self._frames_processed,
            "device": self.device_description,
        }
        source = self._source_info.to_dict() if self._source_info else {}
        return self._reports.generate(session, source, self._config.to_dict(),
                                      self._last_statistics, list(self._events), outputs)

    def session_info(self) -> dict[str, Any]:
        """Information emitted with ``processing_started``."""
        info = self._source_info.to_dict() if self._source_info else {}
        return {
            "session_id": self._session_id,
            "source": info,
            "model": self._config.model_file.name,
            "device": self.device_description,
            "confidence_threshold": self._config.confidence_threshold,
            "speed_limit_kmh": self._config.speed_limit_kmh,
            "output_video": to_project_relative(self._writer.output_path) if self._writer else None,
        }

    @property
    def device_description(self) -> str:
        """Device used for inference, e.g. ``cuda:0 (RTX 3050)``."""
        if self._processor is None:
            return "-"
        return describe_device(self._processor.components.detector.device)

    @property
    def session_id(self) -> str:
        """Identifier ``YYYYMMDD_HHMMSS`` shared by all output files."""
        return self._session_id

    @property
    def output_dir(self) -> Path:
        """Absolute output directory."""
        return self._config.output_path
