"""Factory assembling all processing components from an AppConfig."""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import AppConfig
from app.core.alert_manager import AlertManager
from app.core.counter import TrafficCounter
from app.core.detector import VehicleDetector
from app.core.speed_estimator import SpeedEstimator
from app.core.tracker import VehicleTracker
from app.core.zone_manager import ZoneManager
from app.utils.logger import get_logger
from app.vision.annotations import AnnotationRenderer

logger = get_logger(__name__)


@dataclass
class PipelineComponents:
    """Every stateful component used by the FrameProcessor."""

    detector: VehicleDetector
    tracker: VehicleTracker
    counter: TrafficCounter
    speed_estimator: SpeedEstimator
    zone_manager: ZoneManager
    alert_manager: AlertManager
    renderer: AnnotationRenderer


class ProcessingPipeline:
    """Build :class:`PipelineComponents` for a given source geometry."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def build(
        self,
        frame_size: tuple[int, int],
        fps: float,
        detector: VehicleDetector | None = None,
    ) -> PipelineComponents:
        """Create and initialise the components.

        Args:
            frame_size: ``(width, height)`` of the source.
            fps: frames per second of the source.
            detector: optional pre-built detector (tests); otherwise the model
                is loaded from ``config.model_path``.

        Raises:
            ModelLoadError, PerspectiveError: on invalid model or zone.
        """
        config = self._config
        zone_manager = ZoneManager(
            frame_size, config.line_y_ratio, config.perspective_source, config.perspective_reference_size
        )
        if detector is None:
            detector = VehicleDetector(config.model_file, config.confidence_threshold, config.device)
            detector.load_model()
            detector.warmup()
        tracker = VehicleTracker(fps, track_activation_threshold=min(0.25, config.confidence_threshold))
        counter = TrafficCounter(detector.get_vehicle_classes())
        counter.initialize_line(frame_size[0], frame_size[1], config.line_y_ratio)
        speed_estimator = SpeedEstimator(fps, config.min_track_seconds)
        speed_estimator.initialize_transformer(
            zone_manager.speed_polygon, config.road_width_m, config.road_length_m
        )
        logger.info("Pipeline built for %dx%d @ %.1f fps", frame_size[0], frame_size[1], fps)
        return PipelineComponents(
            detector=detector,
            tracker=tracker,
            counter=counter,
            speed_estimator=speed_estimator,
            zone_manager=zone_manager,
            alert_manager=AlertManager(config.speed_limit_kmh),
            renderer=AnnotationRenderer(frame_size, fps),
        )
