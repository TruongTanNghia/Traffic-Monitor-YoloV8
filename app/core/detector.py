"""Vehicle detection with Ultralytics YOLO.

Pipeline: ``frame -> YOLO -> sv.Detections -> confidence filter -> vehicle class filter``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from app.utils.constants import VEHICLE_CLASSES
from app.utils.exceptions import InferenceError, ModelLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)

ModelFactory = Callable[[str], Any]


def resolve_device(requested: str) -> str:
    """Map ``"auto"``/``"cuda:N"``/``"cpu"`` to an available torch device."""
    requested = (requested or "auto").strip().lower()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:  # pragma: no cover - torch ships with ultralytics
        cuda_available = False
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if requested.startswith("cuda"):
        if cuda_available:
            return requested if ":" in requested else "cuda:0"
        logger.warning("CUDA requested (%s) but not available, falling back to CPU", requested)
        return "cpu"
    logger.warning("Unknown device %r, using auto", requested)
    return resolve_device("auto")


def describe_device(device: str) -> str:
    """Human readable device name, e.g. ``cuda:0 (NVIDIA GeForce RTX 3050)``."""
    if device.startswith("cuda"):
        try:
            import torch

            index = int(device.split(":")[1]) if ":" in device else 0
            return f"{device} ({torch.cuda.get_device_name(index)})"
        except Exception:  # noqa: BLE001 - purely informational
            return device
    return device


def fp16_predict_kwargs(device: str) -> dict[str, Any]:
    """FP16 inference arguments for the installed Ultralytics version.

    Ultralytics >= 8.4 replaced ``half=True`` with ``quantize=16``.
    """
    if not device.startswith("cuda"):
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except ImportError:  # pragma: no cover
        pass
    return {"half": True}


def _default_model_factory(path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(path)


class VehicleDetector:
    """Detect cars, motorcycles, buses and trucks in BGR frames."""

    MAX_INFERENCE_WIDTH = 1280

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.30,
        device: str = "auto",
        vehicle_classes: tuple[str, ...] = VEHICLE_CLASSES,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self._requested_device = device
        self._vehicle_class_names = vehicle_classes
        self._model_factory = model_factory or _default_model_factory
        self._model: Any = None
        self._vehicle_classes: dict[int, str] = {}
        self._device = "cpu"
        self._precision_kwargs: dict[str, Any] = {}

    # --------------------------------------------------------------- loading
    def load_model(self) -> None:
        """Load the YOLO model and resolve vehicle class ids from ``model.names``.

        Raises:
            ModelLoadError: missing file, unreadable model or no vehicle class.
        """
        if not self.model_path.is_file():
            raise ModelLoadError(
                f"Không tìm thấy file model: {self.model_path}. "
                "Chạy 'python scripts/download_model.py' hoặc chọn model khác trong Configuration."
            )
        try:
            model = self._model_factory(str(self.model_path))
        except Exception as exc:  # noqa: BLE001 - any loader failure is a load error
            raise ModelLoadError(f"Không load được model {self.model_path.name}: {exc}") from exc

        names = self._names_of(model)
        vehicle_classes = {
            class_id: name for class_id, name in names.items() if name in self._vehicle_class_names
        }
        if not vehicle_classes:
            raise ModelLoadError(
                f"Model {self.model_path.name} không có lớp nào trong {self._vehicle_class_names}."
            )
        self._model = model
        self._vehicle_classes = vehicle_classes
        self._device = resolve_device(self._requested_device)
        self._precision_kwargs = fp16_predict_kwargs(self._device)
        logger.info("Model loaded: %s on %s (classes: %s)", self.model_path.name,
                    describe_device(self._device), ", ".join(vehicle_classes.values()))

    def warmup(self, frame_size: tuple[int, int] = (640, 640)) -> None:
        """Run one dummy inference so the first real frame is not slow."""
        width, height = frame_size
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))

    @staticmethod
    def _names_of(model: Any) -> dict[int, str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return dict(enumerate(map(str, names)))
        raise ModelLoadError("Model không cung cấp danh sách lớp (names).")

    # ------------------------------------------------------------- inference
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLO on one BGR frame.

        Raises:
            InferenceError: when the model fails.
        """
        if self._model is None:
            raise InferenceError("Model chưa được load. Gọi load_model() trước.")
        try:
            model_input, scale = self._downscale(frame)
            results = self._model.predict(
                model_input,
                conf=self.confidence_threshold,
                classes=list(self._vehicle_classes),
                device=self._device,
                verbose=False,
                **self._precision_kwargs,
            )
            detections = sv.Detections.from_ultralytics(results[0])
        except Exception as exc:  # noqa: BLE001 - wrap all inference failures
            raise InferenceError(f"Inference thất bại: {exc}") from exc
        if scale != 1.0 and len(detections) > 0:
            detections.xyxy = detections.xyxy / scale
        return detections

    def _downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Shrink very large frames before inference.

        YOLO letterboxes to 640 px anyway; resizing 4K frames once here is much
        cheaper than letting the predictor do it. Boxes are scaled back.
        """
        width = frame.shape[1]
        if width <= self.MAX_INFERENCE_WIDTH:
            return frame, 1.0
        scale = self.MAX_INFERENCE_WIDTH / width
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return resized, scale

    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """Keep detections with ``confidence >= threshold`` and a vehicle class."""
        if len(detections) == 0:
            return detections
        mask = np.isin(detections.class_id, list(self._vehicle_classes))
        if detections.confidence is not None:
            mask &= detections.confidence >= self.confidence_threshold
        return detections[mask]

    # ----------------------------------------------------------------- info
    def get_vehicle_classes(self) -> dict[int, str]:
        """Return ``{class_id: name}`` of the vehicle classes of the model."""
        return dict(self._vehicle_classes)

    def class_name(self, class_id: int) -> str:
        """Vehicle type name for ``class_id``."""
        return self._vehicle_classes.get(int(class_id), "unknown")

    @property
    def device(self) -> str:
        """Resolved torch device."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Whether :meth:`load_model` succeeded."""
        return self._model is not None
