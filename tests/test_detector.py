"""Unit tests for VehicleDetector (the YOLO model is mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from app.core.detector import VehicleDetector, resolve_device
from app.utils.exceptions import InferenceError, ModelLoadError
from tests.conftest import COCO_NAMES, VEHICLE_IDS, make_detections


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "fake.pt"
    path.write_bytes(b"not a real model")
    return path


def fake_model(names: dict[int, str] = COCO_NAMES) -> MagicMock:
    model = MagicMock()
    model.names = names
    return model


def loaded_detector(model_file: Path, threshold: float = 0.30, model: MagicMock | None = None) -> VehicleDetector:
    model = model or fake_model()
    detector = VehicleDetector(model_file, threshold, device="cpu", model_factory=lambda _path: model)
    detector.load_model()
    return detector


def test_missing_model_file_raises(tmp_path: Path) -> None:
    detector = VehicleDetector(tmp_path / "missing.pt")
    with pytest.raises(ModelLoadError, match="Không tìm thấy file model"):
        detector.load_model()


def test_unloadable_model_raises(model_file: Path) -> None:
    def broken_factory(_path: str) -> None:
        raise RuntimeError("corrupted weights")

    detector = VehicleDetector(model_file, model_factory=broken_factory)
    with pytest.raises(ModelLoadError, match="corrupted weights"):
        detector.load_model()


def test_model_without_vehicle_classes_raises(model_file: Path) -> None:
    detector = VehicleDetector(model_file, model_factory=lambda _p: fake_model({0: "person", 1: "dog"}))
    with pytest.raises(ModelLoadError):
        detector.load_model()


def test_vehicle_classes_resolved_from_model_names(model_file: Path) -> None:
    detector = loaded_detector(model_file)
    assert detector.get_vehicle_classes() == VEHICLE_IDS
    assert detector.class_name(7) == "truck"
    assert detector.class_name(0) == "unknown"


def test_vehicle_classes_follow_custom_model_ids(model_file: Path) -> None:
    detector = loaded_detector(model_file, model=fake_model({0: "truck", 1: "car", 2: "tree"}))
    assert detector.get_vehicle_classes() == {0: "truck", 1: "car"}


def test_filter_removes_non_vehicle_classes(model_file: Path) -> None:
    detector = loaded_detector(model_file)
    detections = make_detections([[0, 0, 10, 10]] * 4, [0, 2, 9, 7], [0.9, 0.9, 0.9, 0.9])
    filtered = detector.filter_detections(detections)
    assert filtered.class_id.tolist() == [2, 7]


def test_filter_by_confidence_threshold(model_file: Path) -> None:
    detector = loaded_detector(model_file, threshold=0.30)
    detections = make_detections([[0, 0, 10, 10]] * 3, [2, 2, 3], [0.29, 0.30, 0.95])
    filtered = detector.filter_detections(detections)
    assert filtered.confidence.tolist() == pytest.approx([0.30, 0.95])


def test_filter_empty_detections(model_file: Path) -> None:
    detector = loaded_detector(model_file)
    assert len(detector.filter_detections(sv.Detections.empty())) == 0


def test_detect_uses_model_and_converts_to_supervision(model_file: Path) -> None:
    model = fake_model()
    model.predict.return_value = ["raw-result"]
    detector = loaded_detector(model_file, model=model)
    expected = make_detections([[1, 2, 3, 4]], [2], [0.8])
    with patch("app.core.detector.sv.Detections.from_ultralytics", return_value=expected) as convert:
        result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    convert.assert_called_once_with("raw-result")
    kwargs = model.predict.call_args.kwargs
    assert kwargs["conf"] == 0.30 and sorted(kwargs["classes"]) == [2, 3, 5, 7] and kwargs["device"] == "cpu"
    assert result is expected


def test_detect_scales_boxes_back_for_large_frames(model_file: Path) -> None:
    model = fake_model()
    model.predict.return_value = ["raw"]
    detector = loaded_detector(model_file, model=model)
    small_frame_box = make_detections([[100, 100, 200, 200]], [2], [0.9])
    with patch("app.core.detector.sv.Detections.from_ultralytics", return_value=small_frame_box):
        result = detector.detect(np.zeros((2160, 3840, 3), dtype=np.uint8))
    assert model.predict.call_args.args[0].shape[1] == VehicleDetector.MAX_INFERENCE_WIDTH
    assert result.xyxy[0].tolist() == pytest.approx([300, 300, 600, 600])


def test_detect_wraps_model_errors(model_file: Path) -> None:
    model = fake_model()
    model.predict.side_effect = RuntimeError("CUDA out of memory")
    detector = loaded_detector(model_file, model=model)
    with pytest.raises(InferenceError, match="CUDA out of memory"):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detect_before_load_raises(model_file: Path) -> None:
    with pytest.raises(InferenceError):
        VehicleDetector(model_file).detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_resolve_device_cpu_and_fallback() -> None:
    assert resolve_device("cpu") == "cpu"
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("cuda:0") == "cpu"
        assert resolve_device("auto") == "cpu"
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda:0"
