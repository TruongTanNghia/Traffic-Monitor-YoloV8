"""Unit tests for AppConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import AppConfig, load_config_safely
from app.utils.exceptions import ConfigError


def test_defaults_match_specification() -> None:
    config = AppConfig()
    assert config.model_path == "models/yolov8n.pt"
    assert config.confidence_threshold == 0.30
    assert config.speed_limit_kmh == 80
    assert len(config.perspective_source) == 4
    config.validate()


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    config = AppConfig(confidence_threshold=0.45, speed_limit_kmh=100, line_y_ratio=0.6, device="cpu")
    config.save(path)
    loaded = AppConfig.load(path)
    assert loaded == config


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert AppConfig.load(tmp_path / "missing.yaml").speed_limit_kmh == 80


def test_partial_yaml_uses_defaults_for_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("speed_limit_kmh: 60\nunknown_key: 1\n", encoding="utf-8")
    loaded = AppConfig.load(path)
    assert loaded.speed_limit_kmh == 60 and loaded.confidence_threshold == 0.30


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence_threshold": 1.4},
        {"confidence_threshold": 0.0},
        {"speed_limit_kmh": 0},
        {"line_y_ratio": 1.0},
        {"road_width_m": -1},
        {"perspective_source": [[0, 0], [1, 1], [2, 2]]},
        {"source_type": "ftp"},
        {"model_path": ""},
        {"output_dir": " "},
    ],
)
def test_invalid_values_raise(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        AppConfig(**overrides).validate()


def test_non_numeric_yaml_value_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("speed_limit_kmh: fast\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(path)


def test_broken_yaml_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("speed_limit_kmh: [unclosed\n", encoding="utf-8")
    config, error = load_config_safely(path)
    assert error is not None and config.speed_limit_kmh == 80


def test_env_override_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TM_DEVICE", "cpu")
    assert AppConfig.load(tmp_path / "missing.yaml").device == "cpu"
