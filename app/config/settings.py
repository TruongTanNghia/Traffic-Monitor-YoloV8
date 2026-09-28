"""AppConfig dataclass: the single source of configuration.

Business logic receives an :class:`AppConfig` through its constructor and never
reads hard-coded settings. Configuration is persisted as YAML.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from app.config.default_config import (
    DEFAULT_PERSPECTIVE_REFERENCE_SIZE,
    DEFAULT_PERSPECTIVE_SOURCE,
    DEFAULTS,
    RANGES,
)
from app.utils.constants import DEVICE_CHOICES, SOURCE_TYPES
from app.utils.exceptions import ConfigError
from app.utils.file_utils import resolve_path
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AppConfig:
    """All tunable parameters of the application."""

    model_path: str = str(DEFAULTS["model_path"])
    confidence_threshold: float = float(DEFAULTS["confidence_threshold"])  # type: ignore[arg-type]
    speed_limit_kmh: float = float(DEFAULTS["speed_limit_kmh"])  # type: ignore[arg-type]
    line_y_ratio: float = float(DEFAULTS["line_y_ratio"])  # type: ignore[arg-type]
    road_width_m: float = float(DEFAULTS["road_width_m"])  # type: ignore[arg-type]
    road_length_m: float = float(DEFAULTS["road_length_m"])  # type: ignore[arg-type]
    perspective_source: list[list[int]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PERSPECTIVE_SOURCE)
    )
    perspective_reference_size: list[int] = field(
        default_factory=lambda: list(DEFAULT_PERSPECTIVE_REFERENCE_SIZE)
    )
    show_preview: bool = True
    save_output_video: bool = True
    save_report: bool = True
    save_snapshots: bool = True
    source_type: str = str(DEFAULTS["source_type"])
    source_uri: str = str(DEFAULTS["source_uri"])
    output_dir: str = str(DEFAULTS["output_dir"])
    min_track_seconds: float = float(DEFAULTS["min_track_seconds"])  # type: ignore[arg-type]
    device: str = str(DEFAULTS["device"])
    rtsp_timeout_ms: int = int(DEFAULTS["rtsp_timeout_ms"])  # type: ignore[call-overload]

    # ------------------------------------------------------------------ paths
    @property
    def model_file(self) -> Path:
        """Absolute path of the model file."""
        return resolve_path(self.model_path)

    @property
    def output_path(self) -> Path:
        """Absolute path of the output directory."""
        return resolve_path(self.output_dir)

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        """Validate every field.

        Raises:
            ConfigError: listing every invalid field.
        """
        errors = self.collect_errors()
        if errors:
            raise ConfigError("; ".join(f"{name}: {msg}" for name, msg in errors))

    def collect_errors(self) -> list[tuple[str, str]]:
        """Return ``(field, message)`` pairs for every invalid field."""
        errors: list[tuple[str, str]] = []
        for name, (low, high) in RANGES.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not low <= value <= high:
                errors.append((name, f"phải nằm trong [{low}, {high}], nhận {value!r}"))
        if not str(self.model_path).strip():
            errors.append(("model_path", "không được để trống"))
        elif not str(self.model_path).lower().endswith(".pt"):
            errors.append(("model_path", "phải là file .pt"))
        if not str(self.output_dir).strip():
            errors.append(("output_dir", "không được để trống"))
        if self.source_type not in SOURCE_TYPES:
            errors.append(("source_type", f"phải là một trong {SOURCE_TYPES}"))
        if self.device not in DEVICE_CHOICES and not str(self.device).startswith("cuda:"):
            errors.append(("device", f"phải là một trong {DEVICE_CHOICES}"))
        errors.extend(self._perspective_shape_errors())
        return errors

    def _perspective_shape_errors(self) -> list[tuple[str, str]]:
        points = self.perspective_source
        valid = (
            isinstance(points, list)
            and len(points) == 4
            and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points)
            and all(isinstance(v, (int, float)) for p in points for v in p)
        )
        errors = [] if valid else [("perspective_source", "cần đúng 4 điểm [x, y]")]
        size = self.perspective_reference_size
        if not (
            isinstance(size, (list, tuple))
            and len(size) == 2
            and all(isinstance(v, (int, float)) and v > 0 for v in size)
        ):
            errors.append(("perspective_reference_size", "cần [width, height] > 0"))
        return errors

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-friendly dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Build a config from a dictionary, ignoring unknown keys and
        falling back to defaults for missing ones."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning("Ignoring unknown config keys: %s", sorted(unknown))
        config = cls(**{key: value for key, value in data.items() if key in known})
        config._coerce_types()
        return config

    def _coerce_types(self) -> None:
        for name in ("confidence_threshold", "speed_limit_kmh", "line_y_ratio",
                     "road_width_m", "road_length_m", "min_track_seconds"):
            try:
                setattr(self, name, float(getattr(self, name)))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{name}: không phải số ({getattr(self, name)!r})") from exc
        try:
            self.rtsp_timeout_ms = int(self.rtsp_timeout_ms)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"rtsp_timeout_ms: không phải số nguyên") from exc
        for name in ("show_preview", "save_output_video", "save_report", "save_snapshots"):
            setattr(self, name, bool(getattr(self, name)))
        for name in ("model_path", "source_type", "source_uri", "output_dir", "device"):
            setattr(self, name, "" if getattr(self, name) is None else str(getattr(self, name)))

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        """Load configuration from YAML. A missing file yields defaults.

        Raises:
            ConfigError: when the YAML is malformed or values are invalid.
        """
        if not path.is_file():
            logger.info("Config file %s not found, using defaults", path)
            return cls().with_env_overrides()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"Không đọc được {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path.name} phải là một mapping YAML")
        config = cls.from_dict(data).with_env_overrides()
        config.validate()
        logger.info("Configuration loaded from %s", path)
        return config

    def save(self, path: Path) -> None:
        """Validate then write the configuration to YAML."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        path.write_text(text, encoding="utf-8")
        logger.info("Configuration saved to %s", path)

    def with_env_overrides(self) -> AppConfig:
        """Apply ``TM_DEVICE`` and ``TM_RTSP_URL`` environment variables."""
        device = os.getenv("TM_DEVICE")
        if device:
            self.device = device.strip()
        rtsp_url = os.getenv("TM_RTSP_URL")
        if rtsp_url and self.source_type == "rtsp" and not self.source_uri:
            self.source_uri = rtsp_url.strip()
        return self

    def copy(self) -> AppConfig:
        """Return a deep copy."""
        return copy.deepcopy(self)


def load_config_safely(path: Path) -> tuple[AppConfig, str | None]:
    """Load configuration, falling back to defaults on error.

    Returns:
        The configuration and an optional error message to show the user.
    """
    try:
        return AppConfig.load(path), None
    except ConfigError as exc:
        logger.error("Invalid configuration, using defaults: %s", exc)
        return AppConfig().with_env_overrides(), str(exc)
