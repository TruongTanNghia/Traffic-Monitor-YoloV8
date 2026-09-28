"""Application-wide constants: paths, vehicle classes, event types."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

APP_NAME = "Traffic Monitor Desktop"
APP_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
SAMPLES_DIR = DATA_DIR / "samples"
DEFAULT_OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
ASSETS_DIR = PROJECT_ROOT / "assets"
STYLE_FILE = ASSETS_DIR / "styles" / "app.qss"

VIDEOS_SUBDIR = "videos"
REPORTS_SUBDIR = "reports"
SNAPSHOTS_SUBDIR = "snapshots"

VEHICLE_CLASSES: tuple[str, ...] = ("car", "motorcycle", "bus", "truck")
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".wmv")

SOURCE_FILE = "file"
SOURCE_CAMERA = "camera"
SOURCE_RTSP = "rtsp"
SOURCE_TYPES: tuple[str, ...] = (SOURCE_FILE, SOURCE_CAMERA, SOURCE_RTSP)

DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda:0", "cpu")

KMH_PER_MPS = 3.6
DEFAULT_FPS = 25.0
MAX_REASONABLE_FPS = 240.0
MAX_CONSECUTIVE_INFERENCE_ERRORS = 10
RTSP_RECONNECT_DELAYS_S: tuple[float, ...] = (1.0, 2.0, 4.0)
PREVIEW_MAX_WIDTH = 1280
STATISTICS_EMIT_INTERVAL_S = 0.2
LOG_VIEW_MAX_ROWS = 5000


class EventType(str, Enum):
    """Types of events recorded during a processing session."""

    VEHICLE_IN = "VEHICLE_IN"
    VEHICLE_OUT = "VEHICLE_OUT"
    SPEED_VIOLATION = "SPEED_VIOLATION"


class SessionState(str, Enum):
    """Lifecycle states of a processing session."""

    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
