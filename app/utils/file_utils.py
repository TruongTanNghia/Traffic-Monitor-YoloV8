"""Filesystem helpers built on pathlib."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.utils.constants import PROJECT_ROOT


def resolve_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def to_project_relative(path: str | Path) -> str:
    """Return a POSIX-style path relative to the project root when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_filename(prefix: str, session_id: str, extension: str) -> str:
    """Build ``<prefix>_<session_id>.<extension>``."""
    return f"{prefix}_{session_id}.{extension.lstrip('.')}"


def open_in_file_manager(path: Path) -> None:
    """Open a folder in the operating system file manager."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606 - intended OS integration
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])  # noqa: S603,S607
    else:
        subprocess.Popen(["xdg-open", str(path)])  # noqa: S603,S607
