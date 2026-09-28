"""Traffic Monitor Desktop entry point.

Responsibilities only: start the application, load configuration, create the
MainWindow and run the QApplication event loop.
"""

from __future__ import annotations

import os
import sys
from types import TracebackType

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMessageBox

from app.config.settings import load_config_safely
from app.gui.main_window import MainWindow
from app.utils.constants import APP_NAME, CONFIG_FILE, LOG_FILE, PROJECT_ROOT, STYLE_FILE
from app.utils.logger import get_logger, setup_logging

logger = get_logger("main")


def read_stylesheet() -> str:
    """Return the QSS theme, or an empty string when missing."""
    try:
        return STYLE_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Stylesheet not loaded: %s", exc)
        return ""


def install_exception_hook() -> None:
    """Log unexpected exceptions and show them instead of crashing the GUI."""

    def hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        if QApplication.instance() is not None:
            QMessageBox.critical(None, f"{APP_NAME} — Lỗi không mong đợi", f"{exc_type.__name__}: {exc}")

    sys.excepthook = hook


def main() -> int:
    """Start the application."""
    load_dotenv(PROJECT_ROOT / ".env")
    setup_logging(LOG_FILE, os.getenv("TM_LOG_LEVEL", "INFO"))
    install_exception_hook()
    config, config_error = load_config_safely(CONFIG_FILE)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(read_stylesheet())
    window = MainWindow(config, config_error)
    window.show()
    logger.info("%s started", APP_NAME)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
