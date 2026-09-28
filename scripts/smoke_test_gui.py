"""End-to-end GUI smoke test with the real pipeline (YOLO + sample video).

Drives MainWindow like a user: START / PAUSE / RESUME / STOP, Configuration
save/reset/validation, Video Source test/apply, Reports export, Event Logs
filters, error dialogs and closing during processing. Message boxes are
captured instead of blocking.

Usage:
    python scripts/smoke_test_gui.py              # offscreen
    python scripts/smoke_test_gui.py --screens    # also save page screenshots
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import app.gui.main_window as main_window_module  # noqa: E402
import app.gui.report_panel as report_module  # noqa: E402
import app.gui.settings_panel as settings_module  # noqa: E402
import app.gui.source_panel as source_module  # noqa: E402
from app.config.settings import AppConfig  # noqa: E402
from app.gui.main_window import MainWindow  # noqa: E402
from app.utils.constants import LOG_FILE, STYLE_FILE  # noqa: E402
from app.utils.file_utils import resolve_path  # noqa: E402
from app.utils.logger import setup_logging  # noqa: E402

MESSAGES: list[tuple[str, str, str]] = []
RESULTS: list[tuple[str, bool, str]] = []
CONFIRM_ANSWER = {"value": True}


def _record(kind: str) -> Callable[..., None]:
    def recorder(_parent: object, title: str, message: str, *_: object, **__: object) -> None:
        MESSAGES.append((kind, title, message))
    return recorder


def _confirm(_parent: object, title: str, message: str) -> bool:
    MESSAGES.append(("confirm", title, message))
    return CONFIRM_ANSWER["value"]


def patch_dialogs() -> None:
    for module in (main_window_module, report_module, settings_module, source_module):
        for name, kind in (("show_message", "message"), ("show_info", "info")):
            if hasattr(module, name):
                setattr(module, name, _record(kind))
        if hasattr(module, "confirm"):
            module.confirm = _confirm
    report_module.open_in_file_manager = lambda path: MESSAGES.append(("open_folder", str(path), ""))


def wait_until(condition: Callable[[], bool], timeout: float, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(step)
    QApplication.processEvents()
    return condition()


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def last_message() -> tuple[str, str, str] | None:
    return MESSAGES[-1] if MESSAGES else None


def run(screens_dir: Path | None) -> int:
    temp = Path(tempfile.mkdtemp(prefix="tm_e2e_"))
    config_file = temp / "config.yaml"
    for module in (main_window_module, settings_module):
        module.CONFIG_FILE = config_file
    patch_dialogs()
    setup_logging(LOG_FILE)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE_FILE.read_text(encoding="utf-8"))
    config = AppConfig(output_dir=str(temp / "output"))
    window = MainWindow(config)
    window.show()
    panel = window.dashboard.video_panel

    def shot(name: str) -> None:
        if screens_dir:
            screens_dir.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(screens_dir / f"{name}.png"))

    try:
        # 1. Initial state -------------------------------------------------
        check("01 initial state IDLE", window.state == "IDLE" and panel.start_button.isEnabled()
              and not panel.pause_button.isEnabled() and not panel.stop_button.isEnabled())

        # 2. Full run: START → RUNNING → frames/events → PAUSE/RESUME → finish
        panel.start_button.click()
        check("02 START → STARTING (loading)", window.state == "STARTING" and not panel.start_button.isEnabled())
        check("03 RUNNING after model load", wait_until(lambda: window.state == "RUNNING", 60))
        check("04 settings locked while running", not window.settings_panel.save_button.isEnabled())
        ok = wait_until(lambda: panel.video_view.frames_shown > 20, 30)
        check("05 video frames displayed in Qt", ok, f"{panel.video_view.frames_shown} frames")
        ok = wait_until(lambda: window.dashboard.recent_events.total_count() > 0, 30)
        check("06 events appear in live log", ok, f"{window.dashboard.recent_events.total_count()} events")
        system = window.dashboard.system_panel.values
        check("07 system info: FPS + GPU device", system["fps"].text() not in ("—", "0.0")
              and "cuda" in system["device"].text(), f"fps={system['fps'].text()} device={system['device'].text()}")
        shot("dashboard_running")

        panel.pause_button.click()
        wait_until(lambda: False, 0.6)
        frames_at_pause = panel.video_view.frames_shown
        wait_until(lambda: False, 1.2)
        check("08 PAUSE freezes processing", window.state == "PAUSED"
              and panel.video_view.frames_shown - frames_at_pause <= 1 and panel.pause_button.text().endswith("RESUME"))
        panel.pause_button.click()
        ok = wait_until(lambda: panel.video_view.frames_shown > frames_at_pause + 5, 10)
        check("09 RESUME continues", ok and window.state == "RUNNING")

        run_started = time.monotonic()
        ok = wait_until(lambda: window.state == "FINISHED", 180)
        frames = (window.report_panel._summary or {}).get("frames_processed", 0)
        check("10 video processed to the end → FINISHED", ok and frames == 538,
              f"{frames} frames, remaining part took {time.monotonic() - run_started:.1f}s")
        info = last_message()
        check("11 completion message shown", bool(info and info[1] == "Hoàn tất xử lý"))
        cards = window.dashboard.cards.cards
        total = int(cards["total_vehicles"].value_text())
        check("12 dashboard totals consistent", total == int(cards["total_in"].value_text()) + int(cards["total_out"].value_text())
              and total > 0, f"total={total} violations={cards['speed_violations'].value_text()}")
        summary = window.report_panel._summary or {}
        outputs = summary.get("outputs", {})
        files_ok = all(outputs.get(k) and resolve_path(outputs[k]).is_file() for k in ("video", "json", "csv"))
        check("13 outputs written (mp4 + json + csv)", files_ok, str({k: outputs.get(k) for k in ("video", "json", "csv")}))
        report = json.loads(resolve_path(outputs["json"]).read_text(encoding="utf-8")) if files_ok else {}
        check("14 report numbers = dashboard", report.get("total_vehicles") == total
              and report.get("speed_violations", {}).get("count") == int(cards["speed_violations"].value_text()))
        check("15 event log page has same events", window.log_panel.log_view.total_count() == len(report.get("events", [])),
              f"{window.log_panel.log_view.total_count()} rows")
        check("16 violation snapshots saved", len(outputs.get("snapshots", [])) == report.get("speed_violations", {}).get("count"))
        window.show_page("reports")
        shot("reports")

        # 3. Reports buttons ----------------------------------------------
        json_path = window.report_panel.export_json()
        csv_path = window.report_panel.export_csv()
        check("17 Export JSON / CSV buttons", bool(json_path and json_path.is_file() and csv_path and csv_path.is_file()))
        window.report_panel.open_folder_button.click()
        check("18 Open Output Folder", last_message()[0] == "open_folder")

        # 4. Event log filters --------------------------------------------
        window.show_page("logs")
        log = window.log_panel
        log.type_filter.setCurrentIndex(log.type_filter.findData("SPEED_VIOLATION"))
        only_violations = log.log_view.visible_count()
        first_id = report["events"][0]["tracker_id"] if report.get("events") else 0
        log.type_filter.setCurrentIndex(0)
        log.id_filter.setText(str(first_id))
        by_id = log.log_view.visible_count()
        check("19 log filters (type / track id)", only_violations == report["speed_violations"]["count"] and by_id >= 1,
              f"violations={only_violations} id#{first_id}={by_id}")
        shot("event_logs")
        log.id_filter.clear()

        # 5. START again then STOP -----------------------------------------
        window.show_page("dashboard")
        panel.start_button.click()
        wait_until(lambda: window.state == "RUNNING" and panel.video_view.frames_shown > 0, 60)
        wait_until(lambda: False, 2.0)
        panel.stop_button.click()
        ok = wait_until(lambda: window.state == "FINISHED", 30)
        summary2 = window.report_panel._summary or {}
        check("20 STOP mid-video → status stopped + outputs", ok and summary2.get("status") == "stopped"
              and bool(summary2.get("outputs", {}).get("json")), f"frames={summary2.get('frames_processed')}")
        check("21 new session resets counters", summary2.get("session_id") != summary.get("session_id"))

        # 6. Configuration page --------------------------------------------
        window.show_page("config")
        settings = window.settings_panel
        settings.conf_slider.setValue(45)
        settings.limit_spin.setValue(100)
        settings.line_slider.setValue(60)
        check("22 form edits mark dirty", settings.is_dirty and abs(settings.conf_spin.value() - 0.45) < 1e-6)
        settings.save_button.click()
        saved = AppConfig.load(config_file) if config_file.is_file() else None
        check("23 SAVE CONFIG writes config.yaml + applies", bool(saved and saved.confidence_threshold == 0.45
              and saved.speed_limit_kmh == 100 and saved.line_y_ratio == 0.6)
              and window.dashboard.system_panel.values["confidence"].text() == "0.45")
        shot("configuration")
        settings.point_spins[2][0].setValue(-600)  # C left of D → not convex
        settings.save_button.click()
        msg = last_message()
        check("24 invalid perspective rejected", msg[0] == "message" and "perspective" in msg[2]
              and AppConfig.load(config_file).perspective_source[2][0] != -600)
        settings.set_config(window.config)
        settings.model_combo.setCurrentText("models/not_here.pt")
        settings.save_button.click()
        check("25 missing model rejected on save", "model_path" in last_message()[2])
        settings.set_config(window.config)
        settings.reset_button.click()
        check("26 RESET DEFAULT", AppConfig.load(config_file).confidence_threshold == 0.30
              and window.config.speed_limit_kmh == 80)

        # 7. Video Source page ----------------------------------------------
        window.show_page("source")
        source = window.source_panel
        source.test_button.click()
        ok = wait_until(lambda: source.probe_status.text().startswith(("✓", "✕")), 20)
        check("27 TEST SOURCE (background probe)", ok and source.info_values["resolution"].text() == "3840 × 2160")
        source._radios["rtsp"].setChecked(True)
        source.rtsp_edit.setText("http://not-rtsp")
        source.apply_button.click()
        check("28 invalid RTSP URL rejected", "RTSP" in last_message()[2] and window.config.source_type == "file")
        source._radios["camera"].setChecked(True)
        source.camera_spin.setValue(7)
        source.apply_button.click()
        check("29 APPLY camera source", window.config.source_type == "camera" and window.config.source_uri == "7")
        shot("video_source")

        # 8. Error flows (must not crash) -----------------------------------
        window.show_page("dashboard")
        panel.start_button.click()
        ok = wait_until(lambda: window.state == "ERROR", 30)
        check("30 camera not connected → ERROR dialog", ok and last_message()[1] == "Camera không kết nối")
        source._radios["file"].setChecked(True)
        source.file_edit.setText("data/samples/vehicles.mp4")
        source.apply_button.click()
        bad = window.config.copy()
        bad.model_path = "models/missing.pt"
        window._apply_config(bad)
        panel.start_button.click()
        ok = wait_until(lambda: window.state == "ERROR", 30)
        check("31 missing model → ERROR dialog", ok and last_message()[1] == "Không load được model")
        missing_video = window.config.copy()
        missing_video.model_path, missing_video.source_uri = "models/yolov8n.pt", "data/input/missing.mp4"
        window._apply_config(missing_video)
        panel.start_button.click()
        ok = wait_until(lambda: window.state == "ERROR", 30)
        check("32 missing video → ERROR dialog", ok and last_message()[1] == "Video không tồn tại")
        check("33 app still responsive after errors", panel.start_button.isEnabled())

        # 9. Close while running --------------------------------------------
        good = window.config.copy()
        good.source_uri = "data/samples/vehicles.mp4"
        window._apply_config(good)
        panel.start_button.click()
        wait_until(lambda: window.state == "RUNNING", 60)
        started = time.monotonic()
        window.close()
        check("34 close during processing stops worker cleanly",
              window._thread is None or window._thread.isFinished(), f"{time.monotonic() - started:.1f}s")
    finally:
        app.processEvents()
        shutil.rmtree(temp, ignore_errors=True)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print("  FAILED:", name)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screens", type=Path, nargs="?", const=PROJECT_ROOT / "data" / "output" / "snapshots" / "gui",
                        help="Save screenshots of each page")
    args = parser.parse_args()
    return run(args.screens)


if __name__ == "__main__":
    sys.exit(main())
