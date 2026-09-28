"""Frame annotation: boxes, labels, traces, counting line, speed zone, overlay."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

COLOR_ACCENT = sv.Color.from_hex("#D9A441")
COLOR_VIOLATION = sv.Color.from_hex("#E05A5A")
COLOR_IN = (123, 174, 79)  # BGR of #4FAE7B
COLOR_OUT = (213, 155, 91)  # BGR of #5B9BD5
COLOR_WHITE = (235, 235, 235)
TRACK_PALETTE = sv.ColorPalette.from_hex(
    ["#5B9BD5", "#6FBF8E", "#D9A441", "#B58AD6", "#4FB3BF", "#D98A5B", "#9FB2CC", "#C2C46A", "#D97FA0"]
)


@dataclass(frozen=True)
class OverlayStats:
    """Numbers drawn in the dashboard overlay."""

    total_in: int
    total_out: int
    violations: int
    fps: float


class AnnotationRenderer:
    """Draw every annotation required by the specification on a BGR frame."""

    def __init__(self, frame_size: tuple[int, int], fps: float) -> None:
        self._thickness = sv.calculate_optimal_line_thickness(resolution_wh=frame_size)
        self._text_scale = sv.calculate_optimal_text_scale(resolution_wh=frame_size)
        self._text_thickness = max(1, self._thickness // 2)
        self._box = sv.BoxAnnotator(
            color=TRACK_PALETTE, thickness=self._thickness, color_lookup=sv.ColorLookup.TRACK
        )
        self._label = sv.LabelAnnotator(
            color=TRACK_PALETTE,
            text_color=sv.Color.BLACK,
            text_scale=self._text_scale,
            text_thickness=self._text_thickness,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self._violation_box = sv.BoxAnnotator(color=COLOR_VIOLATION, thickness=self._thickness + 1)
        self._violation_label = sv.LabelAnnotator(
            color=COLOR_VIOLATION,
            text_color=sv.Color.WHITE,
            text_scale=self._text_scale,
            text_thickness=self._text_thickness,
            text_position=sv.Position.TOP_LEFT,
        )
        self._trace = sv.TraceAnnotator(
            color=TRACK_PALETTE,
            position=sv.Position.BOTTOM_CENTER,
            trace_length=max(10, int(fps * 2)),
            thickness=self._thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )

    def render(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list[str],
        violation_mask: np.ndarray,
        line_endpoints: tuple[tuple[int, int], tuple[int, int]],
        speed_polygon: np.ndarray,
        stats: OverlayStats,
    ) -> np.ndarray:
        """Return an annotated copy of ``frame``."""
        scene = frame.copy()
        sv.draw_polygon(scene, speed_polygon.astype(int), COLOR_ACCENT, max(1, self._thickness // 2))
        if len(detections) > 0:
            scene = self._trace.annotate(scene, detections)
            normal = ~violation_mask
            if normal.any():
                scene = self._box.annotate(scene, detections[normal])
                scene = self._label.annotate(scene, detections[normal], _select(labels, normal))
            if violation_mask.any():
                scene = self._violation_box.annotate(scene, detections[violation_mask])
                scene = self._violation_label.annotate(
                    scene, detections[violation_mask], _select(labels, violation_mask)
                )
        self._draw_counting_line(scene, line_endpoints, stats)
        self._draw_overlay(scene, stats)
        return scene

    def _draw_counting_line(
        self,
        scene: np.ndarray,
        endpoints: tuple[tuple[int, int], tuple[int, int]],
        stats: OverlayStats,
    ) -> None:
        (x0, y), (x1, _) = endpoints
        cv2.line(scene, (x0, y), (x1, y), COLOR_ACCENT.as_bgr(), self._thickness + 1, cv2.LINE_AA)
        right = x1 - self._thickness * 6
        for text, color in ((f"OUT {stats.total_out}", COLOR_OUT), (f"IN {stats.total_in}", COLOR_IN)):
            right = self._text_box(scene, text, right, y, color, align_right=True) - self._thickness * 4

    def _draw_overlay(self, scene: np.ndarray, stats: OverlayStats) -> None:
        rows = [
            ("IN", str(stats.total_in)),
            ("OUT", str(stats.total_out)),
            ("TOTAL", str(stats.total_in + stats.total_out)),
            ("VIOLATIONS", str(stats.violations)),
            ("FPS", f"{stats.fps:.1f}"),
        ]
        scale = self._text_scale * 1.1
        line_height = int(32 * scale) + 6
        pad = int(14 * scale)
        width = int(300 * scale)
        height = line_height * len(rows) + pad
        region = scene[pad:pad + height, pad:pad + width]
        if region.size:
            scene[pad:pad + height, pad:pad + width] = (region * 0.4).astype(np.uint8)
        for index, (key, value) in enumerate(rows):
            baseline = pad + line_height * (index + 1)
            cv2.putText(scene, key, (pad * 2, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (160, 160, 160), self._text_thickness, cv2.LINE_AA)
            (value_w, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, self._text_thickness)
            cv2.putText(scene, value, (pad + width - pad - value_w, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, COLOR_WHITE, self._text_thickness, cv2.LINE_AA)

    def _text_box(
        self, scene: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], align_right: bool
    ) -> int:
        """Draw text on a filled box centred vertically on ``y``; return left x."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, self._text_scale, self._text_thickness)
        pad = max(4, int(text_h * 0.5))
        left = x - text_w - 2 * pad if align_right else x
        cv2.rectangle(scene, (left, y - text_h // 2 - pad), (left + text_w + 2 * pad, y + text_h // 2 + pad),
                      color, -1)
        cv2.putText(scene, text, (left + pad, y + text_h // 2), font, self._text_scale, (15, 15, 15),
                    self._text_thickness, cv2.LINE_AA)
        return left


def _select(labels: list[str], mask: np.ndarray) -> list[str]:
    return [label for label, keep in zip(labels, mask) if keep]


def build_label(tracker_id: int, vehicle_type: str, confidence: float | None,
                speed_kmh: float | None, violated: bool) -> str:
    """``#21 car 0.87 | 92 km/h`` (prefixed by ``OVER`` when violating)."""
    label = f"#{tracker_id} {vehicle_type}"
    if confidence is not None:
        label += f" {confidence:.2f}"
    if speed_kmh is not None:
        label += f" | {speed_kmh:.0f} km/h"
    return f"OVER {label}" if violated else label
