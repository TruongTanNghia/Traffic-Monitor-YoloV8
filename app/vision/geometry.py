"""Geometry helpers: quadrilateral validation, point-in-polygon, scaling."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from app.utils.exceptions import PerspectiveError

MIN_POLYGON_AREA_PX = 100.0


def validate_quadrilateral(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate perspective source points.

    Rules: exactly 4 points ``[x, y]``, convex, non-degenerate, no three collinear.
    Points may lie outside the frame (road edges often do).

    Returns:
        ``float32`` array of shape ``(4, 2)``.

    Raises:
        PerspectiveError: when a rule is violated.
    """
    try:
        array = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise PerspectiveError(f"Toạ độ perspective không phải số: {exc}") from exc
    if array.shape != (4, 2):
        raise PerspectiveError(
            f"perspective_source phải gồm đúng 4 điểm [x, y], nhận {len(array)} điểm."
        )
    if not np.all(np.isfinite(array)):
        raise PerspectiveError("perspective_source chứa giá trị không hữu hạn.")
    crosses = _edge_cross_products(array)
    if np.any(np.isclose(crosses, 0.0)):
        raise PerspectiveError("perspective_source có 3 điểm thẳng hàng.")
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        raise PerspectiveError(
            "perspective_source không tạo tứ giác lồi. Thứ tự đúng: trên-trái, trên-phải, dưới-phải, dưới-trái."
        )
    if abs(cv2.contourArea(array)) < MIN_POLYGON_AREA_PX:
        raise PerspectiveError("Vùng perspective quá nhỏ.")
    return array


def _edge_cross_products(polygon: np.ndarray) -> np.ndarray:
    """Cross product of consecutive edges for each vertex."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    next_edges = np.roll(edges, -1, axis=0)
    return edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask of ``points`` (N×2) lying inside or on ``polygon``."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    return np.array(
        [cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in points],
        dtype=bool,
    )


def occluded_bottom_mask(xyxy: np.ndarray, min_overlap_ratio: float = 0.3) -> np.ndarray:
    """Detect boxes whose bottom edge is hidden behind a vehicle in front.

    Box ``j`` occludes the bottom of box ``i`` when ``j`` is closer to the
    camera (lower ``y2``) and covers ``i``'s bottom edge with enough horizontal
    overlap. The bottom-center anchor of such boxes is not the real road
    contact point, so it must not be used for speed measurement.
    """
    count = len(xyxy)
    mask = np.zeros(count, dtype=bool)
    for i in range(count):
        x1, _, x2, y2 = xyxy[i]
        width = max(x2 - x1, 1.0)
        for j in range(count):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = xyxy[j]
            overlap = min(x2, ox2) - max(x1, ox1)
            if oy2 > y2 and oy1 < y2 and overlap / width >= min_overlap_ratio:
                mask[i] = True
                break
    return mask


def scale_points(
    points: np.ndarray, from_size: Sequence[float], to_size: Sequence[float]
) -> np.ndarray:
    """Rescale points calibrated on ``from_size`` (w, h) to ``to_size`` (w, h)."""
    factor = np.array(
        [to_size[0] / from_size[0], to_size[1] / from_size[1]], dtype=np.float32
    )
    return (np.asarray(points, dtype=np.float32) * factor).astype(np.float32)


def horizontal_line_y(frame_height: int, ratio: float) -> int:
    """Pixel row of a horizontal line placed at ``ratio`` of the height."""
    return int(round(frame_height * min(max(ratio, 0.0), 1.0)))
