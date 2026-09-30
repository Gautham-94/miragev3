"""Custom distance function for the object tracker.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 5.2.
"""

from __future__ import annotations

import numpy as np


def tracking_distance(detection_points: np.ndarray, estimate_points: np.ndarray) -> float:
    """Both are [[x1,y1],[x2,y2]] point pairs (top-left, bottom-right corners).

    Uses the bottom-center point as the position anchor (approximates the object's
    ground-contact point) and normalizes position delta by the object's own current size
    (scale-invariant), plus a size-ratio-change term to disambiguate objects that overlap
    in position but differ meaningfully in scale.
    """
    est_w, est_h = np.diff(estimate_points, axis=0).flatten()
    det_w, det_h = np.diff(detection_points, axis=0).flatten()

    # Guard against degenerate zero-size boxes (shouldn't happen with real detections,
    # but avoids a ZeroDivisionError/inf if one ever slips through).
    est_w = est_w if est_w != 0 else 1e-6
    est_h = est_h if est_h != 0 else 1e-6

    detection_anchor = np.array([np.mean(detection_points[:, 0]), np.max(detection_points[:, 1])])
    estimate_anchor = np.array([np.mean(estimate_points[:, 0]), np.max(estimate_points[:, 1])])

    dx = (detection_anchor[0] - estimate_anchor[0]) / est_w
    dy = (detection_anchor[1] - estimate_anchor[1]) / est_h

    widths = sorted([est_w, det_w])
    heights = sorted([est_h, det_h])
    width_ratio = widths[1] / widths[0] - 1.0 if widths[0] != 0 else 0.0
    height_ratio = heights[1] / heights[0] - 1.0 if heights[0] != 0 else 0.0

    return float(np.linalg.norm([dx, dy, width_ratio, height_ratio]))


def norfair_distance(detection, tracked_object) -> float:
    """Adapter matching norfair's expected distance_function(Detection, TrackedObject)
    signature.
    """
    return tracking_distance(detection.points, tracked_object.estimate)


def box_to_points(box: tuple[float, float, float, float]) -> np.ndarray:
    """Converts an (x1,y1,x2,y2) box into norfair's expected Nx2 points array."""
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y2]], dtype=float)


def points_to_box(points: np.ndarray) -> tuple[float, float, float, float]:
    """norfair's Kalman filter tracks each corner point independently (see
    box_to_points) with no constraint that the "top-left" point stays left-of/above the
    "bottom-right" one -- under jitter or coasting (no real detection for a frame or
    two) the two corners can drift and cross each other, producing an inverted box
    (x2 < x1 and/or y2 < y1). Confirmed live: real stored Event.snapshot_boxes with
    negative widths/heights. Sorting here guarantees every box this tracker ever
    returns is geometrically valid, regardless of how the underlying points drifted.
    """
    (x1, y1), (x2, y2) = points
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    return x1, y1, x2, y2
