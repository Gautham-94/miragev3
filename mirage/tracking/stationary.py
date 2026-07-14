"""Stationary vs. active classification.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 5.7.
"""

from __future__ import annotations

from collections import deque

import numpy as np

Box = tuple[float, float, float, float]

DEFAULT_HISTORY_SIZE = 10
LOW_PERCENTILE = 15
HIGH_PERCENTILE = 85

# IoU thresholds: asymmetric depending on current state, to prevent boundary oscillation.
ACTIVE_IOU_THRESHOLD = 0.2  # below this IoU (vs. previous trimmed box) -> flag active
STATIONARY_TO_ACTIVE_IOU_THRESHOLD = 0.9  # once stationary, need IoU below this to flip back to active
ACTIVE_TO_STATIONARY_IOU_THRESHOLD = 0.6  # once active, need IoU above this to flip to stationary


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class StationaryClassifier:
    def __init__(self, threshold_frames: int, max_frames: int | None = None, history_size: int = DEFAULT_HISTORY_SIZE) -> None:
        self.threshold_frames = threshold_frames
        self.max_frames = max_frames
        self.history_size = history_size
        self._xmins: deque[float] = deque(maxlen=history_size)
        self._ymins: deque[float] = deque(maxlen=history_size)
        self._xmaxs: deque[float] = deque(maxlen=history_size)
        self._ymaxs: deque[float] = deque(maxlen=history_size)
        self._previous_trimmed_box: Box | None = None
        self.motionless_count = 0
        self.is_active = True

    def trimmed_box(self) -> Box | None:
        if not self._xmins:
            return None
        return (
            float(np.percentile(self._xmins, LOW_PERCENTILE)),
            float(np.percentile(self._ymins, LOW_PERCENTILE)),
            float(np.percentile(self._xmaxs, HIGH_PERCENTILE)),
            float(np.percentile(self._ymaxs, HIGH_PERCENTILE)),
        )

    def update(self, box: Box) -> None:
        # NOTE: comparing the CURRENT raw single-frame box against the rolling trimmed
        # (percentile-smoothed) box -- not trimmed-vs-previous-trimmed -- is the
        # comparison that actually distinguishes sustained real motion from single-frame
        # jitter. A sliding window's 15th/85th percentile bounds track a steadily
        # translating box almost exactly (each successive trimmed box is only a few
        # samples different from the last), so two consecutive TRIMMED boxes stay at
        # high IoU with each other even under continuous real motion -- that comparison
        # would flag real motion as "stationary" almost immediately. Comparing the fresh
        # raw box against the smoothed history box directly measures "has this frame's
        # position drifted away from the established position," which behaves correctly
        # for both cases: jitter (raw box stays inside the smoothed history's high-IoU
        # range) and real motion (raw box quickly diverges from a smoothed box anchored
        # to older, now-stale positions).
        x1, y1, x2, y2 = box
        current_trimmed = self.trimmed_box()

        self._xmins.append(x1)
        self._ymins.append(y1)
        self._xmaxs.append(x2)
        self._ymaxs.append(y2)

        if current_trimmed is None:
            self._previous_trimmed_box = self.trimmed_box()
            return

        overlap = iou(box, current_trimmed)

        if self.is_active:
            became_stationary = overlap >= ACTIVE_TO_STATIONARY_IOU_THRESHOLD
        else:
            became_stationary = overlap >= STATIONARY_TO_ACTIVE_IOU_THRESHOLD

        if overlap < ACTIVE_IOU_THRESHOLD:
            self.is_active = True
            self.motionless_count = 0
        elif became_stationary:
            self.is_active = False
            self.motionless_count += 1
        elif not self.is_active:
            self.motionless_count += 1
        else:
            self.motionless_count = 0

        self._previous_trimmed_box = self.trimmed_box()

    def is_stationary(self) -> bool:
        return self.motionless_count > self.threshold_frames

    def is_expired(self) -> bool:
        if self.max_frames is None:
            return False
        return (self.motionless_count - self.threshold_frames) > self.max_frames
