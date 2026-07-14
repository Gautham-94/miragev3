"""Object lifecycle / event state machine: true-positive vs false-positive classification.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 6.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mirage.config.schema import ObjectFilterConfig

SCORE_HISTORY_SIZE = 10


@dataclass
class ObjectLifecycle:
    """Tracks the score history and false-positive state for one tracked object across
    its lifetime. Section 6.1: an object is a false positive until computed_score (median
    of score_history) first exceeds `threshold` -- once true, it can never revert.
    """

    label: str
    score_history: deque = field(default_factory=lambda: deque(maxlen=SCORE_HISTORY_SIZE))
    _false_positive: bool = True
    position_changes: int = 0
    start_time: float = 0.0
    end_time: float | None = None

    def record_score(self, score: float | None) -> None:
        """Pass None for a frame where the object wasn't actually re-detected (only
        Kalman-estimated) -- this appends 0.0, deliberately dragging the median down for
        intermittently-detected objects (spec section 6.1).
        """
        self.score_history.append(score if score is not None else 0.0)

    def computed_score(self) -> float:
        if not self.score_history:
            return 0.0
        sorted_scores = sorted(self.score_history)
        n = len(sorted_scores)
        mid = n // 2
        if n % 2 == 1:
            return sorted_scores[mid]
        return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

    def update_false_positive_status(self, filter_config: ObjectFilterConfig) -> None:
        if not self._false_positive:
            return  # sticky: once a true positive, always a true positive
        if self.computed_score() >= filter_config.threshold:
            self._false_positive = False

    @property
    def is_false_positive(self) -> bool:
        return self._false_positive

    def record_position_change(self) -> None:
        self.position_changes += 1


def is_object_filtered(
    label: str, score: float, box: tuple[float, float, float, float], frame_shape: tuple[int, int], filter_config: ObjectFilterConfig
) -> bool:
    """Section 6.2: applied at raw-detection time. Returns True if the detection should
    be dropped.
    """
    if score < filter_config.min_score:
        return True

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    ratio = width / max(1, height)

    min_area = filter_config.min_area
    max_area = filter_config.max_area
    if 0 < min_area < 1:
        min_area = min_area * frame_shape[0] * frame_shape[1]
    if 0 < max_area < 1:
        max_area = max_area * frame_shape[0] * frame_shape[1]

    if area < min_area or area > max_area:
        return True
    if ratio < filter_config.min_ratio or ratio > filter_config.max_ratio:
        return True

    return False


def should_save_snapshot(lifecycle: ObjectLifecycle, snapshots_enabled: bool) -> bool:
    """Section 6.5."""
    if lifecycle.is_false_positive:
        return False
    if not snapshots_enabled:
        return False
    if lifecycle.position_changes == 0:
        return False
    return True
