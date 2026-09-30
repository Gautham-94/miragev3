"""Per-object-type tracker tuning table.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 5.3, 5.4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackerTuning:
    r: float  # Kalman measurement-noise multiplier
    q: float  # Kalman process noise
    distance_threshold: float


DEFAULT_TUNING = TrackerTuning(r=3.4, q=0.03, distance_threshold=2.5)

PER_LABEL_TUNING: dict[str, TrackerTuning] = {
    "car": TrackerTuning(r=3.4, q=0.03, distance_threshold=2.5),
    "license_plate": TrackerTuning(r=2.5, q=0.05, distance_threshold=3.75),
    # Wildlife moves less predictably frame-to-frame than vehicles (irregular gait,
    # partial occlusion by vegetation/water causing detection gaps), so the vehicle-tuned
    # default distance_threshold was too tight and split a single animal into multiple
    # tracks across a brief detection gap. Empirically verified against real footage:
    # threshold needs to clear ~7.5-8.0 to survive a 2-frame gap; 9.0 gives margin.
    "animal": TrackerTuning(r=3.4, q=0.03, distance_threshold=9.0),
}


def tuning_for_label(label: str) -> TrackerTuning:
    return PER_LABEL_TUNING.get(label, DEFAULT_TUNING)


def min_initialized(fps: int) -> int:
    return max(fps // 2, 2)


def max_disappeared(fps: int) -> int:
    return fps * 5


def stationary_threshold_frames(fps: int) -> int:
    return fps * 10
