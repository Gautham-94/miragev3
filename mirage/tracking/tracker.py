"""ObjectTracker: wraps norfair's Kalman-filter tracker, one instance per label.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 5.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

from norfair.filter import OptimizedKalmanFilterFactory
from norfair.tracker import Detection, Tracker

from mirage.tracking.config import max_disappeared, min_initialized, tuning_for_label
from mirage.tracking.distance import box_to_points, norfair_distance, points_to_box
from mirage.tracking.stationary import StationaryClassifier

Box = tuple[float, float, float, float]


@dataclass
class TrackedObjectState:
    id: str
    label: str
    box: Box
    score: float
    stationary: StationaryClassifier
    frame_time: float = 0.0
    # Mirrors this object's ObjectLifecycle.is_false_positive (see
    # mirage.tracking.lifecycle), which otherwise lives ONLY inside the per-camera
    # CameraTracker process and never reaches the main process. EventProcessor and
    # ReviewSegmentMaintainer run in the main process and previously had no way to see
    # this status at all -- their "get_lifecycle" callback was permanently stubbed to
    # always return None (see mirage.app.MirageApp._get_lifecycle_stub's old docstring),
    # which meant false_positive was only ever written once at event creation (always
    # True) and review segments never excluded false positives either. Carrying this
    # flag on the plain, already-cross-process TrackedObjectState instead fixes both.
    is_false_positive: bool = True


def _new_id(frame_time: float) -> str:
    rand6 = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{frame_time}-{rand6}"


class ObjectTracker:
    def __init__(self, fps: int, stationary_max_frames: dict[str, int] | None = None) -> None:
        self.fps = fps
        self.stationary_max_frames = stationary_max_frames or {}
        self._trackers_by_label: dict[str, Tracker] = {}
        self._track_id_map: dict[int, str] = {}  # norfair global_id -> our own id
        self._states: dict[str, TrackedObjectState] = {}  # our id -> state

    def _get_or_create_tracker(self, label: str) -> Tracker:
        tracker = self._trackers_by_label.get(label)
        if tracker is None:
            tuning = tuning_for_label(label)
            tracker = Tracker(
                distance_function=norfair_distance,
                distance_threshold=tuning.distance_threshold,
                hit_counter_max=max_disappeared(self.fps),
                initialization_delay=min_initialized(self.fps),
                filter_factory=OptimizedKalmanFilterFactory(R=tuning.r, Q=tuning.q),
            )
            self._trackers_by_label[label] = tracker
        return tracker

    def update(self, frame_time: float, detections: list[tuple[str, float, Box]]) -> dict[str, TrackedObjectState]:
        """detections: list of (label, score, box) tuples for this frame. Returns the
        full current set of confirmed tracked-object states (our own id -> state).
        """
        by_label: dict[str, list[tuple[float, Box]]] = {}
        for label, score, box in detections:
            by_label.setdefault(label, []).append((score, box))

        active_ids_this_frame: set[str] = set()

        # Advance every existing label's tracker, even ones with zero detections this
        # frame, so hit_counter aging/expiry happens correctly (spec section 5.4).
        all_labels = set(self._trackers_by_label.keys()) | set(by_label.keys())
        for label in all_labels:
            tracker = self._get_or_create_tracker(label)
            label_detections = by_label.get(label, [])
            norfair_detections = [
                Detection(points=box_to_points(box), scores=None, data={"score": score})
                for score, box in label_detections
            ]
            tracked_objects = tracker.update(detections=norfair_detections)

            for obj in tracked_objects:
                if not obj.hit_counter_is_positive:
                    continue
                our_id = self._track_id_map.get(obj.global_id)
                score = obj.last_detection.data.get("score", 0.0) if obj.last_detection is not None else 0.0
                box = points_to_box(obj.estimate)

                if our_id is None:
                    our_id = _new_id(frame_time)
                    self._track_id_map[obj.global_id] = our_id
                    stationary = StationaryClassifier(
                        threshold_frames=self._stationary_threshold(),
                        max_frames=self.stationary_max_frames.get(label),
                    )
                    self._states[our_id] = TrackedObjectState(
                        id=our_id, label=label, box=box, score=score, stationary=stationary, frame_time=frame_time
                    )
                else:
                    state = self._states[our_id]
                    state.box = box
                    state.score = score
                    state.frame_time = frame_time

                self._states[our_id].stationary.update(box)
                active_ids_this_frame.add(our_id)

                if self._states[our_id].stationary.is_expired():
                    self._deregister(our_id, obj.global_id)
                    active_ids_this_frame.discard(our_id)

        # Any id we were tracking that norfair no longer reports (hit_counter expired)
        # must be removed from our own state too.
        for our_id in list(self._states.keys()):
            if our_id not in active_ids_this_frame:
                gid = next((g for g, oid in self._track_id_map.items() if oid == our_id), None)
                self._deregister(our_id, gid)

        return dict(self._states)

    def current_states(self) -> dict[str, TrackedObjectState]:
        """Public accessor for the tracker's current confirmed tracked-object states,
        without needing to call update() -- used by region selection (spec section 4) to
        build regions around existing tracks before this frame's detections are known.
        """
        return dict(self._states)

    def _stationary_threshold(self) -> int:
        from mirage.tracking.config import stationary_threshold_frames

        return stationary_threshold_frames(self.fps)

    def _deregister(self, our_id: str, global_id: int | None) -> None:
        self._states.pop(our_id, None)
        if global_id is not None:
            self._track_id_map.pop(global_id, None)
