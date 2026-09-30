"""Per-frame orchestration: the exact control flow tying motion detection, region
selection, object detection, and tracking together for one camera.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mirage.config.schema import CameraConfig
from mirage.detection.remote import RemoteObjectDetector
from mirage.detection.tensor import create_tensor_input
from mirage.motion.detector import MotionDetector
from mirage.regions.reduce import RawDetection, reduce_detections
from mirage.regions.selection import build_regions, denormalize_box
from mirage.tracking.lifecycle import ObjectLifecycle, is_object_filtered
from mirage.tracking.tracker import ObjectTracker, TrackedObjectState

logger = logging.getLogger(__name__)

# TEMP diagnostic: which label to log Gate 0/Gate 1 confidence-gating decisions for.
# Added to investigate a reported miss (a porcupine never got boxed despite a leopard
# elsewhere in the same frame being detected normally) -- neither gate's per-detection
# raw score is normally persisted anywhere (confirmed: no DB column, no existing log
# line), so there was no way to tell "model never saw it" apart from "saw it, scored it,
# got gated out" after the fact. Narrow to one label to keep volume sane on an otherwise
# noisy per-frame/per-region hot path. Remove once the investigation is done.
DIAGNOSTIC_SCORE_LOGGING_LABEL = "animal"
# Gate 1 status is checked every frame a track is still alive; logging that unthrottled
# for a track that lingers unconfirmed for a while would flood the log. Matches the
# cadence mirage.events.processor already uses for its own throttled Event updates.
DIAGNOSTIC_SCORE_LOG_THROTTLE_SECONDS = 5.0

# Mirage V3 PTZ hybrid scheduling: while a PTZ camera is actively moving, direct-sample
# the whole frame to the detector at roughly this rate instead of the normal
# motion-gated region pipeline -- see CameraOrchestrator.process_frame's ptz_moving
# branch and the V3 architecture doc's "Sample at 1-2 FPS" PTZ design.
PTZ_SAMPLE_INTERVAL_SECONDS = 1.0 / 1.5


@dataclass
class FrameResult:
    camera_name: str
    frame_time: float
    tracked_objects: dict[str, TrackedObjectState]
    motion_boxes: list[tuple[int, int, int, int]]
    regions: list[tuple[int, int, int, int]]


class CameraOrchestrator:
    """Owns one camera's motion detector, object tracker, and per-object lifecycle
    state, and runs the exact section 7 per-frame control flow when fed a new frame.
    """

    def __init__(
        self,
        camera: CameraConfig,
        motion_detector: MotionDetector,
        object_tracker: ObjectTracker,
        remote_detector: RemoteObjectDetector,
        ptz_moving_fn: Callable[[], bool] | None = None,
        log_fn: Callable[[str, str, str], None] | None = None,
    ) -> None:
        """`ptz_moving_fn`, if given, is polled once per frame to decide whether this
        camera is currently mid-PTZ-move -- see process_frame's ptz_moving branch.
        None (the default, and the only option for every non-PTZ-enabled camera) means
        this camera never bypasses the normal motion-gated pipeline, preserving the
        exact pre-V3 control flow byte-for-byte (see mirage.tracking.camera_tracker.
        camera_tracker_main, which only passes a real closure when camera.ptz.enabled).

        `log_fn`, if given, is called as log_fn(category, message, camera_name) at a
        handful of edge-triggered milestones (motion start/stop, an object newly
        confirmed past Gate 1) for the Logs page -- see mirage.logging_bus. Deliberately
        NOT called every frame (motion fires at 5-10fps while movement is present, which
        would flood the log with noise); only on the transition. None (the default) is a
        complete no-op, so a non-instrumented caller (e.g. existing tests) is unaffected.
        """
        self.camera = camera
        self.motion_detector = motion_detector
        self.object_tracker = object_tracker
        self.remote_detector = remote_detector
        self.ptz_moving_fn = ptz_moving_fn
        self.log_fn = log_fn
        self._lifecycles: dict[str, ObjectLifecycle] = {}
        self._was_in_motion = False
        self._startup_scan_done = False
        # Last frame_time a direct whole-frame PTZ sample was taken -- see the
        # ptz_moving branch in process_frame. Reset to -inf so the very first moving
        # frame always samples immediately rather than waiting a full interval.
        self._last_ptz_sample_time = float("-inf")
        # Boxes from last frame's consolidated detections that did NOT (yet) belong to a
        # CONFIRMED tracked object -- i.e. norfair candidates still inside their
        # initialization_delay warm-up window. Without carrying these forward as regions,
        # a brand-new object would only ever get detected once (whichever frame first
        # scanned its location, e.g. the one-time startup scan or a single motion event)
        # and could never accumulate the >=2 consecutive matching detections
        # initialization_delay requires to be promoted to a confirmed track, since
        # nothing would trigger scanning that same location again next frame. Region
        # selection normally only re-scans CONFIRMED tracked-object regions plus fresh
        # motion -- this fills the gap for "recently seen, not yet confirmed" candidates.
        self._pending_candidate_boxes: list[tuple[int, int, int, int]] = []
        # TEMP diagnostic (see DIAGNOSTIC_SCORE_LOGGING_LABEL above): obj_id -> frame_time
        # of the last Gate 1 status log for that track, to throttle _update_lifecycles'
        # own logging without touching ObjectLifecycle's own fields.
        self._diagnostic_last_gate1_log: dict[str, float] = {}
        # TEMP diagnostic: last frame_time regions/motion coverage was logged, throttled
        # the same way -- a busy camera can have motion on nearly every frame, and this
        # runs regardless of the animal label filter (there's no label yet at this
        # stage), so it needs its own throttle to stay readable.
        self._diagnostic_last_regions_log: float = float("-inf")

    def process_frame(self, yuv_frame: np.ndarray, frame_time: float) -> FrameResult:
        frame_shape = self.camera.frame_shape  # (height, width) of luma plane
        luma = yuv_frame[: frame_shape[0], : frame_shape[1]]

        # Step: motion detection, unconditional every frame (spec section 7 step 3).
        motion_boxes = self.motion_detector.detect(luma)

        is_in_motion = bool(motion_boxes)
        if self.log_fn is not None and is_in_motion != self._was_in_motion:
            self.log_fn("motion", "motion started" if is_in_motion else "motion stopped", self.camera.name)
        self._was_in_motion = is_in_motion

        # Mirage V3 PTZ hybrid scheduling: while a PTZ camera is physically moving,
        # motion-gated region selection is meaningless -- a tracked-object box or
        # motion box computed before the pan/tilt started describes a location in the
        # OLD field of view, which no longer corresponds to anything in this frame.
        # Bypass build_regions() entirely and instead direct-sample the WHOLE frame to
        # the detector at ~1-2 FPS (PTZ_SAMPLE_INTERVAL_SECONDS), the same rate the V3
        # architecture doc specifies. Between samples, feed the tracker `detections=[]`
        # (same mechanism camera.detect.enabled=False already uses below) so norfair's
        # own max_disappeared ages out every pre-move track on its normal schedule --
        # deliberate, not a bug: an object mid-track when a move starts simply
        # disappears from tracked_objects the next frame, which EventProcessor's
        # existing start/update/end diff already closes out correctly with zero
        # changes needed. self.ptz_moving_fn is None for every non-PTZ camera, so this
        # branch never triggers and control falls straight through to the unchanged
        # motion-gated path below -- fixed cameras are completely unaffected.
        if self.ptz_moving_fn is not None and self.ptz_moving_fn():
            if frame_time - self._last_ptz_sample_time < PTZ_SAMPLE_INTERVAL_SECONDS:
                tracked = self.object_tracker.update(frame_time, detections=[])
                return FrameResult(self.camera.name, frame_time, tracked, motion_boxes=[], regions=[])
            self._last_ptz_sample_time = frame_time
            regions = [(0, 0, frame_shape[1], frame_shape[0])]
            return self._detect_and_track(yuv_frame, frame_time, frame_shape, motion_boxes, regions)

        if not self.camera.detect.enabled:
            # Step: detection disabled -- tracker still runs so existing tracks age out.
            tracked = self.object_tracker.update(frame_time, detections=[])
            return FrameResult(self.camera.name, frame_time, tracked, motion_boxes, regions=[])

        confirmed_boxes = [state.box for state in self.object_tracker.current_states().values()]
        # Include not-yet-confirmed candidate boxes from the previous frame alongside
        # confirmed tracked-object boxes, so a brand-new object keeps being re-scanned
        # long enough to actually reach norfair's initialization_delay and get promoted
        # (see the _pending_candidate_boxes docstring above).
        tracked_object_boxes = confirmed_boxes + self._pending_candidate_boxes

        min_region_size = min(self.remote_detector.model_config.width, self.remote_detector.model_config.height)
        is_calibrating = self.motion_detector.is_calibrating()
        regions = build_regions(
            tracked_object_boxes=[tuple(int(v) for v in box) for box in tracked_object_boxes],
            motion_boxes=motion_boxes,
            min_region_size=min_region_size,
            frame_shape=frame_shape,
            is_calibrating=is_calibrating,
            ptz_moving=False,
        )

        # TEMP diagnostic -- see DIAGNOSTIC_SCORE_LOGGING_LABEL's own comment for the
        # investigation this belongs to. Answers "was this location ever even sent to
        # the detector at all" one layer upstream of Gate 0/Gate 1: a motion box that
        # never produces a region (e.g. is_calibrating, or gets swallowed into an
        # existing tracked-object region's extent) means the detector never ran on that
        # part of the frame this frame, regardless of what it might have scored.
        if logger.isEnabledFor(logging.DEBUG) and (motion_boxes or regions):
            if frame_time - self._diagnostic_last_regions_log >= DIAGNOSTIC_SCORE_LOG_THROTTLE_SECONDS:
                self._diagnostic_last_regions_log = frame_time
                logger.debug(
                    "%s: motion_boxes=%s tracked_object_boxes=%d regions=%s calibrating=%s",
                    self.camera.name, motion_boxes, len(tracked_object_boxes), regions, is_calibrating,
                )

        if not regions and not self._startup_scan_done:
            regions = [(0, 0, frame_shape[1], frame_shape[0])]
        self._startup_scan_done = True

        return self._detect_and_track(yuv_frame, frame_time, frame_shape, motion_boxes, regions)

    def _detect_and_track(
        self,
        yuv_frame: np.ndarray,
        frame_time: float,
        frame_shape: tuple[int, int],
        motion_boxes: list[tuple[int, int, int, int]],
        regions: list[tuple[int, int, int, int]],
    ) -> FrameResult:
        """The shared "run the detector over `regions`, consolidate, feed the tracker"
        tail end of process_frame -- factored out so the PTZ direct-sample branch
        (a single whole-frame region) and the normal motion-gated branch (build_regions()'s
        clustered output) can share it without duplicating the detection loop.
        """
        raw_detections: list[RawDetection] = []
        for region in regions:
            tensor = create_tensor_input(yuv_frame, frame_shape, self.remote_detector.model_config, region)
            for label, score, norm_box in self.remote_detector.detect(tensor):
                full_box = denormalize_box(norm_box, region)
                x1 = max(0, full_box[0])
                y1 = max(0, full_box[1])
                x2 = min(frame_shape[1], full_box[2])
                y2 = min(frame_shape[0], full_box[3])
                if x2 <= x1 or y2 <= y1:
                    continue  # entirely outside frame bounds after clamping

                filter_config = self.camera.objects.filter_for(label)
                if not self.camera.objects.track_all and label not in self.camera.objects.track:
                    continue
                if is_object_filtered(label, score, (x1, y1, x2, y2), frame_shape, filter_config):
                    if label == DIAGNOSTIC_SCORE_LOGGING_LABEL:
                        logger.debug(
                            "%s: %s detection dropped at Gate 0 (score=%.3f, min_score=%.3f, box=%s, region=%s)",
                            self.camera.name, label, score, filter_config.min_score, (x1, y1, x2, y2), region,
                        )
                    continue

                raw_detections.append(RawDetection(label=label, score=score, box=(x1, y1, x2, y2), region=region))

        consolidated = reduce_detections(raw_detections, frame_shape)

        detections_for_tracker = [(d.label, d.score, d.box) for d in consolidated]
        tracked = self.object_tracker.update(frame_time, detections_for_tracker)

        self._update_lifecycles(tracked, frame_time)

        # Carry forward every consolidated detection box as a "pending candidate" region
        # for next frame -- confirmed tracked objects are already covered via
        # current_states() next call, so this specifically keeps unconfirmed candidates
        # (and, harmlessly, re-covers already-confirmed ones too) alive long enough to
        # reach initialization_delay.
        self._pending_candidate_boxes = [d.box for d in consolidated]

        return FrameResult(self.camera.name, frame_time, tracked, motion_boxes, regions)

    def _update_lifecycles(self, tracked: dict[str, TrackedObjectState], frame_time: float) -> None:
        for obj_id, state in tracked.items():
            lifecycle = self._lifecycles.get(obj_id)
            if lifecycle is None:
                lifecycle = ObjectLifecycle(label=state.label, start_time=frame_time)
                self._lifecycles[obj_id] = lifecycle

            was_detected_this_frame = state.frame_time == frame_time
            lifecycle.record_score(state.score if was_detected_this_frame else None)

            filter_config = self.camera.objects.filter_for(state.label)
            was_false_positive = lifecycle.is_false_positive
            lifecycle.update_false_positive_status(filter_config)
            # Carry the computed status back onto the plain state object, since
            # ObjectLifecycle itself never leaves this process (see
            # TrackedObjectState.is_false_positive's docstring).
            state.is_false_positive = lifecycle.is_false_positive

            if state.label == DIAGNOSTIC_SCORE_LOGGING_LABEL:
                if was_false_positive and not lifecycle.is_false_positive:
                    # Gate 1 transition -- sticky, happens at most once per track, so
                    # unconditional logging here can't flood.
                    logger.debug(
                        "%s: %s track %s CONFIRMED at Gate 1 (computed_score=%.3f, threshold=%.3f, frames=%d)",
                        self.camera.name, state.label, obj_id, lifecycle.computed_score(),
                        filter_config.threshold, len(lifecycle.score_history),
                    )
                elif lifecycle.is_false_positive:
                    last_logged = self._diagnostic_last_gate1_log.get(obj_id, float("-inf"))
                    if frame_time - last_logged >= DIAGNOSTIC_SCORE_LOG_THROTTLE_SECONDS:
                        self._diagnostic_last_gate1_log[obj_id] = frame_time
                        logger.debug(
                            "%s: %s track %s still UNCONFIRMED at Gate 1 (computed_score=%.3f, threshold=%.3f, "
                            "history=%s)",
                            self.camera.name, state.label, obj_id, lifecycle.computed_score(),
                            filter_config.threshold, list(lifecycle.score_history),
                        )

        # Evict lifecycles for objects the tracker no longer reports. Object ids are
        # never reused (see tracker.py's _new_id, timestamp-based), so once a track
        # ends there's no future frame that could still reference this obj_id -- and
        # nothing reads a lifecycle after its track ends (get_lifecycle has no
        # remaining callers; TrackedObjectState.is_false_positive is the only thing
        # that ever left this process, mirrored while the track was still live, see
        # its own docstring). Deleting outright here (rather than the old behavior of
        # just setting end_time and leaving the entry in place forever) fixes a real
        # unbounded-memory leak: every distinct object a busy camera has EVER tracked
        # used to accumulate a permanent dict entry for the lifetime of this
        # CameraTracker process, which does not restart on its own.
        for obj_id in list(self._lifecycles.keys()):
            if obj_id not in tracked:
                evicted = self._lifecycles[obj_id]
                if evicted.label == DIAGNOSTIC_SCORE_LOGGING_LABEL and evicted.is_false_positive:
                    # This track disappeared having NEVER cleared Gate 1 -- the "ghost
                    # track" case: something real was detected and tracked for a while
                    # (score_history below shows exactly how long/what it scored), but
                    # never confirmed, so it never got an Event, a snapshot box, or
                    # anything else visible.
                    logger.debug(
                        "%s: %s track %s EVICTED while still unconfirmed (computed_score=%.3f, "
                        "history=%s)",
                        self.camera.name, evicted.label, obj_id, evicted.computed_score(),
                        list(evicted.score_history),
                    )
                self._diagnostic_last_gate1_log.pop(obj_id, None)
                del self._lifecycles[obj_id]

    def get_lifecycle(self, obj_id: str) -> ObjectLifecycle | None:
        return self._lifecycles.get(obj_id)
