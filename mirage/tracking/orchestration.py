"""Per-frame orchestration: the exact control flow tying motion detection, region
selection, object detection, and tracking together for one camera.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
    ) -> None:
        self.camera = camera
        self.motion_detector = motion_detector
        self.object_tracker = object_tracker
        self.remote_detector = remote_detector
        self._lifecycles: dict[str, ObjectLifecycle] = {}
        self._startup_scan_done = False
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

    def process_frame(self, yuv_frame: np.ndarray, frame_time: float) -> FrameResult:
        frame_shape = self.camera.frame_shape  # (height, width) of luma plane
        luma = yuv_frame[: frame_shape[0], : frame_shape[1]]

        # Step: motion detection, unconditional every frame (spec section 7 step 3).
        motion_boxes = self.motion_detector.detect(luma)

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
        regions = build_regions(
            tracked_object_boxes=[tuple(int(v) for v in box) for box in tracked_object_boxes],
            motion_boxes=motion_boxes,
            min_region_size=min_region_size,
            frame_shape=frame_shape,
            is_calibrating=self.motion_detector.is_calibrating(),
            ptz_moving=False,
        )

        if not regions and not self._startup_scan_done:
            regions = [(0, 0, frame_shape[1], frame_shape[0])]
        self._startup_scan_done = True

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
                if label not in self.camera.objects.track:
                    continue
                if is_object_filtered(label, score, (x1, y1, x2, y2), frame_shape, filter_config):
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
            lifecycle.update_false_positive_status(filter_config)
            # Carry the computed status back onto the plain state object, since
            # ObjectLifecycle itself never leaves this process (see
            # TrackedObjectState.is_false_positive's docstring).
            state.is_false_positive = lifecycle.is_false_positive

        # Close out lifecycles for objects the tracker no longer reports.
        for obj_id in list(self._lifecycles.keys()):
            if obj_id not in tracked:
                self._lifecycles[obj_id].end_time = frame_time

    def get_lifecycle(self, obj_id: str) -> ObjectLifecycle | None:
        return self._lifecycles.get(obj_id)
