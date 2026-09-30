"""CameraTracker: the per-camera OS process that runs motion detection, region
selection, object detection, and tracking.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.1 (CameraTracker), section 7
(per-frame orchestration loop).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_module

import cv2

from mirage.config.schema import CameraConfig, DetectorInstanceConfig
from mirage.const import PROCESS_PRIORITY_HIGH, resolve_model_path
from mirage.detection.labelmap import load_labels
from mirage.detection.remote import RemoteObjectDetector
from mirage.detection.tensor import yuv420_to_bgr
from mirage.logging_bus import LogEvent
from mirage.motion.detector import MotionDetector
from mirage.ptz.poller import PtzPoller
from mirage.tracking.orchestration import CameraOrchestrator
from mirage.tracking.tracker import ObjectTracker, TrackedObjectState
from mirage.util.shm import SharedMemoryFrameManager

logger = logging.getLogger(__name__)

FRAME_QUEUE_POLL_TIMEOUT_SECONDS = 1.0

# JPEG quality for the plain (no box yet) full-frame snapshot passed alongside a
# detected-frame result -- see _maybe_encode_frame's docstring for why this is only
# encoded conditionally, not on every single frame.
_SNAPSHOT_JPEG_QUALITY = 85


def _has_qualifying_object(tracked_objects: dict[str, TrackedObjectState]) -> bool:
    """Cheap proxy for "might EventProcessor/ReviewSegmentMaintainer want a snapshot
    of this frame" -- the real qualification logic (label in alert/detection lists,
    not stationary, etc, see mirage.events.review.qualifies_for_review/
    classify_severity) lives in the main process and needs CameraConfig.review, which
    this per-camera tracker process doesn't need to duplicate. Gate 1 alone
    (is_false_positive is False) is a cheap, correct OVER-approximation: every object
    that would actually qualify for a snapshot must have passed Gate 1 first, so
    checking Gate 1 here can only cause some unnecessary encodes (an ongoing event's
    later frames, or a confirmed object whose label isn't alert/detection-worthy) --
    never a missed one.
    """
    return any(not state.is_false_positive for state in tracked_objects.values())


def _maybe_encode_frame(yuv_frame, frame_shape: tuple[int, int], tracked_objects: dict[str, TrackedObjectState]) -> bytes | None:
    """Encodes a plain (no box burned in yet -- that happens later, once
    EventProcessor/ReviewSegmentMaintainer actually decide a snapshot is needed, see
    their own _on_start/_start_segment) full-frame JPEG from the SAME SHM frame this
    tracker just ran detection on -- MUST happen here, synchronously, while the frame
    is still live: mirage.util.shm's ring buffer is finite-depth and this same slot
    will be reused by a newer frame shortly after camera_tracker_main's own
    frame_manager.close() call below, so any later, deferred attempt to re-fetch "the
    frame this detection came from" (e.g. a live go2rtc call, as EventProcessor used
    to do) can easily end up describing a DIFFERENT, later moment than when
    state.box was actually computed -- a real bug this fixes (a moving person's box
    landing nowhere near their actual position in the mismatched snapshot).

    Only encodes at all if _has_qualifying_object is True, to avoid the real
    (measured -- JPEG-encoding a 640x480 frame is not free) cost of encoding on every
    single frame at 5-10fps/camera when the overwhelming majority of frames have no
    activity EventProcessor/ReviewSegmentMaintainer would ever want a snapshot for.
    """
    if not _has_qualifying_object(tracked_objects):
        return None
    bgr = yuv420_to_bgr(yuv_frame, frame_shape)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _SNAPSHOT_JPEG_QUALITY])
    if not ok:
        return None
    return encoded.tobytes()


def camera_tracker_main(
    camera: CameraConfig,
    detector_config: DetectorInstanceConfig,
    detection_queue,
    detector_sub_addr: str,
    frame_queue,
    detected_frames_queue,
    stop_event,
    activity_log_queue=None,
    verbose: bool = False,
) -> None:
    """Runs as the per-camera tracker OS process. Pulls (frame_name, frame_time) off
    frame_queue (populated by the CameraCapture process, see mirage.capture.capture),
    reads the actual pixel data from the shared frame ring, runs the full section 7
    per-frame pipeline, and publishes results to detected_frames_queue for the main
    process's result-consumer thread.
    """
    try:
        os.nice(PROCESS_PRIORITY_HIGH)
    except (AttributeError, PermissionError, OSError):
        pass

    # A separate OS process -- this basicConfig call is independent of whatever level
    # mirage/__main__.py's own call configured for the main process, so -v/--verbose
    # must be threaded all the way down through MirageApp -> CameraTracker -> here for
    # any logger.debug() call anywhere in this process's code (motion, region
    # selection, tracking/lifecycle, this file) to ever actually be emitted.
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    frame_manager = SharedMemoryFrameManager()

    labels = load_labels(resolve_model_path(detector_config.model.labelmap_path))
    remote_detector = RemoteObjectDetector(
        camera_name=camera.name,
        labelmap=labels,
        detection_queue=detection_queue,
        model_config=detector_config.model,
        detector_sub_addr=detector_sub_addr,
        frame_manager=frame_manager,
    )
    motion_detector = MotionDetector(frame_shape=camera.frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)

    # Mirage V3 PTZ hybrid scheduling: only spawn a PtzPoller (and pass a real
    # ptz_moving_fn into CameraOrchestrator) if this camera has PTZ explicitly enabled
    # -- camera.ptz.enabled defaults to False, so every existing/non-PTZ camera gets
    # ptz_moving_fn=None here, preserving the exact pre-V3 control flow byte-for-byte
    # (see CameraOrchestrator.__init__'s own docstring).
    ptz_poller: PtzPoller | None = None
    ptz_moving_fn = None
    if camera.ptz.enabled:
        ptz_poller = PtzPoller(camera.name, camera.ptz)
        ptz_poller.start()
        ptz_moving_fn = ptz_poller.is_moving

    log_fn = None
    if activity_log_queue is not None:
        def log_fn(category: str, message: str, camera_name: str) -> None:
            try:
                activity_log_queue.put_nowait(LogEvent(category=category, message=message, camera=camera_name))
            except queue_module.Full:
                pass  # log queue backpressure -- drop rather than block the tracker's hot loop

    orchestrator = CameraOrchestrator(
        camera, motion_detector, object_tracker, remote_detector, ptz_moving_fn=ptz_moving_fn, log_fn=log_fn,
    )

    logger.info("%s: tracker ready", camera.name)

    while not stop_event.is_set():
        try:
            frame_name, frame_time = frame_queue.get(True, FRAME_QUEUE_POLL_TIMEOUT_SECONDS)
        except queue_module.Empty:
            continue
        except (OSError, EOFError):
            break

        yuv_frame = frame_manager.get(frame_name, camera.frame_shape_yuv)
        if yuv_frame is None:
            continue

        try:
            result = orchestrator.process_frame(yuv_frame, frame_time)
        except Exception:
            logger.exception("%s: error processing frame %s", camera.name, frame_name)
            frame_manager.close(frame_name)
            continue

        # MUST happen before frame_manager.close() -- see _maybe_encode_frame's
        # docstring for why this can't be deferred to the main process.
        frame_jpeg = _maybe_encode_frame(yuv_frame, camera.frame_shape, result.tracked_objects)

        frame_manager.close(frame_name)

        try:
            detected_frames_queue.put(
                (
                    camera.name, frame_name, frame_time, result.tracked_objects,
                    result.motion_boxes, result.regions, frame_jpeg,
                ),
                block=False,
            )
        except queue_module.Full:
            pass  # backpressure: main process fell behind; drop this frame's publish

    remote_detector.close()
    if ptz_poller is not None:
        ptz_poller.stop()
    logger.info("%s: tracker stopped", camera.name)


class CameraTracker(mp.Process):
    def __init__(
        self,
        camera: CameraConfig,
        detector_config: DetectorInstanceConfig,
        detection_queue,
        detector_sub_addr: str,
        frame_queue,
        detected_frames_queue,
        stop_event,
        activity_log_queue=None,
        verbose: bool = False,
    ) -> None:
        super().__init__(name=f"tracker:{camera.name}")
        self.camera = camera
        self.detector_config = detector_config
        self.detection_queue = detection_queue
        self.detector_sub_addr = detector_sub_addr
        self.frame_queue = frame_queue
        self.detected_frames_queue = detected_frames_queue
        self.stop_event = stop_event
        self.activity_log_queue = activity_log_queue
        self.verbose = verbose

    def run(self) -> None:
        camera_tracker_main(
            self.camera,
            self.detector_config,
            self.detection_queue,
            self.detector_sub_addr,
            self.frame_queue,
            self.detected_frames_queue,
            self.stop_event,
            self.activity_log_queue,
            self.verbose,
        )
