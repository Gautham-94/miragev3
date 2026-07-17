"""End-to-end test of CameraOrchestrator: real motion detector + real object tracker +
real detector process (loading the real yolov8n.onnx model) + real photo frames, wired
together via the exact section 7 per-frame control flow, confirming tracked objects with
the correct labels are actually produced across a sequence of frames.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    InputDType,
    ModelConfig,
    ObjectsConfig,
    PixelFormat,
    RecordConfig,
)
from mirage.detection.labelmap import load_labels
from mirage.detection.process import DetectorProcess
from mirage.detection.remote import RemoteObjectDetector
from mirage.ipc.zmq_pubsub import ZmqProxy
from mirage.motion.detector import MotionDetector
from mirage.tracking.orchestration import CameraOrchestrator
from mirage.tracking.tracker import ObjectTracker
from mirage.util.shm import SharedMemoryFrameManager

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
LABELMAP_PATH = Path(__file__).resolve().parent.parent / "models" / "coco_labelmap.txt"
BUS_IMAGE = Path(__file__).resolve().parent.parent / "media" / "bus.jpg"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and BUS_IMAGE.exists()),
    reason="yolov8n.onnx model or test image fixture not available",
)


@pytest.fixture
def short_ipc_dir():
    d = tempfile.mkdtemp(prefix="mrgorch", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _load_bus_as_yuv_frame() -> tuple[np.ndarray, tuple[int, int]]:
    bgr = cv2.imread(str(BUS_IMAGE))
    height, width = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    return yuv, (height, width)


def test_orchestrator_produces_tracked_objects_from_real_photo_sequence(short_ipc_dir):
    ctx = mp.get_context("spawn")
    yuv_frame, frame_shape = _load_bus_as_yuv_frame()
    height, width = frame_shape

    camera = CameraConfig(
        name="orch_test_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height, fps=5),
        record=RecordConfig(enabled=False),
        objects=ObjectsConfig(track=["person", "bus", "car"]),
    )

    detector_config = DetectorInstanceConfig(
        name="general",
        model=ModelConfig(
            width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
            model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        ),
        device="onnx_yolov8",
    )

    pub_addr = f"ipc://{short_ipc_dir}/pub"
    sub_addr = f"ipc://{short_ipc_dir}/sub"
    proxy = ZmqProxy(pub_addr, sub_addr)

    frame_manager = SharedMemoryFrameManager()
    detection_queue = ctx.Queue()
    stop_event = ctx.Event()

    detector_process = DetectorProcess(
        detector_config=detector_config, detection_queue=detection_queue,
        camera_names=[camera.name], detector_pub_addr=pub_addr, stop_event=stop_event,
    )

    labels = load_labels(str(LABELMAP_PATH))
    remote_detector = RemoteObjectDetector(
        camera_name=camera.name, labelmap=labels, detection_queue=detection_queue,
        model_config=detector_config.model, detector_sub_addr=sub_addr, frame_manager=frame_manager,
    )

    motion_detector = MotionDetector(frame_shape=frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)
    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, remote_detector)

    try:
        detector_process.start()

        results = []
        # Feed the same real photo repeatedly (simulating a static camera view) across
        # enough frames for norfair's initialization_delay (min_initialized) to confirm
        # tracks -- min_initialized(5) = 2, so a handful of frames is enough, but feed
        # more to also exercise steady-state tracking.
        for i in range(6):
            result = orchestrator.process_frame(yuv_frame, frame_time=float(i))
            results.append(result)

        final = results[-1]
        assert len(final.tracked_objects) > 0, "expected at least one confirmed tracked object"

        tracked_labels = {state.label for state in final.tracked_objects.values()}
        assert "bus" in tracked_labels or "person" in tracked_labels, (
            f"expected bus/person among tracked labels, got {tracked_labels}"
        )

        # Confirm object IDs are stable across the last two frames (persistent tracking,
        # not a fresh ID every frame).
        ids_second_last = set(results[-2].tracked_objects.keys())
        ids_last = set(results[-1].tracked_objects.keys())
        assert ids_second_last & ids_last, "expected at least one persistent id across frames"

        for result in results:
            assert result.camera_name == "orch_test_cam"

        # Regression check: state.is_false_positive must actually reflect the real
        # ObjectLifecycle computation, not just sit at its default (True) forever. This
        # is the field that carries false-positive status across the CameraTracker ->
        # main-process boundary (see TrackedObjectState.is_false_positive's docstring --
        # a real bug in production had this permanently stubbed to None/True instead).
        # The bus/person in this real repeated photo should have scored consistently
        # high enough, across enough frames, to have been promoted to a true positive by
        # the last frame.
        final_states = list(final.tracked_objects.values())
        assert any(not state.is_false_positive for state in final_states), (
            "expected at least one tracked object to have been promoted out of "
            "false-positive status by the final frame"
        )

    finally:
        stop_event.set()
        detector_process.join(timeout=15)
        if detector_process.is_alive():
            detector_process.terminate()
            detector_process.join(timeout=5)
        remote_detector.unlink()
        proxy.close()


def test_lifecycle_entries_are_evicted_once_a_track_ends():
    """Regression test for a real unbounded-memory leak (see
    OPTIMIZATION_OPPORTUNITIES.md item 2): CameraOrchestrator._lifecycles used to only
    ever mark a closed-out track's ObjectLifecycle with end_time, never actually remove
    it from the dict -- every distinct object id a busy camera ever saw accumulated a
    permanent entry for the life of the (long-running, not-expected-to-restart)
    CameraTracker process. Object ids are never reused (tracker.py's _new_id is
    timestamp-based), so once a track ends there's no future frame that could still
    reference it -- eviction is safe with no grace period needed.
    """
    from mirage.tracking.stationary import StationaryClassifier

    yuv_frame, frame_shape = _load_bus_as_yuv_frame()
    height, width = frame_shape

    camera = CameraConfig(
        name="leak_test_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height, fps=5),
        record=RecordConfig(enabled=False),
    )
    motion_detector = MotionDetector(frame_shape=frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)

    class _UnusedDetector:
        model_config = ModelConfig(width=320, height=320)

    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, _UnusedDetector())

    def _state(obj_id: str, frame_time: float) -> "object":
        from mirage.tracking.tracker import TrackedObjectState

        return TrackedObjectState(
            id=obj_id, label="person", box=(0, 0, 10, 10), score=0.9,
            stationary=StationaryClassifier(threshold_frames=50), frame_time=frame_time,
        )

    # Frame 1: obj1 is present -- a lifecycle entry is created for it.
    orchestrator._update_lifecycles({"obj1": _state("obj1", 1.0)}, frame_time=1.0)
    assert "obj1" in orchestrator._lifecycles

    # Frame 2: obj1 has disappeared (tracker no longer reports it) -- its lifecycle
    # entry must be EVICTED, not just marked with end_time and left in the dict.
    orchestrator._update_lifecycles({}, frame_time=2.0)
    assert "obj1" not in orchestrator._lifecycles
    assert orchestrator._lifecycles == {}


def test_lifecycles_dict_does_not_grow_across_many_distinct_short_lived_objects():
    """Simulates a busy camera where many distinct objects pass through one at a time
    (a fresh id each time, as real tracked objects do) -- the dict must stay bounded by
    however many objects are CURRENTLY tracked, not accumulate one entry per object
    ever seen.
    """
    from mirage.tracking.stationary import StationaryClassifier
    from mirage.tracking.tracker import TrackedObjectState

    yuv_frame, frame_shape = _load_bus_as_yuv_frame()
    height, width = frame_shape

    camera = CameraConfig(
        name="leak_test_cam2",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height, fps=5),
        record=RecordConfig(enabled=False),
    )
    motion_detector = MotionDetector(frame_shape=frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)

    class _UnusedDetector:
        model_config = ModelConfig(width=320, height=320)

    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, _UnusedDetector())

    for i in range(500):
        obj_id = f"obj{i}"
        state = TrackedObjectState(
            id=obj_id, label="person", box=(0, 0, 10, 10), score=0.9,
            stationary=StationaryClassifier(threshold_frames=50), frame_time=float(i),
        )
        orchestrator._update_lifecycles({obj_id: state}, frame_time=float(i))  # appears
        orchestrator._update_lifecycles({}, frame_time=float(i) + 0.5)  # disappears

    assert len(orchestrator._lifecycles) == 0, (
        f"expected zero surviving lifecycle entries after 500 objects passed through, "
        f"got {len(orchestrator._lifecycles)} -- indicates the leak has regressed"
    )


def test_orchestrator_with_detection_disabled_still_ages_tracks():
    """Section 7: if detection is disabled, the tracker still runs (with zero new
    detections) so existing tracks age out correctly -- validated here without needing
    the real detector process at all, since detect.enabled=False short-circuits before
    ever calling remote_detector.detect().
    """
    yuv_frame, frame_shape = _load_bus_as_yuv_frame()
    height, width = frame_shape

    camera = CameraConfig(
        name="disabled_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height, fps=5, enabled=False),
        record=RecordConfig(enabled=False),
    )

    motion_detector = MotionDetector(frame_shape=frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)

    class _UnusedDetector:
        model_config = ModelConfig(width=320, height=320)

        def detect(self, *args, **kwargs):
            raise AssertionError("detect() must not be called when detect.enabled is False")

    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, _UnusedDetector())

    result = orchestrator.process_frame(yuv_frame, frame_time=0.0)
    assert result.tracked_objects == {}
    assert result.regions == []


class _FakeDetector:
    """Stands in for RemoteObjectDetector: returns a FIXED set of detections every
    call, regardless of the real frame content, so the label-filtering gate in
    orchestration.py can be tested in isolation without the real ONNX model process.
    """

    model_config = ModelConfig(width=320, height=320)

    def __init__(self, detections: list[tuple[str, float, tuple]]) -> None:
        self._detections = detections

    def detect(self, *args, **kwargs) -> list[tuple[str, float, tuple]]:
        return self._detections


def _drive_frames(orchestrator: CameraOrchestrator, yuv_frame, count: int = 6):
    """Feeds the same frame repeatedly, matching the real integration test's own
    pattern -- enough frames for norfair's initialization_delay to confirm a track.
    """
    result = None
    for i in range(count):
        result = orchestrator.process_frame(yuv_frame, frame_time=float(i))
    return result


def test_track_all_tracks_a_label_not_in_the_track_list():
    """ObjectsConfig.track_all=True bypasses the `label not in camera.objects.track`
    gate entirely -- a detector-emitted label that was never typed into Track objects
    (here, "dog", with only "person" configured) must still be tracked.
    """
    yuv_frame, frame_shape = _load_bus_as_yuv_frame()
    height, width = frame_shape

    camera = CameraConfig(
        name="track_all_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height, fps=5),
        record=RecordConfig(enabled=False),
        objects=ObjectsConfig(track=["person"], track_all=True),
    )
    motion_detector = MotionDetector(frame_shape=frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)
    # A normalized box comfortably inside the frame, in (y1, x1, y2, x2) order.
    detector = _FakeDetector([("dog", 0.9, (0.1, 0.1, 0.4, 0.4))])
    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, detector)

    final = _drive_frames(orchestrator, yuv_frame)

    tracked_labels = {state.label for state in final.tracked_objects.values()}
    assert "dog" in tracked_labels, f"expected 'dog' to be tracked with track_all=True, got {tracked_labels}"


def test_track_all_false_still_filters_by_the_track_list():
    """The default (track_all=False) behavior is unchanged: a label not in Track
    objects is still silently discarded, same as before this feature existed.
    """
    yuv_frame, frame_shape = _load_bus_as_yuv_frame()
    height, width = frame_shape

    camera = CameraConfig(
        name="track_filtered_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height, fps=5),
        record=RecordConfig(enabled=False),
        objects=ObjectsConfig(track=["person"], track_all=False),
    )
    motion_detector = MotionDetector(frame_shape=frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)
    detector = _FakeDetector([("dog", 0.9, (0.1, 0.1, 0.4, 0.4))])
    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, detector)

    final = _drive_frames(orchestrator, yuv_frame)

    assert final.tracked_objects == {}, "a label outside Track objects must still be filtered when track_all=False"


def test_track_all_does_not_bypass_the_object_filter_thresholds():
    """track_all only bypasses the LABEL membership check -- ObjectFilterConfig's own
    score/area thresholds (is_object_filtered) still apply to every label exactly as
    before, so track_all=True doesn't also disable existing noise filtering.
    """
    from mirage.config.schema import ObjectFilterConfig

    yuv_frame, frame_shape = _load_bus_as_yuv_frame()
    height, width = frame_shape

    camera = CameraConfig(
        name="track_all_filtered_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height, fps=5),
        record=RecordConfig(enabled=False),
        objects=ObjectsConfig(
            track=["person"], track_all=True,
            filters={"dog": ObjectFilterConfig(min_score=0.99)},  # this detection's score (0.5) will never pass
        ),
    )
    motion_detector = MotionDetector(frame_shape=frame_shape, config=camera.motion)
    object_tracker = ObjectTracker(fps=camera.detect.fps)
    detector = _FakeDetector([("dog", 0.5, (0.1, 0.1, 0.4, 0.4))])
    orchestrator = CameraOrchestrator(camera, motion_detector, object_tracker, detector)

    final = _drive_frames(orchestrator, yuv_frame)

    assert final.tracked_objects == {}, "the existing min_score filter must still apply even with track_all=True"
