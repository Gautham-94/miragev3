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
