"""Full pipeline integration test: real CameraCapture (ffmpeg reading a live network
stream) -> real shared-memory frame ring -> real CameraTracker process (motion + region
selection + real ONNX detection + tracking) -> real DetectorProcess -> results delivered
to a detected_frames_queue, exactly matching the real multi-process architecture end to
end.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import signal
import socket
import subprocess as sp
import tempfile
import time
from pathlib import Path

import pytest

from mirage.capture.camera_capture import CameraCapture
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
from mirage.detection.process import DetectorProcess
from mirage.ipc.zmq_pubsub import ZmqProxy
from mirage.tracking.camera_tracker import CameraTracker
from mirage.util.shm import SharedMemoryFrameManager, preallocate_ring, teardown_ring

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
LABELMAP_PATH = Path(__file__).resolve().parent.parent / "models" / "coco_labelmap.txt"
STREAM_PORT = 19488
STREAM_URL = f"tcp://127.0.0.1:{STREAM_PORT}"
SERVER_LISTEN_URL = f"tcp://127.0.0.1:{STREAM_PORT}?listen"

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="yolov8n.onnx model fixture not available")


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_process_group(proc: sp.Popen) -> None:
    """SIGKILL the whole process group (the sh -c wrapper and every ffmpeg child it has
    spawned), rather than just proc.terminate() -- this codebase's own findings
    (IMPLEMENTATION_NOTES.md section 3b) show ffmpeg reading a live socket connection can
    be slow to react even to SIGTERM, so a plain terminate()+wait() is not reliable here.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except sp.TimeoutExpired:
        pass


@pytest.fixture
def tcp_video_server():
    # A real, continuously-live source with actual moving content (not a static test
    # pattern) so motion detection has something genuine to react to across the whole
    # test, in addition to the detector finding real objects. ffmpeg's testsrc includes a
    # moving clock/gradient pattern, which is enough to exercise motion-triggered region
    # selection (not just the one-time startup scan).
    # start_new_session=True puts the "sh -c ... while true ..." wrapper (and the ffmpeg
    # children it spawns, which inherit the same process group) in their own process
    # group, so os.killpg can reliably take down the whole tree in one shot at teardown.
    proc = sp.Popen(
        [
            "sh", "-c",
            f"while true; do "
            f"ffmpeg -hide_banner -loglevel error -re -f lavfi -i 'testsrc=size=640x480:rate=10' "
            f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -f mpegts '{SERVER_LISTEN_URL}'; "
            f"done",
        ],
        stdout=sp.DEVNULL, stderr=sp.DEVNULL, start_new_session=True,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        if _port_open(STREAM_PORT):
            break
        if proc.poll() is not None:
            pytest.fail("video test server exited before listening")
        time.sleep(0.1)
    else:
        _kill_process_group(proc)
        pytest.fail("video test server never started listening")

    yield STREAM_URL

    _kill_process_group(proc)
    # Belt-and-suspenders: also sweep by port number, in case some ffmpeg instance
    # somehow escaped the process group (e.g. a race where the shell loop spawned a new
    # child in the brief window before the group signal was delivered) -- this codebase's
    # own findings (IMPLEMENTATION_NOTES.md section 3b) show ffmpeg reading a live socket
    # can be slow to react even to SIGTERM, so a stray survivor is a real possibility,
    # not just theoretical.
    sp.run(["pkill", "-9", "-f", str(STREAM_PORT)], stdout=sp.DEVNULL, stderr=sp.DEVNULL)


@pytest.fixture
def short_ipc_dir():
    d = tempfile.mkdtemp(prefix="mrgfull", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_full_pipeline_capture_through_tracking(tcp_video_server, short_ipc_dir):
    ctx = mp.get_context("spawn")
    camera_name = "full_pipeline_cam"
    width, height, fps = 640, 480, 10

    camera = CameraConfig(
        name=camera_name,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=tcp_video_server)], retry_interval=2.0),
        detect=DetectConfig(width=width, height=height, fps=fps),
        record=RecordConfig(enabled=False),
        objects=ObjectsConfig(track=["person", "car", "bus", "clock"]),
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
    ring_depth = 20
    preallocate_ring(frame_manager, camera_name, ring_depth, width, height)

    manager = ctx.Manager()
    capture_frame_queue = manager.Queue(maxsize=2)
    current_frame_ts = manager.Value("d", 0.0)
    camera_fps_value = manager.Value("d", 0.0)
    skipped_fps_value = manager.Value("d", 0.0)
    capture_stop = ctx.Event()

    detection_queue = ctx.Queue()
    detected_frames_queue = ctx.Queue(maxsize=10)
    detector_stop = ctx.Event()
    tracker_stop = ctx.Event()

    capture = CameraCapture(
        camera=camera, cache_dir=short_ipc_dir, ring_depth=ring_depth,
        frame_queue=capture_frame_queue, current_frame_ts=current_frame_ts,
        camera_fps_value=camera_fps_value, skipped_fps_value=skipped_fps_value,
        stop_event=capture_stop,
    )
    detector_process = DetectorProcess(
        detector_config=detector_config, detection_queue=detection_queue,
        camera_names=[camera_name], detector_pub_addr=pub_addr, stop_event=detector_stop,
    )
    tracker = CameraTracker(
        camera=camera, detector_config=detector_config, detection_queue=detection_queue,
        detector_sub_addr=sub_addr, frame_queue=capture_frame_queue,
        detected_frames_queue=detected_frames_queue, stop_event=tracker_stop,
    )

    try:
        detector_process.start()
        capture.start()
        tracker.start()

        results = []
        deadline = time.time() + 40
        while time.time() < deadline and len(results) < 5:
            try:
                results.append(detected_frames_queue.get(timeout=5))
            except Exception:
                continue

        assert len(results) >= 5, f"expected at least 5 processed frames through the full pipeline, got {len(results)}"

        for camera_name_result, frame_name, frame_time, tracked_objects, motion_boxes, regions in results:
            assert camera_name_result == camera_name
            assert isinstance(frame_name, str)
            assert isinstance(frame_time, float)
            assert isinstance(tracked_objects, dict)
            assert isinstance(motion_boxes, list)
            assert isinstance(regions, list)

        # At least SOME frame across the whole run should have scanned a region (either
        # via startup scan or motion) -- confirms the pipeline isn't just passing empty
        # results through end to end.
        assert any(len(r[5]) > 0 for r in results), "expected at least one frame with non-empty regions"

    finally:
        tracker_stop.set()
        capture_stop.set()
        detector_stop.set()

        tracker.join(timeout=10)
        if tracker.is_alive():
            tracker.terminate()
            tracker.join(timeout=5)

        capture.join(timeout=15)
        if capture.is_alive():
            capture.terminate()
            capture.join(timeout=5)

        detector_process.join(timeout=15)
        if detector_process.is_alive():
            detector_process.terminate()
            detector_process.join(timeout=5)

        teardown_ring(frame_manager, camera_name, ring_depth)
        # The detector input/output SHM segments are created lazily (by whichever side
        # calls create() first -- here, the CameraTracker subprocess's own
        # RemoteObjectDetector instance) rather than pre-created by a maintainer as in
        # the real multi-camera system (see mirage/detection/remote.py's unlink() vs
        # close() docstrings) -- so this standalone test is responsible for unlinking
        # them itself.
        frame_manager.delete(camera_name)
        frame_manager.delete(f"out-{camera_name}")
        proxy.close()
        manager.shutdown()
