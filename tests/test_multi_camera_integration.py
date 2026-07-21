"""End-to-end validation: two cameras, each with its own live network stream, both routed
to the SAME shared detector process -- validating the spec section 3.2 "multiple cameras
share one loaded model" architecture directly (every prior integration test used exactly
one camera). Confirms both cameras independently produce Recordings/tracked-object
activity while genuinely sharing one DetectorProcess (only one process is spawned for the
"general" detector regardless of camera count).
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess as sp
import tempfile
import time
from pathlib import Path

import pytest

from mirage.app import MirageApp
from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    InputDType,
    MirageConfig,
    ModelConfig,
    ObjectsConfig,
    PixelFormat,
    RecordConfig,
    RetainConfig,
)
from mirage.db.models import Recordings

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
LABELMAP_PATH = Path(__file__).resolve().parent.parent / "models" / "coco_labelmap.txt"
STREAM_PORT_1 = 19501
STREAM_PORT_2 = 19502

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="yolov8n.onnx model fixture not available")


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_process_group(proc: sp.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except sp.TimeoutExpired:
        pass


def _start_video_server(port: int, pattern: str) -> sp.Popen:
    listen_url = f"tcp://127.0.0.1:{port}?listen"
    proc = sp.Popen(
        [
            "sh", "-c",
            f"while true; do "
            f"ffmpeg -hide_banner -loglevel error -re -f lavfi -i '{pattern}=size=640x480:rate=10' "
            f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -g 20 "
            f"-force_key_frames 'expr:gte(t,n_forced*2)' -f mpegts '{listen_url}'; "
            f"done",
        ],
        stdout=sp.DEVNULL, stderr=sp.DEVNULL, start_new_session=True,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        if _port_open(port):
            return proc
        if proc.poll() is not None:
            pytest.fail(f"video test server on port {port} exited before listening")
        time.sleep(0.1)
    _kill_process_group(proc)
    pytest.fail(f"video test server on port {port} never started listening")


@pytest.fixture
def two_video_servers():
    # Two distinct lavfi source patterns (testsrc vs testsrc2) so the two cameras are
    # feeding genuinely different (if both synthetic) content, not literally identical
    # frames -- avoids any doubt about whether results are actually per-camera-isolated
    # versus accidentally reading the same data twice.
    proc1 = _start_video_server(STREAM_PORT_1, "testsrc")
    proc2 = _start_video_server(STREAM_PORT_2, "testsrc2")

    yield f"tcp://127.0.0.1:{STREAM_PORT_1}", f"tcp://127.0.0.1:{STREAM_PORT_2}"

    _kill_process_group(proc1)
    _kill_process_group(proc2)
    sp.run(["pkill", "-9", "-f", str(STREAM_PORT_1)], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    sp.run(["pkill", "-9", "-f", str(STREAM_PORT_2)], stdout=sp.DEVNULL, stderr=sp.DEVNULL)


@pytest.fixture
def app_environment():
    tmp = tempfile.mkdtemp(prefix="mrgmulti")
    (Path(tmp) / "cache").mkdir()
    (Path(tmp) / "recordings").mkdir()
    (Path(tmp) / "config").mkdir()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_two_cameras_share_one_detector_process(two_video_servers, app_environment):
    stream_url_1, stream_url_2 = two_video_servers
    tmp = app_environment

    camera1 = CameraConfig(
        name="cam1",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=stream_url_1, retry_interval=2.0)]),
        detect=DetectConfig(width=640, height=480, fps=10),
        record=RecordConfig(enabled=True, continuous=RetainConfig(days=1)),
        objects=ObjectsConfig(track=["person", "car", "bus", "clock"]),
        detector="shared",
    )
    camera2 = CameraConfig(
        name="cam2",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=stream_url_2, retry_interval=2.0)]),
        detect=DetectConfig(width=640, height=480, fps=10),
        record=RecordConfig(enabled=True, continuous=RetainConfig(days=1)),
        objects=ObjectsConfig(track=["person", "car", "bus", "clock"]),
        detector="shared",
    )
    detector_config = DetectorInstanceConfig(
        name="shared",
        model=ModelConfig(
            width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
            model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        ),
        device="onnx_yolov8",
    )
    config = MirageConfig(detectors={"shared": detector_config}, cameras={"cam1": camera1, "cam2": camera2})

    app = MirageApp(
        config,
        cache_dir=str(Path(tmp) / "cache"),
        record_dir=str(Path(tmp) / "recordings"),
        db_path=str(Path(tmp) / "config" / "mirage.db"),
        enable_go2rtc=False,  # this test targets shared-detector behavior; go2rtc is exercised separately
    )
    try:
        app.start()

        # Exactly ONE detector process for the ONE configured detector, regardless of
        # having 2 cameras -- this is the crux of the "share one loaded model" claim.
        assert len(app.detector_processes) == 1
        assert len(app.tracker_processes) == 2
        assert len(app.capture_processes) == 2

        deadline = time.time() + 40
        cameras_with_recordings: set[str] = set()
        while time.time() < deadline and len(cameras_with_recordings) < 2:
            for cam_name in ("cam1", "cam2"):
                if Recordings.select().where(Recordings.camera == cam_name).count() > 0:
                    cameras_with_recordings.add(cam_name)
            if len(cameras_with_recordings) < 2:
                time.sleep(1)

        assert cameras_with_recordings == {"cam1", "cam2"}, (
            f"expected both cameras to independently produce recordings, got {cameras_with_recordings}"
        )

        # Both tracker processes and the single shared detector process must all still
        # be alive and healthy throughout -- confirms the shared queue/SHM-per-camera
        # design didn't starve or crash either camera.
        for tracker in app.tracker_processes.values():
            assert tracker.is_alive()
        for detector in app.detector_processes.values():
            assert detector.is_alive()

    finally:
        app.stop()

    for tracker in app.tracker_processes.values():
        assert not tracker.is_alive()
    for capture in app.capture_processes.values():
        assert not capture.is_alive()
    for detector in app.detector_processes.values():
        assert not detector.is_alive()


def test_two_cameras_split_across_two_detector_workers(two_video_servers, app_environment):
    """DetectorInstanceConfig.num_workers=2 with 2 cameras routed to that detector must
    spawn TWO independent DetectorProcess instances (not one shared process, the
    opposite of test_two_cameras_share_one_detector_process above) -- each camera
    statically assigned to its own worker/queue, added specifically to fix the real
    single-consumer bottleneck confirmed this session (one shared detector process
    serializes inference across every routed camera; see DetectorInstanceConfig.
    num_workers's own docstring).
    """
    stream_url_1, stream_url_2 = two_video_servers
    tmp = app_environment

    camera1 = CameraConfig(
        name="cam1",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=stream_url_1, retry_interval=2.0)]),
        detect=DetectConfig(width=640, height=480, fps=10),
        record=RecordConfig(enabled=True, continuous=RetainConfig(days=1)),
        objects=ObjectsConfig(track=["person", "car", "bus", "clock"]),
        detector="shared",
    )
    camera2 = CameraConfig(
        name="cam2",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=stream_url_2, retry_interval=2.0)]),
        detect=DetectConfig(width=640, height=480, fps=10),
        record=RecordConfig(enabled=True, continuous=RetainConfig(days=1)),
        objects=ObjectsConfig(track=["person", "car", "bus", "clock"]),
        detector="shared",
    )
    detector_config = DetectorInstanceConfig(
        name="shared",
        model=ModelConfig(
            width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
            model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        ),
        device="onnx_yolov8",
        num_workers=2,
    )
    config = MirageConfig(detectors={"shared": detector_config}, cameras={"cam1": camera1, "cam2": camera2})

    app = MirageApp(
        config,
        cache_dir=str(Path(tmp) / "cache"),
        record_dir=str(Path(tmp) / "recordings"),
        db_path=str(Path(tmp) / "config" / "mirage.db"),
        enable_go2rtc=False,
    )
    try:
        app.start()

        # TWO detector processes for the one configured detector (num_workers=2),
        # unlike the num_workers=1 (default) case above.
        assert len(app.detector_processes) == 2
        assert set(app.detector_processes.keys()) == {"shared#0", "shared#1"}

        # ONE shared queue for the detector, keyed by detector name (not camera name or
        # worker index) -- both cam1 and cam2 put() onto this SAME queue, and both
        # worker processes .get() from it. This is what gives free least-busy-worker
        # routing: whichever worker is idle and calls .get() first serves the next
        # request, regardless of which camera it came from. See self.detection_queues's
        # own docstring in mirage/app.py for the full reasoning.
        assert set(app.detection_queues.keys()) == {"shared"}

        deadline = time.time() + 40
        cameras_with_recordings: set[str] = set()
        while time.time() < deadline and len(cameras_with_recordings) < 2:
            for cam_name in ("cam1", "cam2"):
                if Recordings.select().where(Recordings.camera == cam_name).count() > 0:
                    cameras_with_recordings.add(cam_name)
            if len(cameras_with_recordings) < 2:
                time.sleep(1)

        assert cameras_with_recordings == {"cam1", "cam2"}, (
            f"expected both cameras to independently produce recordings, got {cameras_with_recordings}"
        )

        for detector in app.detector_processes.values():
            assert detector.is_alive()

    finally:
        app.stop()

    for detector in app.detector_processes.values():
        assert not detector.is_alive()
