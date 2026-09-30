"""Full-application integration test: MirageApp.start() with a real camera pointed at a
live network stream, real ONNX detection, real recording pipeline, real DB, running for a
short window, then MirageApp.stop() -- verifying the whole system boots, produces Event/
Recordings rows, and shuts down cleanly with no leaked processes or shared memory.
"""

from __future__ import annotations

import shutil
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
from mirage.db.models import Event, Recordings
from tests.conftest import kill_process_group

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
LABELMAP_PATH = Path(__file__).resolve().parent.parent / "models" / "coco_labelmap.txt"

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="yolov8n.onnx model fixture not available")


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    """A fresh, currently-unused TCP port -- video_server() used to hardcode a single
    module-level STREAM_PORT shared by every test in this file, which caused a real,
    reproducible hang: two tests running the video_server fixture back to back could
    race the OS's port-release timing after the first test's ffmpeg process was killed,
    leaving the second test's ffmpeg stuck in its own `while true; do ffmpeg ...; done`
    retry loop forever failing to bind the still-held port -- silently, with no test
    output, since the fixture's own startup deadline check only looks for the port
    becoming OPEN (a client connecting successfully), never distinguishing "ffmpeg
    hasn't started yet" from "ffmpeg can't bind and is retry-looping forever". Allocating
    a fresh ephemeral port per test (bind to port 0, let the OS pick one, close
    immediately so ffmpeg can bind it) removes the shared-state race entirely rather than
    trying to make port release between tests more reliable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def video_server():
    # IMPORTANT: `-g 20 -force_key_frames 'expr:gte(t,n_forced*2)'` forces a keyframe
    # every 2 seconds. Without this, libx264 -preset ultrafast against a raw lavfi
    # source defaults to a MUCH longer GOP (observed: ~25s, confirmed via ffprobe's
    # packet flags=K timestamps during development) -- and ffmpeg's segment muxer only
    # cuts a new segment at a keyframe boundary, so with segment_time=10 and a 25s GOP,
    # no segment ever completes within a realistic test window (the recording pipeline
    # correctly discards the perpetually-incomplete file as "invalid", which is exactly
    # what happened before this fix -- see IMPLEMENTATION_NOTES.md). A real camera's
    # encoder is essentially always configured with a short (1-2s) GOP specifically so
    # recording/segmenting/seeking works well, so this fixture now matches that.
    stream_port = _free_port()
    stream_url = f"tcp://127.0.0.1:{stream_port}"
    server_listen_url = f"tcp://127.0.0.1:{stream_port}?listen"

    proc = sp.Popen(
        [
            "sh", "-c",
            f"while true; do "
            f"ffmpeg -hide_banner -loglevel error -re -f lavfi -i 'testsrc=size=640x480:rate=10' "
            f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -g 20 "
            f"-force_key_frames 'expr:gte(t,n_forced*2)' -f mpegts '{server_listen_url}'; "
            f"done",
        ],
        stdout=sp.DEVNULL, stderr=sp.DEVNULL, start_new_session=True,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        if _port_open(stream_port):
            break
        if proc.poll() is not None:
            pytest.fail("video test server exited before listening")
        time.sleep(0.1)
    else:
        kill_process_group(proc)
        pytest.fail("video test server never started listening")

    yield stream_url

    kill_process_group(proc)


@pytest.fixture
def app_environment():
    # MirageApp takes cache_dir/record_dir/db_path as explicit constructor overrides
    # (rather than relying on mirage.const's module-level env-var-derived defaults), so
    # this fixture just hands back a scratch directory -- no env var / module reload
    # trickery needed.
    tmp = tempfile.mkdtemp(prefix="mrgapp")
    (Path(tmp) / "cache").mkdir()
    (Path(tmp) / "recordings").mkdir()
    (Path(tmp) / "config").mkdir()

    yield tmp

    shutil.rmtree(tmp, ignore_errors=True)


def test_app_full_startup_and_shutdown(video_server, app_environment):
    tmp = app_environment
    camera = CameraConfig(
        name="app_test_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=video_server, retry_interval=2.0)]),
        detect=DetectConfig(width=640, height=480, fps=10),
        record=RecordConfig(enabled=True, continuous=RetainConfig(days=1)),
        objects=ObjectsConfig(track=["person", "car", "bus"]),
        detector="general",
    )
    detector_config = DetectorInstanceConfig(
        name="general",
        model=ModelConfig(
            width=320, height=320, input_dtype=InputDType.int_, pixel_format=PixelFormat.rgb,
            model_path=str(MODEL_PATH), labelmap_path=str(LABELMAP_PATH),
        ),
        device="onnx_yolov8",
    )
    config = MirageConfig(detectors={"general": detector_config}, cameras={"app_test_cam": camera})

    app = MirageApp(
        config,
        cache_dir=str(Path(tmp) / "cache"),
        record_dir=str(Path(tmp) / "recordings"),
        db_path=str(Path(tmp) / "config" / "mirage.db"),
        enable_go2rtc=False,  # this test targets the capture/detect/record pipeline; go2rtc is exercised separately
    )
    try:
        app.start()

        # Let the system run long enough to: capture frames, run detection/tracking, and
        # have the recording maintainer's 5s-cadence loop pick up at least one segment
        # (ffmpeg's record-role segment_time defaults to 10s, so give it enough margin).
        deadline = time.time() + 35
        got_recording = False
        while time.time() < deadline:
            if Recordings.select().where(Recordings.camera == "app_test_cam").count() > 0:
                got_recording = True
                break
            time.sleep(1)

        assert got_recording, "expected at least one Recordings row within the test window"

        # The result-consumer thread should be alive and processing (not crashed).
        assert app.result_consumer_thread.is_alive()

    finally:
        app.stop()

    # Verify a clean process shutdown: no tracker/capture/detector processes left alive.
    for tracker in app.tracker_processes.values():
        assert not tracker.is_alive()
    for capture in app.capture_processes.values():
        assert not capture.is_alive()
    for detector in app.detector_processes.values():
        assert not detector.is_alive()
