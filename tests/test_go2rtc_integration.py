"""Real integration test: downloads the actual go2rtc binary, launches it against a real
live video stream, and confirms it can actually restream/decode a valid frame -- not just
that the process starts and the config parses.
"""

from __future__ import annotations

import socket
import subprocess as sp
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    MirageConfig,
    ModelConfig,
)
from mirage.go2rtc.process import Go2rtcProcess
from tests.conftest import kill_process_group

STREAM_PORT = 19510
API_PORT = 19584
WEBRTC_PORT = 19855


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def video_server():
    proc = sp.Popen(
        [
            "sh", "-c",
            f"while true; do "
            f"ffmpeg -hide_banner -loglevel error -re -f lavfi -i 'testsrc=size=640x480:rate=10' "
            f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -g 20 "
            f"-force_key_frames 'expr:gte(t,n_forced*2)' -f mpegts 'tcp://127.0.0.1:{STREAM_PORT}?listen'; "
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
        kill_process_group(proc)
        pytest.fail("video test server never started listening")

    yield f"tcp://127.0.0.1:{STREAM_PORT}"

    kill_process_group(proc)


def test_go2rtc_restreams_a_real_live_source(video_server):
    camera = CameraConfig(
        name="go2rtc_test_cam",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=video_server)]),
        detect=DetectConfig(width=640, height=480, fps=10),
        detector="general",
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    config = MirageConfig(detectors={"general": detector}, cameras={"go2rtc_test_cam": camera})

    with tempfile.TemporaryDirectory(prefix="mrggortc") as tmp:
        proc = Go2rtcProcess(config, cache_dir=tmp, api_port=API_PORT, webrtc_port=WEBRTC_PORT)
        try:
            proc.start()

            deadline = time.time() + 15
            while time.time() < deadline and not _port_open(API_PORT):
                time.sleep(0.2)
            assert _port_open(API_PORT), "go2rtc API port never came up"
            assert proc.is_alive()

            # Confirm go2rtc registered our stream under the camera's own name.
            deadline = time.time() + 10
            streams = {}
            while time.time() < deadline:
                with urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/api/streams", timeout=2) as resp:
                    import json
                    streams = json.loads(resp.read())
                if "go2rtc_test_cam" in streams:
                    break
                time.sleep(0.5)
            assert "go2rtc_test_cam" in streams

            # Confirm it can actually produce a real decoded JPEG frame from the live
            # stream -- this is the actual proof the restream pipeline works end to end,
            # not just that config parsing/process launch succeeded.
            deadline = time.time() + 15
            jpeg_bytes = b""
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{API_PORT}/api/frame.jpeg?src=go2rtc_test_cam", timeout=5
                    ) as resp:
                        jpeg_bytes = resp.read()
                    if jpeg_bytes:
                        break
                except Exception:
                    pass
                time.sleep(0.5)

            assert jpeg_bytes.startswith(b"\xff\xd8"), "expected a valid JPEG (SOI marker) from go2rtc's snapshot endpoint"
            assert len(jpeg_bytes) > 1000

        finally:
            proc.stop()

        assert not proc.is_alive()
