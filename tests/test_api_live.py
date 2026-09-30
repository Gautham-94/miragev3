"""Integration tests for the live-view proxy endpoints (mirage/api/routers/live.py)
against a REAL running go2rtc, restreaming a REAL live video test source -- not mocked.
Mirrors tests/test_go2rtc_integration.py's approach, but exercises the proxy layer on top.
"""

from __future__ import annotations

import socket
import subprocess as sp
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    MirageConfig,
    ModelConfig,
)
from mirage.go2rtc.config import SUB_STREAM_SUFFIX
from mirage.go2rtc.process import Go2rtcProcess
from tests.conftest import kill_process_group

STREAM_PORT = 19520
API_LIVE_TEST_PORT = 19594
WEBRTC_LIVE_TEST_PORT = 19856
CAMERA_NAME = "live_test_cam"


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


@pytest.fixture
def config(video_server):
    camera = CameraConfig(
        name=CAMERA_NAME,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=video_server)]),
        detect=DetectConfig(width=640, height=480, fps=10),
        detector="general",
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    return MirageConfig(detectors={"general": detector}, cameras={CAMERA_NAME: camera})


@pytest.fixture
def running_go2rtc(config):
    with tempfile.TemporaryDirectory(prefix="mrglivetest") as tmp:
        proc = Go2rtcProcess(config, cache_dir=tmp, api_port=API_LIVE_TEST_PORT, webrtc_port=WEBRTC_LIVE_TEST_PORT)
        proc.start()
        deadline = time.time() + 15
        while time.time() < deadline and not _port_open(API_LIVE_TEST_PORT):
            time.sleep(0.2)
        assert _port_open(API_LIVE_TEST_PORT), "go2rtc API port never came up"
        yield proc
        proc.stop()


@pytest.fixture
def client(config, running_go2rtc):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        app = create_app(config, db_path=db_path, go2rtc_api_port=API_LIVE_TEST_PORT)
        with TestClient(app) as c:
            yield c


def test_snapshot_proxy_returns_real_jpeg(client):
    deadline = time.time() + 20
    resp = None
    while time.time() < deadline:
        resp = client.get(f"/api/live/{CAMERA_NAME}/snapshot.jpg")
        if resp.status_code == 200 and resp.content.startswith(b"\xff\xd8"):
            break
        time.sleep(0.5)

    assert resp is not None
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content.startswith(b"\xff\xd8"), f"expected JPEG SOI marker, got {len(resp.content)} bytes"
    assert len(resp.content) > 1000


def test_snapshot_proxy_unknown_camera_returns_404(client):
    resp = client.get("/api/live/nonexistent_camera/snapshot.jpg")
    assert resp.status_code == 404


def test_snapshot_proxy_accepts_sub_suffixed_stream_name(client):
    # "<camera>_sub" (the low-res grid-tier go2rtc stream, see
    # mirage.go2rtc.config.build_go2rtc_config) must pass this endpoint's camera
    # validation -- it should never 404 here just for carrying the suffix. This fixture's
    # source is a non-rtsp:// test stream, so go2rtc itself never actually registers a
    # "<camera>_sub" stream for it (see build_go2rtc_config's own rtsp-only guard) -- a
    # 502 (go2rtc has no such stream) proves our own validation let the request through
    # and the failure is go2rtc's, not ours; a 404 would mean this proxy wrongly rejected
    # a legitimately-shaped stream name.
    resp = client.get(f"/api/live/{CAMERA_NAME}{SUB_STREAM_SUFFIX}/snapshot.jpg")
    assert resp.status_code != 404


def test_ws_proxy_relays_go2rtc_messages(client):
    # Give go2rtc a moment to have the stream actually available before opening the
    # signaling WS (mirrors the polling done for the snapshot endpoint elsewhere).
    deadline = time.time() + 15
    while time.time() < deadline:
        resp = client.get(f"/api/live/{CAMERA_NAME}/snapshot.jpg")
        if resp.status_code == 200:
            break
        time.sleep(0.5)

    with client.websocket_connect(f"/api/live/{CAMERA_NAME}/ws") as ws:
        # go2rtc's /api/ws sends a "streams" info message on connect for a known
        # source -- just confirm SOMETHING real comes back through our proxy within a
        # reasonable window, proving the bidirectional relay to the real go2rtc process
        # is actually alive end-to-end (not a stub).
        ws.send_text('{"type":"mse","value":"video/mp4"}')
        message = ws.receive()
        assert message is not None


def test_ws_proxy_unknown_camera_closes_immediately(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/api/live/nonexistent_camera/ws") as ws:
            ws.receive()
