"""Integration test: run the real CameraCapture process against ffmpeg reading from an
actual network stream (not a local file path), to validate the capture pipeline over a
real socket connection end-to-end.

Note on protocol choice: the spec's primary target is RTSP cameras. This machine's ffmpeg
build (8.1, Homebrew) supports RTSP as an output muxer but its `-listen`-mode RTSP server
support is unreliable for this purpose (see IMPLEMENTATION_NOTES.md for the exact failure
modes tried: RTSP -listen never actually binds; RTMP -listen works but a looped-mp4 source
hits a real FLV-demuxer "Packet mismatch" decode error at small test resolutions). MPEG-TS
over a raw listening TCP socket, fed by ffmpeg's `lavfi` testsrc generator, was the
combination that reproduced cleanly and repeatedly, so it's used here as the synthetic
network source. It exercises the exact same runtime code path the pipeline uses for a real
RTSP camera (ffmpeg subprocess reading a live network connection, decoding, scaling, and
piping raw YUV420 frames back to Python) -- the only camera-input-type-specific logic
(-rtsp_transport/timeout arg selection) lives entirely in build_input_args() and is covered
by protocol-selection unit tests in test_ffmpeg_presets.py, not by this test.
"""

from __future__ import annotations

import multiprocessing as mp
import socket
import subprocess as sp
import time
from pathlib import Path

import pytest

from mirage.capture.camera_capture import CameraCapture
from mirage.config.schema import CameraConfig, CameraInputConfig, DetectConfig, FfmpegConfig, RecordConfig
from mirage.util.shm import SharedMemoryFrameManager, preallocate_ring, teardown_ring, yuv_frame_shape

TEST_VIDEO_DIR = Path(__file__).resolve().parent.parent / "media"
STREAM_PORT = 19470
STREAM_URL = f"tcp://127.0.0.1:{STREAM_PORT}"
SERVER_LISTEN_URL = f"tcp://127.0.0.1:{STREAM_PORT}?listen"


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def tcp_stream_server():
    # A `?listen` raw-TCP MPEG-TS server accepts exactly one connection and then the
    # server process exits once that client disconnects (same behavior observed for the
    # RTMP -listen case -- see IMPLEMENTATION_NOTES.md). A real always-on camera accepts
    # reconnects indefinitely, so wrap the server invocation in a shell loop that keeps
    # re-listening after each client disconnects.
    proc = sp.Popen(
        [
            "sh", "-c",
            f"while true; do "
            f"ffmpeg -hide_banner -loglevel error -re -f lavfi -i 'testsrc=size=64x48:rate=10' "
            f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -f mpegts '{SERVER_LISTEN_URL}'; "
            f"done",
        ],
        stdout=sp.DEVNULL, stderr=sp.DEVNULL,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        if _port_open(STREAM_PORT):
            break
        if proc.poll() is not None:
            pytest.fail("tcp stream test server process exited before it started listening")
        time.sleep(0.1)
    else:
        proc.terminate()
        pytest.fail("tcp stream test server never started listening")

    yield STREAM_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except sp.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Match on the port number alone rather than the full URL string -- more robust
    # against any shell quoting/escaping differences between how the process was
    # launched (via `sh -c "..."`) and how pkill's regex matches argv.
    sp.run(["pkill", "-f", str(STREAM_PORT)], stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def test_camera_capture_over_real_network_stream(tcp_stream_server):
    ctx = mp.get_context("spawn")
    camera_name = "network_integration_cam"
    width, height, fps = 64, 48, 10

    camera = CameraConfig(
        name=camera_name,
        # A short retry_interval so an occasional transient handshake race on the very
        # first connection attempt (a real risk with this synthetic `?listen` test
        # server -- see IMPLEMENTATION_NOTES.md) gets retried quickly by the watchdog
        # rather than costing the default 10s backoff.
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=tcp_stream_server)], retry_interval=2.0),
        detect=DetectConfig(width=width, height=height, fps=fps),
        record=RecordConfig(enabled=False),
    )

    frame_manager = SharedMemoryFrameManager()
    ring_depth = 10
    preallocate_ring(frame_manager, camera_name, ring_depth, width, height)

    manager = ctx.Manager()
    frame_queue = manager.Queue(maxsize=2)
    current_frame_ts = manager.Value("d", 0.0)
    camera_fps_value = manager.Value("d", 0.0)
    skipped_fps_value = manager.Value("d", 0.0)
    stop_event = ctx.Event()

    capture = CameraCapture(
        camera=camera,
        cache_dir=str(TEST_VIDEO_DIR),
        ring_depth=ring_depth,
        frame_queue=frame_queue,
        current_frame_ts=current_frame_ts,
        camera_fps_value=camera_fps_value,
        skipped_fps_value=skipped_fps_value,
        stop_event=stop_event,
    )

    try:
        capture.start()

        # NOTE: the first frame commonly takes a couple of seconds to arrive even on a
        # successful connection (ffmpeg's own startup/probe latency against a live
        # socket), so a timeout on any individual get() must NOT abort the whole
        # collection loop -- only the overall deadline should do that.
        received = []
        deadline = time.time() + 25
        while time.time() < deadline and len(received) < 10:
            try:
                received.append(frame_queue.get(timeout=3))
            except Exception:
                continue

        assert len(received) >= 10, f"expected at least 10 frames from the network stream, got {len(received)}"

        first_name, first_ts = received[0]
        arr = frame_manager.get(first_name, yuv_frame_shape(width, height))
        assert arr is not None
        assert arr.sum() > 0, "expected non-zero pixel data decoded from the network stream"

        # Since the source is a continuously-live generator (not a short finite clip),
        # sustaining >=10 frames also confirms the capture loop keeps up with a real
        # long-running connection rather than only working for a brief burst.
        timestamps = [ts for _, ts in received]
        assert timestamps == sorted(timestamps), "frame timestamps should be monotonically increasing"

    finally:
        stop_event.set()
        capture.join(timeout=15)
        if capture.is_alive():
            capture.terminate()
            capture.join(timeout=5)
        teardown_ring(frame_manager, camera_name, ring_depth)
        manager.shutdown()
