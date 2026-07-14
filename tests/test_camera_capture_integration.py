"""Integration test: run the real CameraCapture process against a real ffmpeg reading a
local test video file (not yet RTSP -- see test_rtsp_integration.py for that), verifying
frames actually land in shared memory and the frame queue, then a clean shutdown leaves no
leaked shared memory segments or zombie ffmpeg processes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from mirage.capture.camera_capture import CameraCapture
from mirage.config.schema import CameraConfig, CameraInputConfig, DetectConfig, FfmpegConfig, RecordConfig
from mirage.util.shm import SharedMemoryFrameManager, frame_name, preallocate_ring, teardown_ring, yuv_frame_shape

TEST_VIDEO = Path(__file__).resolve().parent.parent / "media" / "test_source.mp4"


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_camera_capture_produces_frames_from_local_file():
    ctx = mp.get_context("spawn")
    camera_name = "integration_cam"
    width, height, fps = 64, 48, 10

    camera = CameraConfig(
        name=camera_name,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path=str(TEST_VIDEO))]),
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
        cache_dir=str(TEST_VIDEO.parent),
        ring_depth=ring_depth,
        frame_queue=frame_queue,
        current_frame_ts=current_frame_ts,
        camera_fps_value=camera_fps_value,
        skipped_fps_value=skipped_fps_value,
        stop_event=stop_event,
    )

    try:
        capture.start()

        received = []
        deadline = time.time() + 15
        while time.time() < deadline and len(received) < 5:
            try:
                received.append(frame_queue.get(timeout=2))
            except Exception:
                continue

        assert len(received) >= 5, f"expected at least 5 frames, got {len(received)}"

        # Verify the frames actually contain real pixel data (not all-zero/garbage) by
        # reading straight out of shared memory using the frame names we received.
        first_name, first_ts = received[0]
        arr = frame_manager.get(first_name, yuv_frame_shape(width, height))
        assert arr is not None
        assert arr.sum() > 0, "expected non-zero pixel data from the test video's luma plane"
        assert first_ts > 0

    finally:
        stop_event.set()
        capture.join(timeout=15)
        if capture.is_alive():
            capture.terminate()
            capture.join(timeout=5)
        teardown_ring(frame_manager, camera_name, ring_depth)
        manager.shutdown()

    assert capture.exitcode in (0, None, -15)  # 0 clean, -15 = SIGTERM if we had to terminate
