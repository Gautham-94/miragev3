"""One-off manual debug script (not part of the test suite) used to isolate a real
network-stream capture issue during development. Kept for reference; run directly with
`python3 scripts/debug_network_capture.py` while a test TCP/mpegts server is listening.
"""

from __future__ import annotations

import multiprocessing as mp
import time

from mirage.capture.camera_capture import CameraCapture
from mirage.config.schema import CameraConfig, CameraInputConfig, DetectConfig, FfmpegConfig, RecordConfig
from mirage.util.shm import SharedMemoryFrameManager, preallocate_ring, teardown_ring


def main() -> None:
    ctx = mp.get_context("spawn")
    camera_name = "spawncam"
    width, height, fps = 64, 48, 10
    camera = CameraConfig(
        name=camera_name,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="tcp://127.0.0.1:19470")], retry_interval=2.0),
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
        camera=camera, cache_dir="/tmp", ring_depth=ring_depth, frame_queue=frame_queue,
        current_frame_ts=current_frame_ts, camera_fps_value=camera_fps_value,
        skipped_fps_value=skipped_fps_value, stop_event=stop_event,
    )
    capture.start()
    print("capture process pid:", capture.pid)

    received = []
    deadline = time.time() + 20
    while time.time() < deadline and len(received) < 5:
        try:
            received.append(frame_queue.get(timeout=3))
            print("got frame", len(received))
        except Exception as e:
            print("queue empty/timeout:", e)

    print("TOTAL:", len(received))
    stop_event.set()
    capture.join(timeout=10)
    print("capture alive after join:", capture.is_alive())
    teardown_ring(frame_manager, camera_name, ring_depth)
    manager.shutdown()


if __name__ == "__main__":
    main()
