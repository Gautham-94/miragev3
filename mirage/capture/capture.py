"""The capture read loop: reads raw YUV420 frames off ffmpeg's stdout into the shared
memory ring buffer, and signals a frame's readiness via a small bounded queue.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.4.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Optional

from mirage.util.rate import EventsPerSecond
from mirage.util.shm import SharedMemoryFrameManager, frame_name

logger = logging.getLogger(__name__)


def capture_frames(
    camera_name: str,
    ffmpeg_process,
    frame_size: int,
    ring_depth: int,
    frame_manager: SharedMemoryFrameManager,
    frame_queue,
    current_frame_ts,  # multiprocessing.Value("d", ...)
    camera_fps: EventsPerSecond,
    skipped_fps: EventsPerSecond,
    stop_event,
    start_index: int = 0,
) -> None:
    """Runs in a dedicated thread inside the per-camera capture process.

    Reads exactly one raw YUV420 frame per iteration from ffmpeg's stdout, writes it
    straight into a shared-memory slot (no numpy conversion here -- that happens only in
    the consumer), then signals readiness via a small bounded queue. If the queue is full
    (consumer fell behind), only the queue entry for this frame is dropped -- the shared
    memory slot was still fully written and will simply be overwritten on the ring's next
    lap. This bounded-queue-plus-nonblocking-put IS the entire backpressure mechanism;
    there is no other frame-rate/drop logic needed on the Python side, because ffmpeg's own
    `-r`/`fps=` filter already throttles frame production to the configured detect fps.
    """
    frame_index = start_index

    while not stop_event.is_set():
        name = frame_name(camera_name, frame_index)
        frame_buffer = frame_manager.write(name, frame_size)
        if frame_buffer is None:
            # Segment doesn't exist yet (race with the maintainer's preallocation, or a
            # runtime-added camera). Brief backoff rather than a hot spin.
            time.sleep(0.05)
            continue

        try:
            data = ffmpeg_process.stdout.read(frame_size)
        except Exception:
            if stop_event.is_set():
                break
            if ffmpeg_process.poll() is not None:
                logger.error("%s: ffmpeg process exited; capture thread stopping", camera_name)
                break
            continue

        if len(data) != frame_size:
            # Short read: ffmpeg's stdout closed or was interrupted mid-frame.
            if ffmpeg_process.poll() is not None:
                logger.error("%s: ffmpeg process exited (short read); capture thread stopping", camera_name)
                break
            continue

        frame_buffer[:] = data

        now = time.time()
        current_frame_ts.value = now
        camera_fps.update(now)

        try:
            frame_queue.put((name, now), block=False)
            frame_manager.close(name)
        except queue.Full:
            skipped_fps.update(now)

        frame_index = 0 if frame_index == ring_depth - 1 else frame_index + 1
