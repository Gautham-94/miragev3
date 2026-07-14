from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from io import BytesIO

from mirage.capture.capture import capture_frames
from mirage.util.rate import EventsPerSecond
from mirage.util.shm import SharedMemoryFrameManager, frame_name


class FakeStdout:
    """Simulates ffmpeg's stdout: yields a fixed number of full frames, then a short read
    (simulating EOF/broken pipe), matching real read() semantics closely enough for the
    capture loop's short-read handling to be exercised.
    """

    def __init__(self, frame_size: int, num_frames: int, frame_value: int = 0x42):
        self._chunks = [bytes([frame_value]) * frame_size for _ in range(num_frames)]
        self._chunks.append(b"")  # simulate short/EOF read after frames are exhausted
        self._idx = 0
        self.closed = False

    def read(self, n: int) -> bytes:
        if self._idx >= len(self._chunks):
            return b""
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class FakeProcess:
    def __init__(self, stdout: FakeStdout):
        self.stdout = stdout
        self._exited = False

    def poll(self):
        # Report "exited" only once stdout has been fully drained (matches real ffmpeg:
        # process exit is what eventually produces EOF on the pipe).
        return 0 if self.stdout._idx >= len(self.stdout._chunks) else None


def test_capture_frames_writes_ring_and_signals_queue():
    camera = "testcam_capture"
    width, height = 16, 12
    frame_size = width * height * 3 // 2
    ring_depth = 4
    num_frames = 6  # more frames than ring depth, to exercise wraparound

    fm = SharedMemoryFrameManager()
    for i in range(ring_depth):
        fm.create(frame_name(camera, i), frame_size)

    try:
        # maxsize is deliberately >= num_frames here: this test verifies ring-buffer
        # wraparound and naming, not the backpressure/drop behavior (which has its own
        # dedicated test below) -- a small maxsize combined with a background drainer
        # thread would make "how many frames actually got queued" a timing-dependent
        # race, which isn't what this test is checking.
        frame_q = queue.Queue(maxsize=num_frames)
        current_ts = type("V", (), {"value": 0.0})()
        stop_event = threading.Event()
        camera_fps = EventsPerSecond()
        skipped_fps = EventsPerSecond()

        proc = FakeProcess(FakeStdout(frame_size, num_frames))

        # Drain the queue concurrently so backpressure doesn't block the (synchronous)
        # capture loop call below -- capture_frames runs to completion when ffmpeg "exits".
        received = []

        def drain():
            while len(received) < num_frames:
                try:
                    received.append(frame_q.get(timeout=2))
                except queue.Empty:
                    break

        drainer = threading.Thread(target=drain, daemon=True)
        drainer.start()

        capture_frames(
            camera_name=camera,
            ffmpeg_process=proc,
            frame_size=frame_size,
            ring_depth=ring_depth,
            frame_manager=fm,
            frame_queue=frame_q,
            current_frame_ts=current_ts,
            camera_fps=camera_fps,
            skipped_fps=skipped_fps,
            stop_event=stop_event,
        )
        drainer.join(timeout=3)

        assert len(received) == num_frames
        # Names must cycle through the ring, wrapping after ring_depth.
        expected_names = [frame_name(camera, i % ring_depth) for i in range(num_frames)]
        assert [n for n, _ in received] == expected_names
        assert current_ts.value > 0
    finally:
        for i in range(ring_depth):
            fm.delete(frame_name(camera, i))


def test_capture_frames_drops_on_full_queue_but_keeps_shm_written():
    camera = "testcam_backpressure"
    width, height = 8, 6
    frame_size = width * height * 3 // 2
    ring_depth = 2
    num_frames = 3

    fm = SharedMemoryFrameManager()
    for i in range(ring_depth):
        fm.create(frame_name(camera, i), frame_size)

    try:
        # maxsize=1: force queue.Full quickly since nothing drains it during the call.
        frame_q = queue.Queue(maxsize=1)
        current_ts = type("V", (), {"value": 0.0})()
        stop_event = threading.Event()
        camera_fps = EventsPerSecond()
        skipped_fps = EventsPerSecond()

        proc = FakeProcess(FakeStdout(frame_size, num_frames, frame_value=0x7A))

        capture_frames(
            camera_name=camera,
            ffmpeg_process=proc,
            frame_size=frame_size,
            ring_depth=ring_depth,
            frame_manager=fm,
            frame_queue=frame_q,
            current_frame_ts=current_ts,
            camera_fps=camera_fps,
            skipped_fps=skipped_fps,
            stop_event=stop_event,
        )

        # With maxsize=1 and nothing draining the queue during this call, at least one of
        # the 3 frames must have been dropped from the queue (queue.Full path) -- but the
        # SHM ring slot must still contain real frame data regardless, since capture
        # writes to shared memory BEFORE attempting the queue put.
        assert skipped_fps.count() > 0, "expected at least one frame to be dropped due to a full queue"
        arr = fm.get(frame_name(camera, 0), (frame_size,))
        assert arr is not None
        assert arr[0] == 0x7A
    finally:
        for i in range(ring_depth):
            fm.delete(frame_name(camera, i))


def test_capture_frames_stops_cleanly_on_stop_event():
    camera = "testcam_stopevent"
    frame_size = 100
    ring_depth = 2
    fm = SharedMemoryFrameManager()
    for i in range(ring_depth):
        fm.create(frame_name(camera, i), frame_size)

    try:
        frame_q = queue.Queue(maxsize=2)
        current_ts = type("V", (), {"value": 0.0})()
        stop_event = threading.Event()
        stop_event.set()  # already stopped -- loop must exit immediately without reading

        proc = FakeProcess(FakeStdout(frame_size, num_frames=5))
        capture_frames(
            camera_name=camera,
            ffmpeg_process=proc,
            frame_size=frame_size,
            ring_depth=ring_depth,
            frame_manager=fm,
            frame_queue=frame_q,
            current_frame_ts=current_ts,
            camera_fps=EventsPerSecond(),
            skipped_fps=EventsPerSecond(),
            stop_event=stop_event,
        )
        assert frame_q.empty()
    finally:
        for i in range(ring_depth):
            fm.delete(frame_name(camera, i))
