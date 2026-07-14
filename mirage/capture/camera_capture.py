"""CameraCapture: the per-camera OS process that owns ffmpeg subprocess(es).

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.1.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os

from mirage.capture.watchdog import CameraWatchdog
from mirage.config.schema import CameraConfig
from mirage.const import PROCESS_PRIORITY_HIGH
from mirage.util.rate import EventsPerSecond
from mirage.util.shm import SharedMemoryFrameManager

logger = logging.getLogger(__name__)


class CameraCapture(mp.Process):
    def __init__(
        self,
        camera: CameraConfig,
        cache_dir: str,
        ring_depth: int,
        frame_queue,
        current_frame_ts,
        camera_fps_value,
        skipped_fps_value,
        stop_event,
    ) -> None:
        super().__init__(name=f"capture:{camera.name}")
        self.camera = camera
        self.cache_dir = cache_dir
        self.ring_depth = ring_depth
        self.frame_queue = frame_queue
        self.current_frame_ts = current_frame_ts
        self.camera_fps_value = camera_fps_value
        self.skipped_fps_value = skipped_fps_value
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            os.nice(PROCESS_PRIORITY_HIGH)
        except (AttributeError, PermissionError, OSError):
            pass  # best-effort; not fatal if the OS/user can't raise priority

        logging.basicConfig(level=logging.INFO)
        frame_manager = SharedMemoryFrameManager()

        # Bridge multiprocessing.Value counters used for cross-process stats reporting to
        # the in-thread EventsPerSecond objects the watchdog/capture loop operate on.
        camera_fps = _ValueBackedEventsPerSecond(self.camera_fps_value)
        skipped_fps = _ValueBackedEventsPerSecond(self.skipped_fps_value)

        watchdog = CameraWatchdog(
            camera=self.camera,
            cache_dir=self.cache_dir,
            ring_depth=self.ring_depth,
            frame_manager=frame_manager,
            frame_queue=self.frame_queue,
            current_frame_ts=self.current_frame_ts,
            camera_fps=camera_fps,
            skipped_fps=skipped_fps,
            stop_event=self.stop_event,
        )
        watchdog.start()
        watchdog.join()


class _ValueBackedEventsPerSecond(EventsPerSecond):
    """An EventsPerSecond whose latest computed rate is also mirrored into a
    multiprocessing.Value, so the main process's stats collector can read it without IPC.
    """

    def __init__(self, shared_value, window_seconds: float = 10.0) -> None:
        super().__init__(window_seconds)
        self._shared_value = shared_value

    def update(self, now: float | None = None) -> None:
        super().update(now)
        self._shared_value.value = self.eps(now)
