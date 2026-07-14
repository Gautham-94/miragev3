"""CameraWatchdog: supervises the ffmpeg subprocess(es) for one camera.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.7.

Runs as a thread inside the per-camera CameraCapture process. Ticks once per second,
checking for a dead capture thread, runaway fps, stalled frames, and dead "other role"
(e.g. record) ffmpeg processes -- restarting whichever piece failed, with a backoff gate so
a persistently-crashing camera doesn't spin-loop.
"""

from __future__ import annotations

import logging
import threading
import time

from mirage.capture.capture import capture_frames
from mirage.capture.ffmpeg_presets import build_all_ffmpeg_cmds
from mirage.capture.process_utils import start_ffmpeg, stop_ffmpeg
from mirage.config.schema import CameraConfig, CameraRole
from mirage.util.logpipe import LogPipe
from mirage.util.rate import BoundedWindowCounter, EventsPerSecond
from mirage.util.shm import SharedMemoryFrameManager

logger = logging.getLogger(__name__)

FPS_OVERFLOW_MARGIN = 10
FPS_OVERFLOW_TICKS_REQUIRED = 3
STALLED_FRAME_SECONDS = 20
RECORD_STALE_SECONDS = 120
RECORD_STARTUP_GRACE_SECONDS = 90


class CameraWatchdog(threading.Thread):
    def __init__(
        self,
        camera: CameraConfig,
        cache_dir: str,
        ring_depth: int,
        frame_manager: SharedMemoryFrameManager,
        frame_queue,
        current_frame_ts,
        camera_fps: EventsPerSecond,
        skipped_fps: EventsPerSecond,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name=f"watchdog-{camera.name}")
        self.camera = camera
        self.cache_dir = cache_dir
        self.ring_depth = ring_depth
        self.frame_manager = frame_manager
        self.frame_queue = frame_queue
        self.current_frame_ts = current_frame_ts
        self.camera_fps = camera_fps
        self.skipped_fps = skipped_fps
        self.stop_event = stop_event

        self.frame_size = camera.frame_size_bytes
        self.retry_interval = camera.ffmpeg.retry_interval

        self.detect_process = None
        self.detect_logpipe: LogPipe | None = None
        self.capture_thread: threading.Thread | None = None
        self._last_restart_time = 0.0
        self._fps_overflow_ticks = 0

        self.other_processes: list[dict] = []  # [{"roles", "cmd", "process", "logpipe"}]
        self._last_record_activity_time = time.time()

        self.reconnects_last_hour = BoundedWindowCounter(3600)
        self.stalls_last_hour = BoundedWindowCounter(3600)

    # ---------------------------------------------------------------------------------
    # ffmpeg lifecycle
    # ---------------------------------------------------------------------------------

    def start_all_ffmpeg(self) -> None:
        cmds = build_all_ffmpeg_cmds(self.camera, self.cache_dir)
        for entry in cmds:
            if CameraRole.detect in entry["roles"]:
                self._start_detect_ffmpeg(entry["cmd"])
            else:
                logpipe = LogPipe(f"ffmpeg.{self.camera.name}.{'_'.join(sorted(r.value for r in entry['roles']))}")
                logpipe.start()
                process = start_ffmpeg(entry["cmd"], needs_stdout_pipe=False, stderr=logpipe)
                self.other_processes.append({"roles": entry["roles"], "cmd": entry["cmd"], "process": process, "logpipe": logpipe})

    def _start_detect_ffmpeg(self, cmd: list[str]) -> None:
        logpipe = LogPipe(f"ffmpeg.{self.camera.name}.detect")
        logpipe.start()
        process = start_ffmpeg(cmd, needs_stdout_pipe=True, frame_size=self.frame_size, stderr=logpipe)
        self.detect_process = process
        self.detect_logpipe = logpipe

        thread_stop = threading.Event()
        self._capture_stop = thread_stop
        self.capture_thread = threading.Thread(
            target=capture_frames,
            name=f"capture-{self.camera.name}",
            kwargs=dict(
                camera_name=self.camera.name,
                ffmpeg_process=process,
                frame_size=self.frame_size,
                ring_depth=self.ring_depth,
                frame_manager=self.frame_manager,
                frame_queue=self.frame_queue,
                current_frame_ts=self.current_frame_ts,
                camera_fps=self.camera_fps,
                skipped_fps=self.skipped_fps,
                stop_event=thread_stop,
            ),
            daemon=True,
        )
        self.capture_thread.start()

    def stop_all_ffmpeg(self, drain_output: bool = True) -> None:
        if self.detect_process is not None:
            if hasattr(self, "_capture_stop"):
                self._capture_stop.set()
            stop_ffmpeg(self.detect_process, drain_output=drain_output)
            if self.detect_logpipe is not None:
                self.detect_logpipe.dump()
                self.detect_logpipe.close()
            self.detect_process = None
        for entry in self.other_processes:
            stop_ffmpeg(entry["process"], drain_output=drain_output)
            entry["logpipe"].dump()
            entry["logpipe"].close()
        self.other_processes = []

    def reset_capture_thread(self, drain_output: bool = True) -> None:
        now = time.time()
        if now - self._last_restart_time < self.retry_interval:
            return  # backoff gate -- don't restart faster than retry_interval
        self._last_restart_time = now
        self.reconnects_last_hour.record(now)
        logger.warning("%s: restarting detect ffmpeg process", self.camera.name)

        if self.detect_process is not None:
            if hasattr(self, "_capture_stop"):
                self._capture_stop.set()
            stop_ffmpeg(self.detect_process, drain_output=drain_output)
            if self.detect_logpipe is not None:
                self.detect_logpipe.dump()
                self.detect_logpipe.close()

        cmds = build_all_ffmpeg_cmds(self.camera, self.cache_dir)
        detect_cmd = next((c["cmd"] for c in cmds if CameraRole.detect in c["roles"]), None)
        if detect_cmd is not None:
            self._start_detect_ffmpeg(detect_cmd)
        self._fps_overflow_ticks = 0

    # ---------------------------------------------------------------------------------
    # Supervision loop
    # ---------------------------------------------------------------------------------

    def run(self) -> None:
        self.start_all_ffmpeg()
        while not self.stop_event.wait(1.0):
            self._tick()
        self.stop_all_ffmpeg()

    def _tick(self) -> None:
        now = time.time()

        if self.capture_thread is not None and not self.capture_thread.is_alive():
            logger.error("%s: capture thread died; restarting ffmpeg", self.camera.name)
            self.reset_capture_thread(drain_output=False)
            return

        observed_fps = self.camera_fps.eps(now)
        if observed_fps >= self.camera.detect.fps + FPS_OVERFLOW_MARGIN:
            self._fps_overflow_ticks += 1
            if self._fps_overflow_ticks >= FPS_OVERFLOW_TICKS_REQUIRED:
                logger.error("%s: runaway fps detected (%.1f), force-restarting", self.camera.name, observed_fps)
                self.reset_capture_thread(drain_output=False)
                return
        else:
            self._fps_overflow_ticks = 0

        last_frame_time = self.current_frame_ts.value
        if last_frame_time and (now - last_frame_time > STALLED_FRAME_SECONDS):
            logger.error("%s: no frames received in %ds, restarting", self.camera.name, STALLED_FRAME_SECONDS)
            self.stalls_last_hour.record(now)
            self.reset_capture_thread(drain_output=True)
            return

        self._check_other_processes()

    def _check_other_processes(self) -> None:
        for entry in self.other_processes:
            if entry["process"].poll() is not None:
                logger.error("%s: %s ffmpeg process exited, restarting", self.camera.name, entry["roles"])
                entry["logpipe"].dump()
                entry["logpipe"].close()
                new_logpipe = LogPipe(f"ffmpeg.{self.camera.name}.{'_'.join(sorted(r.value for r in entry['roles']))}")
                new_logpipe.start()
                entry["process"] = start_ffmpeg(entry["cmd"], needs_stdout_pipe=False, stderr=new_logpipe)
                entry["logpipe"] = new_logpipe

    def notify_record_activity(self) -> None:
        """Called by the recording pipeline when a new valid segment is observed for this
        camera, so the watchdog can detect a "silently stuck" (still-running but
        producing nothing) record ffmpeg process -- section 1.7 point 4.
        """
        self._last_record_activity_time = time.time()

    def record_process_seems_stale(self) -> bool:
        if not self.camera.record.enabled:
            return False
        elapsed_since_start = time.time() - self._last_restart_time
        if elapsed_since_start < RECORD_STARTUP_GRACE_SECONDS:
            return False
        return (time.time() - self._last_record_activity_time) > RECORD_STALE_SECONDS
