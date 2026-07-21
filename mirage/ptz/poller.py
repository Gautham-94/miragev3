"""PtzPoller: a background thread run INSIDE each PTZ-enabled camera's own
CameraTracker process (see mirage.tracking.camera_tracker.camera_tracker_main) --
NOT a separate OS process. CameraOrchestrator.process_frame already runs in this same
process, so a thread-local, Lock-guarded flag is all a per-frame "is this camera
currently moving" check needs -- no multiprocessing.Value/Queue/IPC required, since
both the writer (this thread, polling ONVIF GetStatus) and the reader
(CameraOrchestrator, via the ptz_moving_fn closure this class exposes) live in the same
process.

Also drives the optional preset patrol loop (cycling through camera.ptz.patrol_presets
on a timer) when camera.ptz.patrol_enabled -- deliberately part of the SAME thread/loop
as status polling, not a second thread, since patrol is just "sometimes also issue a
GotoPreset command," which shares the exact same async ONVIF connection this thread
already holds open.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from mirage.config.schema import PtzConfig
from mirage.ptz.client import PtzClient

logger = logging.getLogger(__name__)

STATUS_POLL_INTERVAL_SECONDS = 1.0
# How long to wait before retrying after a failed connect()/poll -- real hardware over
# SOAP can go briefly unreachable (network blip, camera reboot); backing off avoids a
# tight failure loop hammering the device, distinct from resolve_onvif_stream's one-shot
# try/except-then-raise (mirage/api/routers/onvif.py), which is fine for a single
# request but wrong for a thread that must keep running indefinitely.
RECONNECT_BACKOFF_SECONDS = 5.0


class PtzPoller:
    """One instance per PTZ-enabled camera, owned by that camera's CameraTracker
    process. Call start() once (spawns the background thread), moving_fn() any number
    of times per frame from CameraOrchestrator, and stop() at process shutdown.
    """

    def __init__(self, camera_name: str, ptz_config: PtzConfig, client_factory=None) -> None:
        self.camera_name = camera_name
        self.ptz_config = ptz_config
        # Overridable for tests -- default builds a real PtzClient (real ONVIF network
        # calls); tests inject a fake client here instead of mocking the onvif library.
        self._client_factory = client_factory or (lambda: PtzClient(
            host=ptz_config.onvif_host, port=ptz_config.onvif_port,
            username=ptz_config.onvif_username, password=ptz_config.onvif_password,
        ))
        self._moving = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"ptz:{self.camera_name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def is_moving(self) -> bool:
        """Thread-safe read of the current moving state -- this is the ptz_moving_fn
        closure CameraOrchestrator polls once per frame (see
        mirage.tracking.camera_tracker.camera_tracker_main's PtzPoller wiring).
        """
        with self._lock:
            return self._moving

    def _set_moving(self, value: bool) -> None:
        with self._lock:
            self._moving = value

    def _run(self) -> None:
        """Runs its own asyncio event loop on this thread -- PtzClient's methods are
        all async (matching mirage/api/routers/onvif.py's existing async ONVIF call
        style), but this thread itself is a plain threading.Thread, not a coroutine.
        """
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        client = self._client_factory()
        last_patrol_advance = 0.0
        patrol_index = 0

        while not self._stop_event.is_set():
            try:
                await client.connect()
            except Exception:
                logger.exception("%s: ptz connect failed, retrying in %.0fs", self.camera_name, RECONNECT_BACKOFF_SECONDS)
                self._set_moving(False)
                await asyncio.sleep(RECONNECT_BACKOFF_SECONDS)
                continue

            logger.info("%s: ptz poller connected", self.camera_name)
            try:
                while not self._stop_event.is_set():
                    try:
                        status = await client.get_status()
                        self._set_moving(status.moving)
                    except Exception:
                        logger.exception("%s: ptz GetStatus failed", self.camera_name)
                        self._set_moving(False)
                        break  # reconnect from scratch, see outer while loop

                    if self.ptz_config.patrol_enabled and self.ptz_config.patrol_presets and not status.moving:
                        now = time.monotonic()
                        if now - last_patrol_advance >= self.ptz_config.patrol_interval_seconds:
                            preset = self.ptz_config.patrol_presets[patrol_index % len(self.ptz_config.patrol_presets)]
                            try:
                                await client.goto_preset(preset.token)
                            except Exception:
                                logger.exception("%s: ptz patrol goto_preset(%s) failed", self.camera_name, preset.token)
                            else:
                                patrol_index += 1
                                last_patrol_advance = now

                    await asyncio.sleep(STATUS_POLL_INTERVAL_SECONDS)
            finally:
                await client.close()

        self._set_moving(False)
