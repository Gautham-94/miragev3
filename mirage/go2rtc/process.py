"""Go2rtcProcess: launches and supervises the go2rtc binary as a subprocess.

Restreams each enabled camera's raw source into WebRTC/MSE/MJPEG/snapshot-consumable
formats, per section "Live view" of the mirage extension plan (mirrors Frigate's
`init_go2rtc()` in frigate/app.py). Runs as a plain subprocess owned by the main mirage
process (not a multiprocessing.Process -- there's no Python code running inside it to
share memory/queues with, it's an external binary we just launch and supervise).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess as sp

from mirage.config.schema import MirageConfig
from mirage.go2rtc.config import DEFAULT_API_PORT, DEFAULT_WEBRTC_PORT, write_go2rtc_config
from mirage.go2rtc.download import ensure_go2rtc_binary
from mirage.util.proc import windows_no_console_flags

logger = logging.getLogger(__name__)


class Go2rtcProcess:
    def __init__(
        self,
        config: MirageConfig,
        cache_dir: str,
        bin_dir: str | None = None,
        api_port: int = DEFAULT_API_PORT,
        webrtc_port: int = DEFAULT_WEBRTC_PORT,
        stream_overrides: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.bin_dir = bin_dir
        self.api_port = api_port
        self.webrtc_port = webrtc_port
        self.stream_overrides = stream_overrides
        self._proc: sp.Popen | None = None
        self._config_path = os.path.join(cache_dir, "go2rtc.yaml")

    def start(self) -> None:
        kwargs = {} if self.bin_dir is None else {"bin_dir": self.bin_dir}
        binary_path = ensure_go2rtc_binary(**kwargs)
        write_go2rtc_config(
            self.config, self._config_path, api_port=self.api_port, webrtc_port=self.webrtc_port,
            stream_overrides=self.stream_overrides,
        )

        self._proc = sp.Popen(
            [binary_path, "-config", self._config_path],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            stdin=sp.DEVNULL,
            start_new_session=True,
            creationflags=windows_no_console_flags(),
        )
        logger.info("go2rtc started (pid=%d), api on :%d, webrtc on :%d", self._proc.pid, self.api_port, self.webrtc_port)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc is None:
            return
        if os.name == "nt":
            self._stop_windows(timeout)
        else:
            self._stop_posix(timeout)
        logger.info("go2rtc stopped")

    def _stop_posix(self, timeout: float) -> None:
        # killpg (not just killing self._proc.pid) also reaps go2rtc's OWN children --
        # most importantly the ffmpeg transcode processes it spawns for each camera's
        # low-res "_sub" grid-view stream (see mirage.go2rtc.config.SUB_STREAM_SUFFIX),
        # which start_new_session=True in start() makes members of this same process
        # group. Killing just the go2rtc PID would leave every one of those orphaned.
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self._proc.wait(timeout=timeout)
        except sp.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=5)
            except sp.TimeoutExpired:
                pass

    def _stop_windows(self, timeout: float) -> None:
        """os.killpg/os.getpgid don't exist on Windows at all (confirmed live: this
        used to raise AttributeError -- not caught by the POSIX code's own
        `except ProcessLookupError`, silently aborting stop() entirely) -- start()'s
        start_new_session=True doesn't give Windows an equivalent process-group handle
        to target either. `taskkill /T` is the pragmatic Windows equivalent: /T walks
        the live process tree by PID at kill time and terminates every descendant,
        which is what's actually needed here since go2rtc spawns its own ffmpeg
        transcode children for the "_sub" grid-view streams (same reason _stop_posix
        uses killpg over a plain terminate() -- see its own docstring). No graceful
        SIGTERM-equivalent negotiation window on Windows either way (TerminateProcess
        is unconditional), so there's no softer first attempt worth making before /F.
        """
        try:
            sp.run(
                ["taskkill", "/F", "/T", "/PID", str(self._proc.pid)],
                capture_output=True, timeout=timeout,
            )
        except (sp.TimeoutExpired, OSError) as e:
            logger.warning("taskkill failed for go2rtc tree (pid=%d): %s", self._proc.pid, e)
        try:
            self._proc.wait(timeout=5)
        except sp.TimeoutExpired:
            pass
