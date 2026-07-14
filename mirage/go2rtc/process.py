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
        )
        logger.info("go2rtc started (pid=%d), api on :%d, webrtc on :%d", self._proc.pid, self.api_port, self.webrtc_port)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc is None:
            return
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
        logger.info("go2rtc stopped")
