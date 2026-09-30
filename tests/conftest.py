"""Shared test helpers. Plain functions here (not just fixtures) are importable as
`from tests.conftest import ...` since tests/ is a real package (tests/__init__.py).
"""

from __future__ import annotations

import os
import subprocess as sp


def kill_process_group(proc: sp.Popen, timeout: float = 5.0) -> None:
    """Forcefully kills `proc` AND every child process it has spawned -- most commonly
    a `sh -c "while true; do ffmpeg ...; done"` test-video-server loop and its ffmpeg
    children, or a go2rtc process and its own ffmpeg transcode children (see
    mirage.go2rtc.config.SUB_STREAM_SUFFIX). Plain proc.terminate()/.kill() only ever
    touch `proc` itself, leaving every child orphaned -- every caller starts its
    process with start_new_session=True specifically so this can target the whole
    group/tree at once.

    Cross-platform: os.killpg/os.getpgid (the POSIX way to do this) have no Windows
    equivalent at all -- confirmed live, calling them on Windows used to raise
    AttributeError, aborting teardown before ever reaching proc.wait() below and
    leaking the entire process tree on every single test run (this hit and fixed the
    exact same bug already found and fixed in production code, see
    mirage.go2rtc.process.Go2rtcProcess._stop_windows for the full story). `taskkill
    /T` is the Windows equivalent: it walks the live process tree by PID at kill time,
    so it catches children regardless of how/when they were spawned.
    """
    if os.name == "nt":
        try:
            sp.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=timeout)
        except (sp.TimeoutExpired, OSError):
            pass
    else:
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=timeout)
    except sp.TimeoutExpired:
        pass
