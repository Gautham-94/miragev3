"""Hang-proof helper for running a short-lived subprocess and capturing its output.

Do NOT use `subprocess.run(..., capture_output=True, timeout=N)` for this: on Windows,
if the first internal `communicate(timeout=N)` call times out, CPython's own retry path
(see `subprocess.run`'s source) kills the process and then calls a SECOND,
*unbounded* `communicate()` to drain whatever's left in the pipes before re-raising
`TimeoutExpired`. That second call can hang forever -- confirmed in production here: a
`probe_duration()` call (see mirage.recording.segments) wedged the entire recording-
maintainer thread for 20+ minutes with the target ffprobe.exe process already gone,
stuck in `Popen._communicate`'s pipe-reader-thread `.join()`, silently halting all
recording promotion for every camera with no exception ever raised (`timeout=N` was
passed and simply not honored). Piping stdout/stderr to real temp files instead of
`PIPE` sidesteps this whole failure class: `Popen.wait(timeout=N)` only waits on the
process handle itself, never on a pipe reaching EOF, so a plain `kill()` truly bounds
the total wall-clock time regardless of what's holding any pipe handles open.
"""

from __future__ import annotations

import logging
import subprocess as sp
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def windows_no_console_flags() -> int:
    """subprocess.Popen `creationflags` for suppressing a console window when launching
    a console-subsystem child binary (ffmpeg, ffprobe, go2rtc, the SpeciesNet worker
    exe) from the packaged Windows desktop app. Redirecting stdout/stderr/stdin to
    DEVNULL or PIPE only controls where the child's I/O goes -- it does NOT stop Windows
    from allocating and SHOWING a new console window for a console-subsystem process;
    CREATE_NO_WINDOW is the separate, additional flag that suppresses the window itself.

    Confirmed live: go2rtc.exe (a plain Go build, no windowsgui manifest) opened a
    visible console -- hosted in Windows Terminal, the Windows 11 default terminal
    host -- on every launch of the packaged desktop app, despite its own Popen call
    already redirecting all three streams to DEVNULL. Every Popen call in this codebase
    that launches an external console-subsystem binary needs this, not just go2rtc's.

    A no-op (0) everywhere except win32, where this concept doesn't exist -- safe to
    pass as `creationflags=` unconditionally from any call site, dev-tree or frozen.
    """
    return sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0


@dataclass
class ProcResult:
    returncode: int
    stdout: str
    stderr: str


def run_capturing(cmd: list[str], timeout: float) -> ProcResult | None:
    """Runs `cmd` to completion, capturing stdout/stderr as text. Returns None if the
    process couldn't be started, didn't exit within `timeout`, or was killed -- in every
    case the caller's existing "treat as failed, retry or discard" handling applies, same
    as the old `except (TimeoutExpired, OSError): return None` contract this replaces.
    """
    with (
        tempfile.TemporaryFile() as stdout_f,
        tempfile.TemporaryFile() as stderr_f,
    ):
        try:
            proc = sp.Popen(
                cmd, stdout=stdout_f, stderr=stderr_f, stdin=sp.DEVNULL,
                creationflags=windows_no_console_flags(),
            )
        except OSError as e:
            logger.error("failed to start %r: %s", cmd, e)
            return None

        try:
            proc.wait(timeout=timeout)
        except sp.TimeoutExpired:
            proc.kill()
            proc.wait()  # bounded: TerminateProcess is unconditional, no pipe involved
            logger.error("%r timed out after %ss, killed", cmd, timeout)
            return None

        stdout_f.seek(0)
        stderr_f.seek(0)
        return ProcResult(
            returncode=proc.returncode,
            stdout=stdout_f.read().decode(errors="replace"),
            stderr=stderr_f.read().decode(errors="replace"),
        )
