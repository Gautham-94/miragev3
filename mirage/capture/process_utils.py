"""ffmpeg subprocess launch/stop helpers.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.3 / 1.7.
"""

from __future__ import annotations

import logging
import subprocess as sp

logger = logging.getLogger(__name__)


def start_ffmpeg(cmd: list[str], needs_stdout_pipe: bool, frame_size: int = 0, stderr=None) -> sp.Popen:
    if needs_stdout_pipe:
        return sp.Popen(
            cmd,
            stdout=sp.PIPE,
            stderr=stderr if stderr is not None else sp.DEVNULL,
            stdin=sp.DEVNULL,
            bufsize=frame_size * 10,  # room for 10 raw frames in the pipe buffer
            start_new_session=True,  # own process group: clean signal handling
        )
    return sp.Popen(
        cmd,
        stdout=sp.DEVNULL,
        stderr=stderr if stderr is not None else sp.DEVNULL,
        stdin=sp.DEVNULL,
        start_new_session=True,
    )


def stop_ffmpeg(process: sp.Popen, timeout: int = 30, drain_output: bool = True) -> None:
    """Section 1.7 graceful-stop helper. If drain_output is False, skip trying to read any
    remaining piped stdout (fast path used for the runaway-fps hard-restart case).
    """
    process.terminate()
    try:
        if drain_output:
            process.communicate(timeout=timeout)
        else:
            process.wait(timeout=timeout)
    except sp.TimeoutExpired:
        process.kill()
        try:
            process.communicate(timeout=5)
        except sp.TimeoutExpired:
            pass
