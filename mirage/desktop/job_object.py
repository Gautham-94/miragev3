"""Groups every process this app ever spawns (transitively) into one Windows Job Object
with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so a hard kill of the top-level mirage_app.exe
process (Task Manager "End task", an unhandled crash, a forced Windows shutdown/logoff)
takes every descendant down with it too.

Why this matters for a client install specifically: during this project's own
development, leftover ffmpeg.exe/go2rtc.exe/python worker processes surviving an
ungraceful exit were cleaned up by hand (taskkill) more than once. A client has no such
recourse -- an orphaned ffmpeg.exe silently holding a camera's RTSP connection open, or
a leftover go2rtc.exe still bound to its ports, would just look like "the app is broken"
on the next launch, with no obvious cause and no terminal to diagnose it from.

Mechanism: Windows guarantees that every process still assigned to a job is terminated
the instant the job object's last open handle is closed. Handles are per-process kernel
objects that Windows closes automatically on process exit, by ANY means (normal exit,
crash, TerminateProcess, a forced shutdown) -- so as long as this process holds the only
handle to the job and never closes it itself, "this process is gone" and "everything in
its job is gone" become the same event, enforced by the OS, not by any cleanup code here
that could itself fail to run.

Only the TOP-level process needs to call enable_kill_on_exit() (see
mirage/desktop/launcher.py's run_desktop_app(), which calls this before starting the API,
supervisor, or webview window -- i.e. before anything gets a chance to spawn a child).
Every process created afterwards is automatically added to the same job by Windows,
including grandchildren several layers down (the supervisor's pipeline-worker subprocess,
that worker's own multiprocessing-spawned DetectorProcess/CameraTracker/SpeciesProcess
children, and the ffmpeg/go2rtc subprocesses those start) -- nothing in this codebase's
subprocess.Popen/multiprocessing call sites passes CREATE_BREAKAWAY_FROM_JOB, which is
the only thing that would opt a child out of this.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger("mirage.desktop")

# Kept alive for the lifetime of the process -- letting this get garbage-collected would
# close the handle early, which (per the KILL_ON_JOB_CLOSE flag below) would immediately
# terminate every process in the job while the app itself is still trying to run.
_job_handle: int | None = None

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def enable_kill_on_exit() -> None:
    """No-op on non-Windows (this app is Windows-only in its packaged form, but this
    module stays importable for local dev/testing on any platform). Never raises --
    a failure here (e.g. this process is already in a non-nestable job, observed on some
    CI runners/containers) degrades to "no guaranteed cleanup on crash", not a startup
    failure; logged as a warning so it's visible without being fatal.
    """
    global _job_handle
    if sys.platform != "win32":
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # HANDLE is pointer-sized (64-bit on x64 Windows). Without explicit restype/argtypes,
    # ctypes defaults every return value and argument to a 32-bit c_int -- which silently
    # TRUNCATES a 64-bit handle instead of raising, so this looked fine right up until
    # AssignProcessToJobObject rejected the mangled value with ERROR_INVALID_HANDLE.
    # Confirmed live: every one of these calls "succeeded" (non-null/non-zero returns)
    # except the final AssignProcessToJobObject, which is exactly the failure mode a
    # truncated GetCurrentProcess() pseudo-handle (real value: -1, i.e. all 64 bits set)
    # produces once cast down to 32 bits and back up incorrectly.
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    try:
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

        current_process = kernel32.GetCurrentProcess()
        ok = kernel32.AssignProcessToJobObject(job, current_process)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

        _job_handle = job
        logger.info("desktop: child-process job object created (kill-on-close enabled)")
    except OSError as e:
        logger.warning(
            "desktop: could not set up job object for child-process cleanup (%s) -- "
            "the app will still run, but a crash/force-kill may leave child processes "
            "(ffmpeg/go2rtc/worker) running", e,
        )
