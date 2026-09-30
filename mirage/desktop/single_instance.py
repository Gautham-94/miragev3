"""Prevents a second copy of the desktop app from running at the same time.

Why this matters for a client install specifically: a non-technical user who double-
clicks the desktop icon twice (plausible if the app is slow to open, or already running
minimized) would otherwise end up with two independent pipelines writing to the SAME
SQLite database and the SAME recording directory concurrently -- a real data-corruption
risk, not just wasted resources. A dev running from a terminal would notice two windows
and close one; a client would not.

Mechanism: a named Windows mutex, created once at process startup. The mutex name is
global to the session (not per-user) so this catches "already running" even if launched
via a scheduled task or a different login session on the same machine. CreateMutexW
itself atomically creates-or-opens the named object -- checking GetLastError() ==
ERROR_ALREADY_EXISTS afterwards is the documented, race-free way to detect "someone else
already holds this name", not a separate check-then-create (which would race).
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger("mirage.desktop")

# Fixed, unique-enough name -- Windows mutex names live in a shared namespace, but a
# GUID-derived suffix makes an accidental collision with an unrelated app effectively
# impossible. "Global\\" makes this visible across sessions (Terminal Services aware),
# not just within the current login session.
_MUTEX_NAME = r"Global\Mirage_SingleInstance_8F1E3A2C-6B4D-4E9A-9F0D-2C7A5B6E1D34"
_ERROR_ALREADY_EXISTS = 183

# Kept alive for the process lifetime -- Windows releases a mutex automatically when its
# owning process exits (by any means), which is exactly the desired "instance slot freed
# up the moment the app is truly gone" behavior; closing it ourselves early would defeat
# that, so this reference just needs to not be garbage-collected before then.
_mutex_handle: int | None = None


def acquire_or_exit() -> None:
    """No-op on non-Windows. On Windows: if another instance already holds the mutex,
    shows a native message box explaining why and exits the process immediately (before
    anything -- API, pipeline supervisor, webview window -- has been started), so the
    second launch never touches the database or recording directory at all.
    """
    global _mutex_handle
    if sys.platform != "win32":
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Explicit restype -- CreateMutexW returns a pointer-sized HANDLE; without this,
    # ctypes' default 32-bit c_int return type risks mishandling it on 64-bit Windows
    # (see mirage/desktop/job_object.py's own fix for the same class of bug, which did
    # bite in practice for a different handle there).
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    already_running = ctypes.get_last_error() == _ERROR_ALREADY_EXISTS

    if not handle:
        # Could not even create the mutex (exotic failure) -- fail open rather than
        # blocking every future launch of the app over a guard that itself broke.
        logger.warning("desktop: could not create single-instance mutex, continuing anyway")
        return

    _mutex_handle = handle

    if already_running:
        logger.info("desktop: another instance is already running, exiting")
        ctypes.windll.user32.MessageBoxW(
            0,
            "Mirage is already running.\n\nCheck your taskbar or system tray -- "
            "only one copy can run at a time.",
            "Mirage",
            0x30,  # MB_ICONWARNING
        )
        sys.exit(0)
