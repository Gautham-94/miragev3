"""Supervises the `python -m mirage` pipeline subprocess so the frontend's "Apply
changes" button (Manage Cameras / Add Camera / Manage Detectors / Queries pages) can
request a clean restart without the user needing a terminal.

Run this INSTEAD OF `python -m mirage` directly: `python -m mirage.supervisor [same
args as python -m mirage]` -- every arg is passed straight through to the pipeline
subprocess unmodified (see main() below), so existing run instructions/scripts only need
the module name changed.

Hot-reload (automatically detecting and applying config changes to an already-running
pipeline) was built, tested, and then explicitly reverted per user request: it caused a
real, hard-to-diagnose issue in practice (TODO_FIX_LIST.md item 2/11) -- a camera whose
config changed via hot-reload silently stopped producing new detections after its
tracker was restarted in-process, with no error logged. An explicit, user-triggered full
process restart is simpler to reason about and matches how this system already behaved
before hot-reload was attempted (the ONLY thing this file changes is making that restart
a UI button instead of a terminal command).

IPC with mirage.api (a separate, independently-started process) is via small files
under the same cache_dir every other IPC mechanism in this codebase already uses (see
mirage/const.py's ipc_addr) -- no new dependency (no socket, no message broker):
  - STATUS_FILENAME: this process writes its current state as JSON after every
    transition (starting/running/stopping/stopped/crashed) -- mirage/api/routers/system.py
    reads it to answer GET /api/system/status for the frontend to poll.
  - RESTART_REQUEST_FILENAME: mirage/api/routers/system.py writes an empty sentinel file
    here when the user presses "Apply changes" (POST /api/system/restart) -- this
    process polls for that file's existence once a second and, when found, deletes it
    and performs a restart (ask the running pipeline to stop, wait for clean exit,
    start a fresh one).
  - STOP_REQUEST_FILENAME: how "ask the running pipeline to stop" above is actually
    implemented on Windows -- see _stop_pipeline()'s own comment for why this exists
    (Popen.send_signal(SIGTERM) doesn't work the way it does on POSIX there) and
    mirage.__main__'s main loop for the other end of this same file-based handshake.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from mirage.const import CACHE_DIR, ensure_dirs

logger = logging.getLogger("mirage.supervisor")

STATUS_FILENAME = "supervisor_status.json"
RESTART_REQUEST_FILENAME = "restart_requested"
STOP_REQUEST_FILENAME = "stop_requested"

POLL_INTERVAL_SECONDS = 1.0
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 20.0


class PipelineSupervisor:
    def __init__(
        self, pipeline_args: list[str], cache_dir: str = CACHE_DIR, pipeline_command: list[str] | None = None,
    ) -> None:
        """`pipeline_command`, if given, REPLACES the default `[sys.executable, "-m",
        "mirage", *pipeline_args]` command entirely -- exists so tests can point this at
        a small throwaway script instead of the real (heavyweight, multi-process)
        pipeline, without any PYTHONPATH/module-shadowing tricks (which proved unsafe:
        `mirage` is an editable install resolvable from the project directory itself,
        which can win the resolution race unpredictably and actually spawn the real
        pipeline during a test run). Real usage (mirage.supervisor's own main()) never
        passes this.
        """
        self.pipeline_args = pipeline_args
        self.cache_dir = cache_dir
        self.pipeline_command = pipeline_command or [sys.executable, "-m", "mirage", *pipeline_args]
        self.status_path = Path(cache_dir) / STATUS_FILENAME
        self.restart_request_path = Path(cache_dir) / RESTART_REQUEST_FILENAME
        self.stop_request_path = Path(cache_dir) / STOP_REQUEST_FILENAME
        self.process: subprocess.Popen | None = None
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Status file
    # ------------------------------------------------------------------

    def _write_status(self, state: str, pid: int | None = None) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": state, "pid": pid, "updated_at": time.time()}
        tmp_path = self.status_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload))
        tmp_path.replace(self.status_path)  # atomic on POSIX -- readers never see a partial write

    # ------------------------------------------------------------------
    # Pipeline process lifecycle
    # ------------------------------------------------------------------

    def _start_pipeline(self) -> None:
        # A stop-request file can only ever be left behind by an instance that got hard-
        # killed before reaching the unlink() in mirage.__main__'s own loop (e.g. the
        # GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS fallback below actually firing) -- if left in
        # place, the brand-new pipeline process spawned right below would see it on its
        # very first poll and immediately shut itself back down. Same stale-request
        # cleanup pattern run() already does for restart_request_path.
        self.stop_request_path.unlink(missing_ok=True)
        self._write_status("starting")
        logger.info("starting pipeline: %s", " ".join(self.pipeline_command))
        self.process = subprocess.Popen(self.pipeline_command)
        self._write_status("running", pid=self.process.pid)

    def _stop_pipeline(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self._write_status("stopping", pid=self.process.pid)
        logger.info("stopping pipeline (pid=%d)", self.process.pid)
        if sys.platform == "win32":
            # Popen.send_signal(SIGTERM) on Windows is NOT a real, catchable signal --
            # CPython implements it as a plain TerminateProcess() call (see the stdlib
            # subprocess docs' own note on this), which is an unconditional, immediate
            # OS-level kill that never gives the target process a chance to run its
            # registered signal handler, let alone any cleanup code. Confirmed live:
            # mirage.__main__'s SIGTERM handler IS correctly written and would work
            # fine on POSIX, but on Windows it simply never fires -- the pipeline
            # process was dying in under 30ms every restart (a hard kill, not a clean
            # exit) and every child it owns (go2rtc.exe, ffmpeg.exe, every detector/
            # tracker/species worker process) was being orphaned on every single
            # "Apply changes", silently accumulating -- e.g. three live go2rtc.exe
            # processes all fighting over the same ports after three restarts, the
            # newest ones unable to bind and the oldest one left serving a stale
            # config that never picked up a just-added camera.
            #
            # The fix: a file-based "please stop" request, the same IPC pattern this
            # class already uses for restart requests -- mirage.__main__'s own main
            # loop polls for this file (already polling every 0.5s regardless) and
            # exits its loop when it appears, running the SAME graceful app.stop()
            # path a real POSIX SIGTERM would have triggered. Deliberately NOT used on
            # POSIX too "for consistency" -- real SIGTERM delivery there is simpler,
            # already proven correct, and involves strictly fewer moving parts than a
            # polling handshake.
            self.stop_request_path.parent.mkdir(parents=True, exist_ok=True)
            self.stop_request_path.write_text("")
        else:
            self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("pipeline did not exit within %ss, killing", GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS)
            self.process.kill()
            self.process.wait(timeout=5)
        logger.info("pipeline stopped (exit code %s)", self.process.returncode)

    def _restart_pipeline(self) -> None:
        self._stop_pipeline()
        self._start_pipeline()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Non-signal equivalent of SIGINT/SIGTERM for embedders that don't have a
        terminal to send a real OS signal from -- e.g. mirage/desktop/launcher.py's
        window-closed handler, driving this same run() loop from a background thread
        where register_signal_handlers() (main-thread only) was never called.
        """
        self._stop_requested = True

    def register_signal_handlers(self) -> None:
        """Only callable from the process's real main thread (a hard Python/OS
        constraint on signal.signal itself) -- kept separate from run()'s poll loop
        specifically so tests can exercise the loop's restart/crash-detection logic by
        calling run() from a background thread (pytest's own main thread is unavailable
        for this), without needing a real OS-level SIGINT/SIGTERM to reach the test.
        main() (the real `python -m mirage.supervisor` entrypoint) always calls this
        before run(); nothing else should skip it.
        """
        def _handle_signal(signum, frame):
            logger.info("supervisor received signal %s, shutting down", signal.Signals(signum).name)
            self._stop_requested = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def run(self) -> None:
        ensure_dirs(self.cache_dir)
        self.restart_request_path.unlink(missing_ok=True)  # stale request from a prior run

        self._start_pipeline()
        crashed = False

        try:
            while not self._stop_requested:
                time.sleep(POLL_INTERVAL_SECONDS)

                if self.restart_request_path.exists():
                    self.restart_request_path.unlink(missing_ok=True)
                    logger.info("restart requested via %s", self.restart_request_path)
                    self._restart_pipeline()
                    continue

                if self.process is not None and self.process.poll() is not None:
                    # The pipeline exited on its own (crash, or someone sent it a
                    # signal directly) -- surface this honestly rather than silently
                    # restarting behind the user's back; TODO_FIX_LIST.md's own
                    # ServiceWatchdog already handles crash-restart for INDIVIDUAL
                    # detector processes inside MirageApp, this is a different,
                    # coarser layer (the whole pipeline process) that intentionally
                    # does NOT auto-restart, since an unexpected exit is exactly the
                    # kind of thing that should surface as "crashed" in the UI, not be
                    # silently papered over.
                    logger.error("pipeline exited unexpectedly (code %s)", self.process.returncode)
                    crashed = True
                    break
        finally:
            # _stop_pipeline() is a no-op if the process already exited on its own (the
            # crashed=True path) -- self.process.poll() is not None guards its early
            # return. The terminal status write happens exactly ONCE, here, choosing
            # "crashed" vs "stopped" based on how the loop actually ended, rather than
            # letting a later unconditional write silently overwrite "crashed" with
            # "stopped" a moment after it was set (the original version of this method
            # did exactly that -- caught by a real test, see test_supervisor.py).
            self._stop_pipeline()
            self._write_status("crashed" if crashed else "stopped", pid=None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    supervisor = PipelineSupervisor(pipeline_args=sys.argv[1:])
    supervisor.register_signal_handlers()
    supervisor.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
