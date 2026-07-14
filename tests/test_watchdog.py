from __future__ import annotations

import time

from mirage.watchdog import MAX_RESTARTS, RESTART_WINDOW_SECONDS, ServiceWatchdog


class FakeProcess:
    """crash_immediately=True models a process that dies right after start() is called
    (before the watchdog's next tick would observe it as alive) -- used to simulate a
    genuine crash-loop, since a real freshly-started multiprocessing.Process IS alive
    immediately after start() (calling start() is exactly what makes is_alive() true),
    so a factory whose replacement process should also be treated as "already dead by
    the time the watchdog checks it again" needs to explicitly model that, rather than
    just never going alive at all (which doesn't correspond to any real process
    lifecycle and produced a misleading test failure during development -- see
    IMPLEMENTATION_NOTES.md).
    """

    def __init__(self, alive: bool = True, exitcode: int | None = None, crash_immediately: bool = False):
        self._alive = alive
        self.exitcode = exitcode
        self.closed = False
        self._crash_immediately = crash_immediately

    def is_alive(self) -> bool:
        return self._alive

    def close(self) -> None:
        self.closed = True

    def start(self) -> None:
        self._alive = not self._crash_immediately


def test_watchdog_does_nothing_while_process_alive():
    wd = ServiceWatchdog()
    proc = FakeProcess(alive=True)
    restarted = []
    wd.register("svc", proc, factory=lambda: FakeProcess(), on_restart=lambda p: restarted.append(p))

    wd._check_all()

    assert restarted == []


def test_watchdog_does_not_restart_clean_exit():
    wd = ServiceWatchdog()
    proc = FakeProcess(alive=False, exitcode=0)
    restarted = []
    wd.register("svc", proc, factory=lambda: FakeProcess(), on_restart=lambda p: restarted.append(p))

    wd._check_all()

    assert restarted == []


def test_watchdog_restarts_crashed_process():
    wd = ServiceWatchdog()
    proc = FakeProcess(alive=False, exitcode=1)
    new_proc_holder = []

    def factory():
        p = FakeProcess()
        new_proc_holder.append(p)
        return p

    restarted = []
    wd.register("svc", proc, factory=factory, on_restart=lambda p: restarted.append(p))

    wd._check_all()

    assert len(restarted) == 1
    assert restarted[0] is new_proc_holder[0]
    assert proc.closed is True


def test_watchdog_gives_up_after_max_restarts_within_window():
    wd = ServiceWatchdog()
    dead_proc = FakeProcess(alive=False, exitcode=1)
    restart_count = [0]

    def factory():
        restart_count[0] += 1
        # Simulates a genuine crash-loop: the replacement process starts (briefly
        # alive, like a real process would be) but is modeled here as already dead by
        # the NEXT watchdog tick, since exitcode=1/crash_immediately semantics only
        # affect what start() does -- is_alive() is checked on the FOLLOWING _check_all()
        # call, at which point this fake needs to report dead again to simulate a repeat
        # crash. Flip _alive back to False right after start() would have run by
        # overriding start() via crash_immediately.
        return FakeProcess(alive=False, exitcode=1, crash_immediately=True)

    wd.register("svc", dead_proc, factory=factory, on_restart=lambda p: None)

    for _ in range(MAX_RESTARTS + 3):
        wd._check_all()

    # Should have stopped restarting once MAX_RESTARTS was hit within the window.
    assert restart_count[0] == MAX_RESTARTS


def test_watchdog_restart_rate_gate_uses_actual_time_window():
    wd = ServiceWatchdog()
    entry_holder = {}

    def factory():
        return FakeProcess(alive=False, exitcode=1)

    dead_proc = FakeProcess(alive=False, exitcode=1)
    wd.register("svc", dead_proc, factory=factory, on_restart=lambda p: None)

    with wd._lock:
        entry = wd._entries["svc"]

    # Simulate MAX_RESTARTS restarts that happened long ago (outside the window).
    old_time = time.time() - RESTART_WINDOW_SECONDS - 10
    for _ in range(MAX_RESTARTS):
        entry.restart_timestamps.append(old_time)

    assert wd._is_restarting_too_fast(entry, time.time()) is False


def test_stop_blocks_until_thread_exits_and_prevents_late_restart():
    # Regression test: MirageApp.stop() calls watchdog.stop() BEFORE terminating any
    # monitored process, relying on stop() to guarantee no restart can happen after it
    # returns. Before this fix, stop() only set the stop event and returned immediately
    # -- the watchdog thread's own wait() loop could still be asleep for up to
    # tick_seconds, so it could wake up, see a process the caller was in the middle of
    # tearing down as "dead", and restart it right in the middle of shutdown (observed
    # for real running the CLI: "detector:general died (exitcode=1), restarting" logged
    # seconds after Ctrl+C). Using a short tick_seconds here so the race window (if the
    # fix regressed) would reliably reproduce within the test's own timeout.
    wd = ServiceWatchdog(tick_seconds=0.2)
    proc = FakeProcess(alive=True)
    restarted = []
    wd.register("svc", proc, factory=lambda: FakeProcess(), on_restart=lambda p: restarted.append(p))
    wd.start()

    time.sleep(0.05)  # ensure the thread has entered its wait() before we stop it
    proc._alive = False
    proc.exitcode = 1

    wd.stop()

    assert not wd.is_alive(), "stop() must block until the watchdog thread has actually exited"
    assert restarted == [], "no restart should happen once stop() has returned"


def test_multiple_registered_services_checked_independently():
    wd = ServiceWatchdog()
    alive_proc = FakeProcess(alive=True)
    dead_proc = FakeProcess(alive=False, exitcode=1)
    restarted = []

    wd.register("alive_svc", alive_proc, factory=lambda: FakeProcess(), on_restart=lambda p: restarted.append(("alive_svc", p)))
    wd.register("dead_svc", dead_proc, factory=lambda: FakeProcess(), on_restart=lambda p: restarted.append(("dead_svc", p)))

    wd._check_all()

    assert len(restarted) == 1
    assert restarted[0][0] == "dead_svc"
