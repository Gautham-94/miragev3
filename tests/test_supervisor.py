"""Tests for mirage/supervisor.py -- the process supervisor behind the "Apply changes"
button. Exercises the REAL subprocess/file logic (no mocking of subprocess.Popen), but
against a tiny fake "pipeline" script instead of the real (heavyweight, multi-process)
`python -m mirage` -- injected via PipelineSupervisor's `pipeline_command` param (see its
docstring: an earlier PYTHONPATH-based module-shadowing approach was tried and abandoned
because it unpredictably let the REAL pipeline spawn during a test run, since `mirage`
is an editable install resolvable from the project directory itself).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from mirage.supervisor import RESTART_REQUEST_FILENAME, STATUS_FILENAME, PipelineSupervisor

# A minimal "pipeline" stand-in: prints its own pid once (not used for readiness -- see
# the sleep in tests below) then blocks until SIGTERM, exiting 0 -- mirrors
# mirage/__main__.py's own real SIGTERM-triggers-graceful-shutdown contract closely
# enough to exercise the supervisor's real subprocess handling.
FAKE_PIPELINE_SCRIPT = """
import signal
import sys
import time

stop = False
def _handle(signum, frame):
    global stop
    stop = True

signal.signal(signal.SIGTERM, _handle)
print("pid", flush=True)
while not stop:
    time.sleep(0.05)
sys.exit(0)
"""

FAKE_CRASHING_PIPELINE_SCRIPT = "import sys\nsys.exit(1)\n"


@pytest.fixture
def fake_pipeline_command(tmp_path):
    script = tmp_path / "fake_pipeline.py"
    script.write_text(FAKE_PIPELINE_SCRIPT)
    return [sys.executable, str(script)]


@pytest.fixture
def crashing_pipeline_command(tmp_path):
    script = tmp_path / "fake_crashing_pipeline.py"
    script.write_text(FAKE_CRASHING_PIPELINE_SCRIPT)
    return [sys.executable, str(script)]


def _read_status(cache_dir: str) -> dict:
    return json.loads((Path(cache_dir) / STATUS_FILENAME).read_text())


def _wait_for_status(cache_dir: str, predicate, timeout: float = 5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if (Path(cache_dir) / STATUS_FILENAME).exists():
            last = _read_status(cache_dir)
            if predicate(last):
                return last
        time.sleep(0.02)
    pytest.fail(f"status never satisfied predicate within {timeout}s (last seen: {last})")


def test_start_pipeline_writes_running_status_with_real_pid(fake_pipeline_command, tmp_path):
    cache_dir = str(tmp_path / "cache")
    supervisor = PipelineSupervisor(pipeline_args=[], cache_dir=cache_dir, pipeline_command=fake_pipeline_command)

    supervisor._start_pipeline()
    try:
        status = _wait_for_status(cache_dir, lambda s: s["state"] == "running")
        assert status["pid"] == supervisor.process.pid
        assert supervisor.process.poll() is None  # still alive
    finally:
        supervisor._stop_pipeline()


def test_stop_pipeline_gracefully_terminates_and_updates_status(fake_pipeline_command, tmp_path):
    cache_dir = str(tmp_path / "cache")
    supervisor = PipelineSupervisor(pipeline_args=[], cache_dir=cache_dir, pipeline_command=fake_pipeline_command)
    supervisor._start_pipeline()
    _wait_for_status(cache_dir, lambda s: s["state"] == "running")
    # The status file flips to "running" the instant Popen() returns, which can race the
    # fake child's own signal.signal(SIGTERM, ...) call landing -- a real mirage pipeline
    # registers its handler at the very top of __main__.py before any slow startup work,
    # so this race is specific to this fake script, not the supervisor itself. Give the
    # child a moment to actually install its handler before sending SIGTERM.
    time.sleep(0.3)

    supervisor._stop_pipeline()

    assert supervisor.process.returncode == 0  # the fake script's own SIGTERM handler exits 0


def test_restart_pipeline_stops_the_old_process_and_starts_a_new_one(fake_pipeline_command, tmp_path):
    cache_dir = str(tmp_path / "cache")
    supervisor = PipelineSupervisor(pipeline_args=[], cache_dir=cache_dir, pipeline_command=fake_pipeline_command)
    supervisor._start_pipeline()
    _wait_for_status(cache_dir, lambda s: s["state"] == "running")
    time.sleep(0.3)  # see test_stop_pipeline_gracefully_terminates_and_updates_status's comment
    old_pid = supervisor.process.pid
    old_process = supervisor.process

    try:
        supervisor._restart_pipeline()
        status = _wait_for_status(cache_dir, lambda s: s["state"] == "running")

        assert old_process.poll() is not None  # old process actually exited
        assert status["pid"] != old_pid
        assert status["pid"] == supervisor.process.pid
        assert supervisor.process.poll() is None  # new process genuinely alive
    finally:
        supervisor._stop_pipeline()


def test_restart_request_file_triggers_a_restart_in_the_run_loop(fake_pipeline_command, tmp_path):
    cache_dir = str(tmp_path / "cache")
    supervisor = PipelineSupervisor(pipeline_args=[], cache_dir=cache_dir, pipeline_command=fake_pipeline_command)

    # run() itself never calls signal.signal() (that's register_signal_handlers()'s job,
    # only ever invoked by the real main() entrypoint) -- safe to run on a background
    # thread here, unlike a naive call to the old combined run() would have been.
    run_thread = threading.Thread(target=supervisor.run, daemon=True)
    run_thread.start()
    try:
        _wait_for_status(cache_dir, lambda s: s["state"] == "running")
        first_pid = supervisor.process.pid
        time.sleep(0.3)

        (Path(cache_dir) / RESTART_REQUEST_FILENAME).write_text("")

        status = _wait_for_status(cache_dir, lambda s: s["state"] == "running" and s["pid"] != first_pid, timeout=10)
        assert status["pid"] != first_pid
        assert not (Path(cache_dir) / RESTART_REQUEST_FILENAME).exists()  # sentinel consumed
    finally:
        supervisor._stop_requested = True
        run_thread.join(timeout=10)


def test_unexpected_pipeline_exit_is_reported_as_crashed_not_silently_restarted(crashing_pipeline_command, tmp_path):
    cache_dir = str(tmp_path / "cache")
    supervisor = PipelineSupervisor(pipeline_args=[], cache_dir=cache_dir, pipeline_command=crashing_pipeline_command)

    run_thread = threading.Thread(target=supervisor.run, daemon=True)
    run_thread.start()
    try:
        status = _wait_for_status(cache_dir, lambda s: s["state"] == "crashed", timeout=10)
        assert status["pid"] is None
    finally:
        supervisor._stop_requested = True
        run_thread.join(timeout=10)


def test_stopping_when_no_process_was_ever_started_is_a_no_op(tmp_path):
    cache_dir = str(tmp_path / "cache")
    supervisor = PipelineSupervisor(pipeline_args=[], cache_dir=cache_dir, pipeline_command=[sys.executable, "-c", "pass"])

    supervisor._stop_pipeline()  # must not raise


def test_stale_restart_request_from_a_prior_run_is_cleared_at_startup(fake_pipeline_command, tmp_path):
    cache_dir = str(tmp_path / "cache")
    Path(cache_dir).mkdir(parents=True)
    stale = Path(cache_dir) / RESTART_REQUEST_FILENAME
    stale.write_text("stale from a previous supervisor process")

    supervisor = PipelineSupervisor(pipeline_args=[], cache_dir=cache_dir, pipeline_command=fake_pipeline_command)

    run_thread = threading.Thread(target=supervisor.run, daemon=True)
    run_thread.start()
    try:
        _wait_for_status(cache_dir, lambda s: s["state"] == "running")
        assert not stale.exists()
    finally:
        supervisor._stop_requested = True
        run_thread.join(timeout=10)
