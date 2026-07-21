"""Tests for mirage/ptz/poller.py (PtzPoller) -- the background thread that lives
inside each PTZ-enabled camera's CameraTracker process, polling ONVIF GetStatus and
exposing a thread-safe is_moving() flag for CameraOrchestrator's ptz_moving_fn.

Uses a fake PtzClient (injected via the client_factory constructor param) rather than
mocking the onvif library -- these tests only need to exercise PtzPoller's own
threading/polling/patrol logic, not real ONVIF network calls (see
mirage/ptz/client.py's own tests for that, and tests/test_api_onvif.py's module
docstring for why this codebase prefers real deterministic failure paths or lightweight
fakes over a mocking framework for ONVIF calls specifically).
"""

from __future__ import annotations

import threading
import time

from mirage.config.schema import PtzConfig, PtzPresetConfig
from mirage.ptz.poller import PtzPoller


class _FakePtzClient:
    """Records calls made to it and returns a scripted sequence of statuses --
    connect()/close() are no-ops (no real network), get_status() pops the next
    scripted PtzStatus, goto_preset() just records the token.
    """

    def __init__(self, statuses):
        from mirage.ptz.client import PtzStatus

        self._statuses = list(statuses) if statuses else [PtzStatus(moving=False)]
        self.connect_calls = 0
        self.goto_preset_calls: list[str] = []
        self.close_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1

    async def get_status(self):
        # Repeat the last status forever once exhausted, so a poller that outlives the
        # scripted sequence doesn't crash mid-test.
        if len(self._statuses) > 1:
            return self._statuses.pop(0)
        return self._statuses[0]

    async def goto_preset(self, token: str) -> None:
        self.goto_preset_calls.append(token)

    async def close(self) -> None:
        self.close_calls += 1


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_is_moving_starts_false_before_any_poll():
    poller = PtzPoller("cam1", PtzConfig(enabled=True))
    assert poller.is_moving() is False


def test_poller_reflects_a_moving_status_from_the_client():
    from mirage.ptz.client import PtzStatus

    fake_client = _FakePtzClient([PtzStatus(moving=True)])
    poller = PtzPoller("cam1", PtzConfig(enabled=True), client_factory=lambda: fake_client)

    try:
        poller.start()
        assert _wait_until(lambda: poller.is_moving() is True), "expected is_moving() to become True"
    finally:
        poller.stop()

    assert fake_client.connect_calls >= 1


def test_poller_reflects_returning_to_idle():
    from mirage.ptz.client import PtzStatus

    fake_client = _FakePtzClient([PtzStatus(moving=True), PtzStatus(moving=False)])
    poller = PtzPoller("cam1", PtzConfig(enabled=True), client_factory=lambda: fake_client)

    try:
        poller.start()
        assert _wait_until(lambda: poller.is_moving() is True)
        assert _wait_until(lambda: poller.is_moving() is False), "expected is_moving() to flip back to False"
    finally:
        poller.stop()


def test_stop_sets_moving_false_and_joins_the_thread():
    from mirage.ptz.client import PtzStatus

    fake_client = _FakePtzClient([PtzStatus(moving=True)])
    poller = PtzPoller("cam1", PtzConfig(enabled=True), client_factory=lambda: fake_client)
    poller.start()
    _wait_until(lambda: poller.is_moving() is True)

    poller.stop()

    assert poller.is_moving() is False
    assert not poller._thread.is_alive()


def test_patrol_advances_through_presets_while_stationary():
    from mirage.ptz.client import PtzStatus

    fake_client = _FakePtzClient([PtzStatus(moving=False)])
    ptz_config = PtzConfig(
        enabled=True, patrol_enabled=True,
        patrol_presets=[PtzPresetConfig(token="1"), PtzPresetConfig(token="2")],
        patrol_interval_seconds=0,  # advance immediately every poll tick, for a fast test
    )
    poller = PtzPoller("cam1", ptz_config, client_factory=lambda: fake_client)

    try:
        poller.start()
        assert _wait_until(lambda: len(fake_client.goto_preset_calls) >= 2, timeout=3.0)
    finally:
        poller.stop()

    assert fake_client.goto_preset_calls[:2] == ["1", "2"]


def test_patrol_disabled_never_calls_goto_preset():
    from mirage.ptz.client import PtzStatus

    fake_client = _FakePtzClient([PtzStatus(moving=False)])
    ptz_config = PtzConfig(
        enabled=True, patrol_enabled=False,
        patrol_presets=[PtzPresetConfig(token="1")],
    )
    poller = PtzPoller("cam1", ptz_config, client_factory=lambda: fake_client)

    try:
        poller.start()
        time.sleep(0.3)
    finally:
        poller.stop()

    assert fake_client.goto_preset_calls == []


def test_connect_failure_is_retried_not_fatal():
    """A client whose connect() always raises must not crash the poller thread --
    is_moving() should just stay False, and the thread should keep retrying (proven
    by confirming the thread is still alive after the failure).
    """

    class _AlwaysFailsClient:
        async def connect(self):
            raise ConnectionError("camera unreachable")

        async def get_status(self):
            raise AssertionError("should never reach get_status if connect() always fails")

        async def close(self):
            pass

    poller = PtzPoller("cam1", PtzConfig(enabled=True), client_factory=lambda: _AlwaysFailsClient())
    import mirage.ptz.poller as poller_module

    original_backoff = poller_module.RECONNECT_BACKOFF_SECONDS
    poller_module.RECONNECT_BACKOFF_SECONDS = 0.05
    try:
        poller.start()
        time.sleep(0.3)
        assert poller.is_moving() is False
        assert poller._thread.is_alive()
    finally:
        poller.stop()
        poller_module.RECONNECT_BACKOFF_SECONDS = original_backoff
