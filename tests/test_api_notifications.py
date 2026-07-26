"""Tests for the notify tailer (mirage/api/routers/notifications.py) -- the background
asyncio task that bridges mirage.notify_bus's file-based notification log into the SSE
fan-out set of connected clients.

Deliberately tests _fetch_and_serialize and the tailer's row-lookup/skip behavior
directly rather than driving a full SSE connection through FastAPI's TestClient --
sync TestClient's streaming support has known rough edges around detecting a client
disconnect for a long-lived generator response (the route's `while True` loop depends
on `await request.is_disconnected()`, which doesn't reliably resolve under the sync test
transport), making an end-to-end streaming test flaky/hang-prone. The route itself was
manually verified against a real running `python -m mirage.api` process (curl against
GET /api/events/stream while appending to notify_log.jsonl) during development.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mirage.api.routers.notifications import _fetch_and_serialize
from mirage.db.database import close_database, init_database
from mirage.db.models import Event
from mirage.util.time import utc_from_timestamp


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


def test_fetch_and_serialize_existing_event(db):
    Event.create(
        id="ev1", label="animal", camera="backyard",
        start_time=utc_from_timestamp(1000.0), end_time=None,
        score=0.9, top_score=0.9, false_positive=False,
    )

    message = _fetch_and_serialize("event", "ev1")

    assert message is not None
    assert message["type"] == "event"
    assert message["data"]["id"] == "ev1"
    assert message["data"]["camera"] == "backyard"


def test_fetch_and_serialize_missing_row_returns_none(db):
    assert _fetch_and_serialize("event", "does-not-exist") is None


def test_fetch_and_serialize_unknown_table_returns_none(db):
    assert _fetch_and_serialize("not_a_real_table", "x") is None
