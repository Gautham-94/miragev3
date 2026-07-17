"""Tests for /api/system/* (mirage/api/routers/system.py) -- the "Apply changes" button's
backend: reads mirage.supervisor's status file, and writes the restart-request sentinel
file mirage.supervisor polls for. No real supervisor process runs in these tests -- they
only verify the file-based protocol each side speaks (write status.json here, read it
there; write restart_requested here, supervisor tests separately verify it acts on it).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.supervisor import RESTART_REQUEST_FILENAME, STATUS_FILENAME


@pytest.fixture
def client_and_cache_dir():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        cache_dir = str(Path(tmp) / "cache")
        app = create_app(db_path=db_path, cache_dir=cache_dir)
        with TestClient(app) as c:
            yield c, cache_dir


def test_status_is_unknown_when_no_status_file_exists(client_and_cache_dir):
    client, _ = client_and_cache_dir
    resp = client.get("/api/system/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "unknown"
    assert body["pid"] is None


def test_status_reflects_a_real_status_file(client_and_cache_dir):
    client, cache_dir = client_and_cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / STATUS_FILENAME).write_text(
        json.dumps({"state": "running", "pid": 12345, "updated_at": 1700000000.0})
    )

    resp = client.get("/api/system/status")
    body = resp.json()
    assert body["state"] == "running"
    assert body["pid"] == 12345
    assert body["updated_at"] == 1700000000.0


def test_status_survives_a_corrupt_status_file(client_and_cache_dir):
    client, cache_dir = client_and_cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / STATUS_FILENAME).write_text("{not valid json")

    resp = client.get("/api/system/status")
    assert resp.status_code == 200
    assert resp.json()["state"] == "unknown"


def test_restart_request_writes_the_sentinel_file(client_and_cache_dir):
    client, cache_dir = client_and_cache_dir
    sentinel = Path(cache_dir) / RESTART_REQUEST_FILENAME
    assert not sentinel.exists()

    resp = client.post("/api/system/restart")

    assert resp.status_code == 202
    assert resp.json() == {"ok": True}
    assert sentinel.exists()


def test_restart_request_creates_cache_dir_if_missing(client_and_cache_dir):
    client, cache_dir = client_and_cache_dir
    assert not Path(cache_dir).exists()

    resp = client.post("/api/system/restart")

    assert resp.status_code == 202
    assert (Path(cache_dir) / RESTART_REQUEST_FILENAME).exists()


def test_restart_request_is_idempotent_if_pressed_twice_quickly(client_and_cache_dir):
    client, cache_dir = client_and_cache_dir

    first = client.post("/api/system/restart")
    time.sleep(0.01)
    second = client.post("/api/system/restart")

    assert first.status_code == 202
    assert second.status_code == 202
    assert (Path(cache_dir) / RESTART_REQUEST_FILENAME).exists()
