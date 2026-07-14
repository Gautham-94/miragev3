"""Tests for the API reading config LIVE from the database (create_app(config=None), the
real production mode) rather than a fixed config object passed in at construction time --
confirms /api/cameras reflects whatever is currently saved via MirageConfig.save_to_db(),
including changes made after the app started (e.g. by the config-write endpoints /
add-camera wizard).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    FfmpegConfig,
    MirageConfig,
)
from mirage.db.database import close_database, init_database


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "test.db")


def test_api_reads_default_config_from_fresh_db(db_path):
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/cameras")
        assert resp.status_code == 200
        assert resp.json() == []  # default config: no cameras yet


def test_api_reflects_a_camera_saved_to_db_before_app_started(db_path):
    database = init_database(db_path)
    config = MirageConfig.from_db()
    config.cameras["front_door"] = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1/front_door")]),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
    )
    config.save_to_db()
    close_database(database)

    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/cameras")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.json()]
        assert names == ["front_door"]


def test_api_reflects_a_camera_saved_to_db_while_app_is_running(db_path):
    # This is the real scenario the wizard depends on: the config-write endpoint (a
    # separate request, potentially even a separate process in real deployment) saves a
    # new camera to the DB, and a LATER request to this same running API instance sees
    # it -- proving config isn't cached/frozen at app startup.
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/cameras")
        assert resp.json() == []

        database = init_database(db_path)
        config = MirageConfig.from_db()
        config.cameras["backyard"] = CameraConfig(
            name="backyard",
            ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1/backyard")]),
            detect=DetectConfig(width=640, height=480, fps=5),
            detector="general",
        )
        config.save_to_db()
        close_database(database)

        resp2 = client.get("/api/cameras")
        assert resp2.status_code == 200
        names = [c["name"] for c in resp2.json()]
        assert names == ["backyard"]
