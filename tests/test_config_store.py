"""Tests for the DB-backed config store (mirage/config/store.py) and
MirageConfig.from_db()/save_to_db()/default() -- the persistence layer the add-camera
wizard writes through, replacing hand-edited YAML as the source of truth.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    MirageConfig,
    ModelConfig,
)
from mirage.db.database import close_database, init_database
from mirage.db.models import AppConfig


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


def test_default_config_has_general_detector_and_no_cameras():
    config = MirageConfig.default()

    assert "general" in config.detectors
    assert config.detectors["general"].device == "onnx_yolov8"
    assert config.cameras == {}


def test_from_db_seeds_default_on_fresh_database(db):
    assert AppConfig.select().count() == 0

    config = MirageConfig.from_db()

    assert "general" in config.detectors
    assert config.cameras == {}
    # The seed should have actually been persisted, not just returned in-memory.
    assert AppConfig.select().count() == 1


def test_from_db_returns_same_singleton_row_on_repeated_calls(db):
    first = MirageConfig.from_db()
    second = MirageConfig.from_db()

    assert first.detectors.keys() == second.detectors.keys()
    assert AppConfig.select().count() == 1


def test_save_to_db_then_from_db_round_trips_a_camera(db):
    config = MirageConfig.from_db()
    config.cameras["front_door"] = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1/front_door")]),
        detect=DetectConfig(width=640, height=480, fps=5),
        detector="general",
    )

    config.save_to_db()
    reloaded = MirageConfig.from_db()

    assert "front_door" in reloaded.cameras
    assert reloaded.cameras["front_door"].ffmpeg.inputs[0].path == "rtsp://127.0.0.1/front_door"


def test_save_to_db_overwrites_previous_save_not_duplicates_row(db):
    config = MirageConfig.from_db()
    config.save_to_db()
    config.save_to_db()
    config.save_to_db()

    assert AppConfig.select().count() == 1


def test_save_to_db_persists_a_new_detector(db):
    config = MirageConfig.from_db()
    config.detectors["animals"] = DetectorInstanceConfig(
        name="animals",
        device="onnx_yolov8",
        model=ModelConfig(model_path="models/animals.onnx", labelmap_path="models/animals.txt"),
    )

    config.save_to_db()
    reloaded = MirageConfig.from_db()

    assert "animals" in reloaded.detectors
    assert "general" in reloaded.detectors  # existing detector wasn't clobbered
