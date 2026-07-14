"""Integration tests for the FastAPI read API (mirage/api). Uses FastAPI's TestClient
against a real temporary SQLite DB with real rows inserted via the peewee models
directly -- no mocking of the DB layer.
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
    DetectorInstanceConfig,
    FfmpegConfig,
    MirageConfig,
    ModelConfig,
    ObjectsConfig,
    RecordConfig,
)
from mirage.db.database import close_database, init_database
from mirage.db.models import Event, Recordings, ReviewSegment
from mirage.util.time import utc_from_timestamp


def _config() -> MirageConfig:
    camera = CameraConfig(
        name="front_door",
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://127.0.0.1/front_door")]),
        detect=DetectConfig(width=640, height=480, fps=5),
        objects=ObjectsConfig(track=["person", "car"]),
        record=RecordConfig(enabled=True),
        detector="general",
    )
    detector = DetectorInstanceConfig(name="general", model=ModelConfig(model_path="x.onnx", labelmap_path="x.txt"), device="onnx_yolov8")
    return MirageConfig(detectors={"general": detector}, cameras={"front_door": camera})


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        app = create_app(_config(), db_path=db_path)
        with TestClient(app) as c:
            yield c


def test_health():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_config(), db_path=str(Path(tmp) / "test.db"))
        with TestClient(app) as c:
            resp = c.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_cameras_returns_configured_camera(client):
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "front_door"
    assert data[0]["width"] == 640
    assert data[0]["height"] == 480
    assert data[0]["track_objects"] == ["person", "car"]
    assert data[0]["record_enabled"] is True


def test_get_camera_by_name(client):
    resp = client.get("/api/cameras/front_door")
    assert resp.status_code == 200
    assert resp.json()["name"] == "front_door"


def test_get_camera_unknown_returns_404(client):
    resp = client.get("/api/cameras/nonexistent")
    assert resp.status_code == 404


def test_list_events_excludes_false_positive_by_default(client):
    Event.create(
        id="ev1", label="person", camera="front_door",
        start_time=utc_from_timestamp(1000.0), score=0.9, top_score=0.9,
        false_positive=False,
    )
    Event.create(
        id="ev2", label="person", camera="front_door",
        start_time=utc_from_timestamp(1001.0), score=0.4, top_score=0.4,
        false_positive=True,
    )

    resp = client.get("/api/events")
    assert resp.status_code == 200
    ids = {e["id"] for e in resp.json()}
    assert ids == {"ev1"}

    resp_all = client.get("/api/events", params={"include_false_positive": True})
    ids_all = {e["id"] for e in resp_all.json()}
    assert ids_all == {"ev1", "ev2"}


def test_list_events_filters_by_camera_and_label(client):
    Event.create(id="ev1", label="person", camera="front_door", start_time=utc_from_timestamp(1000.0), score=0.9, top_score=0.9, false_positive=False)
    Event.create(id="ev2", label="car", camera="front_door", start_time=utc_from_timestamp(1001.0), score=0.9, top_score=0.9, false_positive=False)
    Event.create(id="ev3", label="person", camera="backyard", start_time=utc_from_timestamp(1002.0), score=0.9, top_score=0.9, false_positive=False)

    resp = client.get("/api/events", params={"camera": "front_door", "label": "person"})
    ids = {e["id"] for e in resp.json()}
    assert ids == {"ev1"}


def test_list_events_filters_by_time_range(client):
    Event.create(id="ev1", label="person", camera="front_door", start_time=utc_from_timestamp(1000.0), score=0.9, top_score=0.9, false_positive=False)
    Event.create(id="ev2", label="person", camera="front_door", start_time=utc_from_timestamp(2000.0), score=0.9, top_score=0.9, false_positive=False)

    resp = client.get("/api/events", params={"after": 1500.0})
    ids = {e["id"] for e in resp.json()}
    assert ids == {"ev2"}


def test_get_event_not_found(client):
    resp = client.get("/api/events/nonexistent")
    assert resp.status_code == 404


def test_get_event_roundtrip(client):
    Event.create(
        id="ev1", label="person", camera="front_door",
        start_time=utc_from_timestamp(1000.0), end_time=utc_from_timestamp(1010.0),
        score=0.9, top_score=0.95, false_positive=False, zones=["yard"],
        has_clip=True, has_snapshot=True,
    )
    resp = client.get("/api/events/ev1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["start_time"] == 1000.0
    assert body["end_time"] == 1010.0
    assert body["zones"] == ["yard"]
    assert body["has_clip"] is True


def test_get_event_snapshot_no_snapshot_returns_404(client):
    Event.create(
        id="ev1", label="person", camera="front_door",
        start_time=utc_from_timestamp(1000.0), score=0.9, top_score=0.9, false_positive=False,
    )
    resp = client.get("/api/events/ev1/snapshot")
    assert resp.status_code == 404


def test_get_event_snapshot_missing_file_returns_410(client):
    Event.create(
        id="ev1", label="person", camera="front_door",
        start_time=utc_from_timestamp(1000.0), score=0.9, top_score=0.9, false_positive=False,
        snapshot_path="/tmp/does_not_exist_snapshot.jpg",
    )
    resp = client.get("/api/events/ev1/snapshot")
    assert resp.status_code == 410


def test_get_event_snapshot_serves_real_file(client):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8fakejpegbytes")
        real_path = f.name
    try:
        Event.create(
            id="ev1", label="person", camera="front_door",
            start_time=utc_from_timestamp(1000.0), score=0.9, top_score=0.9, false_positive=False,
            snapshot_path=real_path,
        )
        resp = client.get("/api/events/ev1/snapshot")
        assert resp.status_code == 200
        assert resp.content == b"\xff\xd8fakejpegbytes"
        assert resp.headers["content-type"] == "image/jpeg"
    finally:
        Path(real_path).unlink(missing_ok=True)


def test_list_recordings(client):
    Recordings.create(
        id="rec1", camera="front_door", path="/tmp/rec1.mp4",
        start_time=utc_from_timestamp(1000.0), end_time=utc_from_timestamp(1010.0),
        duration=10.0, segment_size_mb=1.5,
    )
    resp = client.get("/api/recordings")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "rec1"
    assert data[0]["duration"] == 10.0


def test_get_recording_clip_missing_file_returns_410(client):
    Recordings.create(
        id="rec1", camera="front_door", path="/tmp/does_not_exist_rec1.mp4",
        start_time=utc_from_timestamp(1000.0), end_time=utc_from_timestamp(1010.0),
        duration=10.0, segment_size_mb=1.5,
    )
    resp = client.get("/api/recordings/rec1/clip")
    assert resp.status_code == 410


def test_get_recording_clip_serves_real_file(client):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"fake mp4 bytes")
        real_path = f.name
    try:
        Recordings.create(
            id="rec1", camera="front_door", path=real_path,
            start_time=utc_from_timestamp(1000.0), end_time=utc_from_timestamp(1010.0),
            duration=10.0, segment_size_mb=1.5,
        )
        resp = client.get("/api/recordings/rec1/clip")
        assert resp.status_code == 200
        assert resp.content == b"fake mp4 bytes"
        assert resp.headers["content-type"] == "video/mp4"
    finally:
        Path(real_path).unlink(missing_ok=True)


def test_list_review_segments_filters_by_severity(client):
    ReviewSegment.create(id="rs1", camera="front_door", start_time=utc_from_timestamp(1000.0), severity="alert", data={})
    ReviewSegment.create(id="rs2", camera="front_door", start_time=utc_from_timestamp(1001.0), severity="detection", data={})

    resp = client.get("/api/review", params={"severity": "alert"})
    ids = {s["id"] for s in resp.json()}
    assert ids == {"rs1"}


def test_get_review_thumbnail_no_thumb_returns_404(client):
    ReviewSegment.create(id="rs1", camera="front_door", start_time=utc_from_timestamp(1000.0), severity="alert", data={}, thumb_path=None)
    resp = client.get("/api/review/rs1/thumbnail")
    assert resp.status_code == 404
