"""Tests for GET /api/events/{event_id}/clip (mirage/api/routers/events.py) -- lets the
Events page play a clip inline (in a video-lightbox, same as the Review page) instead
of deep-linking to the Recordings page. Stitches overlapping recording clips together
via mirage/recording/stitch.py, same primitive the review-segment clip endpoint uses.
"""

from __future__ import annotations

import subprocess as sp
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.db.database import close_database, init_database
from mirage.db.models import Event, Recordings
from mirage.util.time import utc_from_timestamp

TEST_VIDEO = Path(__file__).resolve().parent.parent / "media" / "test_source.mp4"


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "test.db")


def _seed_event(db_path, event_id="ev1", camera="backyard", start=1000.0, end=1010.0):
    database = init_database(db_path)
    Event.create(
        id=event_id, camera=camera, label="animal",
        start_time=utc_from_timestamp(start),
        end_time=utc_from_timestamp(end) if end is not None else None,
        score=0.9, top_score=0.9, false_positive=False,
    )
    close_database(database)


def test_get_clip_for_unknown_event_returns_404(db_path, tmp_path):
    _seed_event(db_path)
    app = create_app(db_path=db_path, export_dir=str(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/events/nonexistent/clip")
        assert resp.status_code == 404


def test_get_clip_with_no_overlapping_recordings_returns_404(db_path, tmp_path):
    _seed_event(db_path)
    app = create_app(db_path=db_path, export_dir=str(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/events/ev1/clip")
        assert resp.status_code == 404


def test_get_clip_for_event_with_no_end_time_still_looks_up_a_padded_window(db_path, tmp_path):
    # An in-progress event (end_time=None) shouldn't 409/error -- it should search a
    # window padded around start_time only, same as EventsPage's old deep-link padding.
    _seed_event(db_path, end=None)
    app = create_app(db_path=db_path, export_dir=str(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/events/ev1/clip")
        assert resp.status_code == 404  # no recordings seeded -- still a clean 404, not a crash


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_get_clip_stitches_and_serves_real_video(db_path, tmp_path):
    clip_a = tmp_path / "clip_a.mp4"
    clip_b = tmp_path / "clip_b.mp4"
    sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(clip_a)],
           capture_output=True, timeout=30, check=True)
    sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(clip_b)],
           capture_output=True, timeout=30, check=True)

    database = init_database(db_path)
    Event.create(
        id="ev1", camera="backyard", label="animal", start_time=utc_from_timestamp(1000.0),
        end_time=utc_from_timestamp(1010.0), score=0.9, top_score=0.9, false_positive=False,
    )
    Recordings.create(
        id="rec_a", camera="backyard", path=str(clip_a), start_time=utc_from_timestamp(1000.0),
        end_time=utc_from_timestamp(1005.0), duration=5.0, segment_size_mb=1.0,
    )
    Recordings.create(
        id="rec_b", camera="backyard", path=str(clip_b), start_time=utc_from_timestamp(1005.0),
        end_time=utc_from_timestamp(1010.0), duration=5.0, segment_size_mb=1.0,
    )
    close_database(database)

    export_dir = tmp_path / "exports"
    app = create_app(db_path=db_path, export_dir=str(export_dir))
    with TestClient(app) as client:
        resp = client.get("/api/events/ev1/clip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert len(resp.content) > 0
        assert "content-disposition" not in resp.headers  # must stay inline-playable

        stitched_path = export_dir / "event_clips" / "ev1.mp4"
        assert stitched_path.exists()

        # second request reuses the cached stitched file rather than re-stitching
        resp2 = client.get("/api/events/ev1/clip")
        assert resp2.status_code == 200
        assert resp2.content == resp.content
