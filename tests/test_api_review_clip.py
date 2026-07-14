"""Tests for GET /api/review/{segment_id}/clip (mirage/api/routers/review.py) -- the
"play the footage that led to this alert" endpoint, stitching overlapping recording
clips together via mirage/recording/stitch.py.
"""

from __future__ import annotations

import subprocess as sp
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.db.database import close_database, init_database
from mirage.db.models import Recordings, ReviewSegment
from mirage.util.time import utc_from_timestamp

TEST_VIDEO = Path(__file__).resolve().parent.parent / "media" / "test_source.mp4"


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "test.db")


def _seed_segment(db_path, seg_id="seg1", camera="backyard", start=1000.0, end=1010.0):
    database = init_database(db_path)
    ReviewSegment.create(
        id=seg_id, camera=camera, start_time=utc_from_timestamp(start),
        end_time=utc_from_timestamp(end) if end is not None else None,
        severity="alert", data={},
    )
    close_database(database)


def test_get_clip_for_unknown_segment_returns_404(db_path, tmp_path):
    _seed_segment(db_path)
    app = create_app(db_path=db_path, export_dir=str(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/review/nonexistent/clip")
        assert resp.status_code == 404


def test_get_clip_for_segment_still_open_returns_409(db_path, tmp_path):
    _seed_segment(db_path, end=None)
    app = create_app(db_path=db_path, export_dir=str(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/review/seg1/clip")
        assert resp.status_code == 409


def test_get_clip_with_no_overlapping_recordings_returns_404(db_path, tmp_path):
    _seed_segment(db_path)
    app = create_app(db_path=db_path, export_dir=str(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/review/seg1/clip")
        assert resp.status_code == 404


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_get_clip_stitches_and_serves_real_video(db_path, tmp_path):
    clip_a = tmp_path / "clip_a.mp4"
    clip_b = tmp_path / "clip_b.mp4"
    sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(clip_a)],
           capture_output=True, timeout=30, check=True)
    sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(clip_b)],
           capture_output=True, timeout=30, check=True)

    database = init_database(db_path)
    ReviewSegment.create(
        id="seg1", camera="backyard", start_time=utc_from_timestamp(1000.0),
        end_time=utc_from_timestamp(1010.0), severity="alert", data={},
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
        resp = client.get("/api/review/seg1/clip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert len(resp.content) > 0

        stitched_path = export_dir / "review_clips" / "seg1.mp4"
        assert stitched_path.exists()

        # second request reuses the cached stitched file rather than re-stitching
        resp2 = client.get("/api/review/seg1/clip")
        assert resp2.status_code == 200
        assert resp2.content == resp.content
