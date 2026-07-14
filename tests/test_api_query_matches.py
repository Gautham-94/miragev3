"""Tests for GET /api/query-matches (mirage/api/routers/query_matches.py) -- the read
endpoint the Queries page's match list is built on.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.db.database import close_database, init_database
from mirage.db.models import QueryMatch
from mirage.util.time import utc_from_timestamp


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield str(Path(tmp) / "test.db")


def _seed_matches(db_path, *rows):
    database = init_database(db_path)
    for row in rows:
        defaults = dict(query_id="q1", query_text="x", camera="backyard", box=[])
        defaults.update(row)
        QueryMatch.create(**defaults)
    close_database(database)


def test_list_query_matches_empty_by_default(db_path):
    _seed_matches(db_path)
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches")
        assert resp.status_code == 200
        assert resp.json() == []


def test_list_query_matches_returns_seeded_row(db_path):
    _seed_matches(
        db_path,
        dict(
            id="m1", query_id="q1", query_text="a dog off-leash", camera="backyard", object_id="obj1",
            matched_at=utc_from_timestamp(1000.0), score=0.42, box=[1.0, 2.0, 3.0, 4.0],
        ),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "m1"
        assert data[0]["query_text"] == "a dog off-leash"
        assert data[0]["camera"] == "backyard"
        assert data[0]["score"] == 0.42
        assert data[0]["box"] == [1.0, 2.0, 3.0, 4.0]
        assert data[0]["has_thumb"] is False


def test_list_query_matches_filters_by_camera(db_path):
    _seed_matches(
        db_path,
        dict(id="m1", camera="backyard", object_id="o1", matched_at=utc_from_timestamp(1000.0), score=0.5),
        dict(id="m2", camera="garage", object_id="o2", matched_at=utc_from_timestamp(1001.0), score=0.5),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches", params={"camera": "garage"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "m2"


def test_list_query_matches_filters_by_query_id(db_path):
    _seed_matches(
        db_path,
        dict(id="m1", query_id="q1", object_id="o1", matched_at=utc_from_timestamp(1000.0), score=0.5),
        dict(id="m2", query_id="q2", object_id="o2", matched_at=utc_from_timestamp(1001.0), score=0.5),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches", params={"query_id": "q2"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "m2"


def test_list_query_matches_orders_most_recent_first(db_path):
    _seed_matches(
        db_path,
        dict(id="older", object_id="o1", matched_at=utc_from_timestamp(1000.0), score=0.5),
        dict(id="newer", object_id="o2", matched_at=utc_from_timestamp(2000.0), score=0.5),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches")
        data = resp.json()
        assert [row["id"] for row in data] == ["newer", "older"]


def test_get_query_match_by_id(db_path):
    _seed_matches(
        db_path,
        dict(id="m1", object_id="o1", matched_at=utc_from_timestamp(1000.0), score=0.5),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches/m1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "m1"


def test_get_unknown_query_match_returns_404(db_path):
    _seed_matches(db_path)
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches/nonexistent")
        assert resp.status_code == 404


def test_get_thumbnail_for_match_with_no_thumb_returns_404(db_path):
    _seed_matches(
        db_path,
        dict(id="m1", object_id="o1", matched_at=utc_from_timestamp(1000.0), score=0.5, thumb_path=None),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches/m1/thumbnail")
        assert resp.status_code == 404


def test_get_thumbnail_for_unknown_match_returns_404(db_path):
    _seed_matches(db_path)
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches/nonexistent/thumbnail")
        assert resp.status_code == 404


def test_get_thumbnail_for_missing_file_on_disk_returns_410(db_path, tmp_path):
    missing_path = str(tmp_path / "does_not_exist.jpg")
    _seed_matches(
        db_path,
        dict(id="m1", object_id="o1", matched_at=utc_from_timestamp(1000.0), score=0.5, thumb_path=missing_path),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches/m1/thumbnail")
        assert resp.status_code == 410


def test_get_thumbnail_returns_real_jpeg_bytes(db_path, tmp_path):
    thumb_path = tmp_path / "real_thumb.jpg"
    fake_jpeg = b"\xff\xd8\xff\xe0fake jpeg bytes for testing"
    thumb_path.write_bytes(fake_jpeg)

    _seed_matches(
        db_path,
        dict(id="m1", object_id="o1", matched_at=utc_from_timestamp(1000.0), score=0.5, thumb_path=str(thumb_path)),
    )
    app = create_app(db_path=db_path)
    with TestClient(app) as client:
        resp = client.get("/api/query-matches/m1/thumbnail")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == fake_jpeg
