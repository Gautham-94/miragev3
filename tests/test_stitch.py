"""Tests for mirage/recording/stitch.py -- stitching multiple recording clips into one
continuous video for a review segment's "play the footage that led to this alert"
feature.
"""

from __future__ import annotations

import datetime
import subprocess as sp
import tempfile
from pathlib import Path

import pytest

from mirage.recording.stitch import recordings_overlapping, stitch_recordings
from mirage.db.database import close_database, init_database
from mirage.db.models import Recordings

TEST_VIDEO = Path(__file__).resolve().parent.parent / "media" / "test_source.mp4"


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


def _mk(id_, camera, start, end):
    return Recordings.create(
        id=id_, camera=camera, path=f"/tmp/{id_}.mp4",
        start_time=start, end_time=end, duration=(end - start).total_seconds(),
        segment_size_mb=1.0,
    )


def test_recordings_overlapping_finds_clips_within_window(db):
    t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
    _mk("r1", "cam1", t0, t0 + datetime.timedelta(seconds=10))
    _mk("r2", "cam1", t0 + datetime.timedelta(seconds=10), t0 + datetime.timedelta(seconds=20))
    _mk("r3", "cam1", t0 + datetime.timedelta(seconds=20), t0 + datetime.timedelta(seconds=30))

    result = recordings_overlapping("cam1", t0 + datetime.timedelta(seconds=5), t0 + datetime.timedelta(seconds=25))
    assert [r.id for r in result] == ["r1", "r2", "r3"]


def test_recordings_overlapping_excludes_clips_entirely_outside_window(db):
    t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
    _mk("before", "cam1", t0 - datetime.timedelta(seconds=20), t0 - datetime.timedelta(seconds=10))
    _mk("during", "cam1", t0, t0 + datetime.timedelta(seconds=10))
    _mk("after", "cam1", t0 + datetime.timedelta(seconds=100), t0 + datetime.timedelta(seconds=110))

    result = recordings_overlapping("cam1", t0, t0 + datetime.timedelta(seconds=10))
    assert [r.id for r in result] == ["during"]


def test_recordings_overlapping_includes_clip_that_started_before_window(db):
    # A recording segment that started slightly before the review window began, but
    # whose end_time is still after the window's start, must be included -- it contains
    # the first moments of the alert.
    t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
    _mk("straddling", "cam1", t0 - datetime.timedelta(seconds=5), t0 + datetime.timedelta(seconds=5))

    result = recordings_overlapping("cam1", t0, t0 + datetime.timedelta(seconds=10))
    assert [r.id for r in result] == ["straddling"]


def test_recordings_overlapping_filters_by_camera(db):
    t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
    _mk("cam1_rec", "cam1", t0, t0 + datetime.timedelta(seconds=10))
    _mk("cam2_rec", "cam2", t0, t0 + datetime.timedelta(seconds=10))

    result = recordings_overlapping("cam1", t0, t0 + datetime.timedelta(seconds=10))
    assert [r.id for r in result] == ["cam1_rec"]


def test_stitch_recordings_returns_false_for_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out.mp4"
        assert stitch_recordings([], dest) is False


def test_stitch_recordings_returns_false_when_all_files_missing(db):
    t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
    rec = _mk("missing", "cam1", t0, t0 + datetime.timedelta(seconds=10))
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out.mp4"
        assert stitch_recordings([rec], dest) is False


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_stitch_recordings_produces_a_real_concatenated_video(db):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clip_a = tmp_path / "clip_a.mp4"
        clip_b = tmp_path / "clip_b.mp4"
        sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(clip_a)],
               capture_output=True, timeout=30, check=True)
        sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(clip_b)],
               capture_output=True, timeout=30, check=True)

        t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
        rec_a = Recordings.create(
            id="a", camera="cam1", path=str(clip_a), start_time=t0,
            end_time=t0 + datetime.timedelta(seconds=5), duration=5.0, segment_size_mb=1.0,
        )
        rec_b = Recordings.create(
            id="b", camera="cam1", path=str(clip_b), start_time=t0 + datetime.timedelta(seconds=5),
            end_time=t0 + datetime.timedelta(seconds=10), duration=5.0, segment_size_mb=1.0,
        )

        dest = tmp_path / "stitched.mp4"
        ok = stitch_recordings([rec_a, rec_b], dest)
        assert ok is True
        assert dest.exists()

        # real proof: the stitched video's duration is roughly the sum of both clips,
        # not just one of them (i.e. concatenation actually happened)
        probe = sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(dest)],
            capture_output=True, text=True, timeout=15, check=True,
        )
        stitched_duration = float(probe.stdout.strip())

        probe_a = sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip_a)],
            capture_output=True, text=True, timeout=15, check=True,
        )
        single_clip_duration = float(probe_a.stdout.strip())

        assert stitched_duration > single_clip_duration * 1.5


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_stitch_recordings_skips_missing_files_and_stitches_the_rest(db):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clip_a = tmp_path / "clip_a.mp4"
        sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(clip_a)],
               capture_output=True, timeout=30, check=True)

        t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
        rec_a = Recordings.create(
            id="a", camera="cam1", path=str(clip_a), start_time=t0,
            end_time=t0 + datetime.timedelta(seconds=5), duration=5.0, segment_size_mb=1.0,
        )
        rec_missing = Recordings.create(
            id="missing", camera="cam1", path=str(tmp_path / "does_not_exist.mp4"),
            start_time=t0 + datetime.timedelta(seconds=5), end_time=t0 + datetime.timedelta(seconds=10),
            duration=5.0, segment_size_mb=1.0,
        )

        dest = tmp_path / "stitched.mp4"
        ok = stitch_recordings([rec_a, rec_missing], dest)
        assert ok is True
        assert dest.exists()
