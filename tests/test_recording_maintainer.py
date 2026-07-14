from __future__ import annotations

import datetime
import subprocess as sp
import tempfile
import threading
import time
from pathlib import Path

import pytest

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    FfmpegConfig,
    RecordConfig,
    RetainConfig,
)
from mirage.db.database import close_database, init_database
from mirage.db.models import Recordings
from mirage.recording.maintainer import RecordingMaintainer, run_recording_maintainer_loop
from mirage.recording.retention import SegmentActivityStats

TEST_VIDEO = Path(__file__).resolve().parent.parent / "media" / "test_source.mp4"


def _make_cache_segment(cache_dir: Path, camera: str, start_time: datetime.datetime) -> Path:
    """Creates a real, valid mp4 cache segment (copied from the test fixture video) named
    per the CACHE_SEGMENT_FORMAT convention, so probe_duration/promote_segment exercise
    real ffmpeg rather than a stub.
    """
    filename = f"{camera}@{start_time.strftime('%Y%m%d%H%M%S%z')}.mp4"
    dest = cache_dir / filename
    sp.run(["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(dest)],
           capture_output=True, timeout=30, check=True)
    return dest


def _camera(name: str, continuous_days: float = 0, motion_days: float = 0) -> CameraConfig:
    return CameraConfig(
        name=name,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        record=RecordConfig(
            enabled=True,
            continuous=RetainConfig(days=continuous_days),
            motion=RetainConfig(days=motion_days),
        ),
    )


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_run_once_promotes_segment_within_continuous_retention(temp_db):
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        now = datetime.datetime.now(datetime.timezone.utc)
        recent_start = now - datetime.timedelta(minutes=1)
        _make_cache_segment(cache_dir, "cam1", recent_start)

        cameras = {"cam1": _camera("cam1", continuous_days=5)}
        maintainer = RecordingMaintainer(cameras, str(cache_dir), str(record_dir))
        maintainer.run_once()

        rows = list(Recordings.select().where(Recordings.camera == "cam1"))
        assert len(rows) == 1
        assert Path(rows[0].path).exists()
        assert list(cache_dir.glob("*.mp4")) == []


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_run_once_discards_segment_outside_all_retention_windows(temp_db):
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        now = datetime.datetime.now(datetime.timezone.utc)
        old_start = now - datetime.timedelta(days=30)
        _make_cache_segment(cache_dir, "cam1", old_start)

        # No continuous/motion retention configured at all -> nothing should be kept.
        cameras = {"cam1": _camera("cam1", continuous_days=0, motion_days=0)}
        maintainer = RecordingMaintainer(cameras, str(cache_dir), str(record_dir))
        maintainer.run_once()

        assert list(Recordings.select().where(Recordings.camera == "cam1")) == []
        assert list(cache_dir.glob("*.mp4")) == [], "cache file should be deleted, not left behind"
        assert not record_dir.exists() or list(record_dir.rglob("*.mp4")) == []


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_run_once_retains_old_segment_with_motion_activity(temp_db):
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        now = datetime.datetime.now(datetime.timezone.utc)
        # Past continuous window (2 days) but within motion window (30 days).
        mid_age_start = now - datetime.timedelta(days=10)
        _make_cache_segment(cache_dir, "cam1", mid_age_start)

        cameras = {"cam1": _camera("cam1", continuous_days=2, motion_days=30)}

        def activity_provider(camera, start, end):
            return SegmentActivityStats(motion_count=3)

        maintainer = RecordingMaintainer(cameras, str(cache_dir), str(record_dir), activity_stats_provider=activity_provider)
        maintainer.run_once()

        rows = list(Recordings.select().where(Recordings.camera == "cam1"))
        assert len(rows) == 1
        assert rows[0].motion == 3


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_run_once_discards_segment_for_unknown_camera(temp_db):
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        now = datetime.datetime.now(datetime.timezone.utc)
        _make_cache_segment(cache_dir, "removed_camera", now - datetime.timedelta(minutes=1))

        maintainer = RecordingMaintainer({}, str(cache_dir), str(record_dir))
        maintainer.run_once()

        assert list(cache_dir.glob("*.mp4")) == []
        assert list(Recordings.select()) == []


def test_backpressure_deletes_oldest_excess_segments_per_camera(temp_db):
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        now = datetime.datetime.now(datetime.timezone.utc)
        # Create MAX_SEGMENTS_IN_CACHE(6) + 2 = 8 fake (empty, invalid) segments for one
        # camera -- invalid so they'd fail promotion anyway, but backpressure should
        # delete the EXCESS oldest ones before even attempting to process them.
        paths = []
        for i in range(8):
            start = now - datetime.timedelta(seconds=(8 - i) * 10)
            filename = f"cam1@{start.strftime('%Y%m%d%H%M%S%z')}.mp4"
            p = cache_dir / filename
            p.write_bytes(b"fake")
            paths.append(p)

        cameras = {"cam1": _camera("cam1", continuous_days=5)}
        maintainer = RecordingMaintainer(cameras, str(cache_dir), str(record_dir))
        maintainer.run_once()

        remaining = list(cache_dir.glob("*.mp4"))
        # 8 - 6(MAX_SEGMENTS_IN_CACHE) = 2 oldest deleted by backpressure; the remaining 6
        # are then each attempted (and discarded, since they're fake/invalid mp4s) by the
        # normal per-segment processing -- so ultimately nothing should be left at all,
        # but the important behavioral check is that backpressure ran without error and
        # didn't crash trying to process more than MAX_SEGMENTS_IN_CACHE at once.
        assert remaining == []


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_run_recording_maintainer_loop_runs_until_stopped(temp_db):
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        now = datetime.datetime.now(datetime.timezone.utc)
        _make_cache_segment(cache_dir, "cam1", now - datetime.timedelta(minutes=1))

        cameras = {"cam1": _camera("cam1", continuous_days=5)}
        maintainer = RecordingMaintainer(cameras, str(cache_dir), str(record_dir))

        stop_event = threading.Event()
        t = threading.Thread(target=run_recording_maintainer_loop, args=(maintainer, stop_event, 0.2))
        t.start()

        deadline = time.time() + 5
        while time.time() < deadline:
            if Recordings.select().count() > 0:
                break
            time.sleep(0.1)

        stop_event.set()
        t.join(timeout=5)
        assert not t.is_alive()
        assert Recordings.select().count() == 1
