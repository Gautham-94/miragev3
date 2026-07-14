from __future__ import annotations

import datetime
import subprocess as sp
import tempfile
import time
from pathlib import Path

import pytest

from mirage.recording.segments import (
    is_open_by_ffmpeg,
    is_valid_segment_duration,
    list_cache_segments,
    make_recording_id,
    parse_cache_segment_filename,
    probe_duration,
    promote_segment,
)

TEST_VIDEO = Path(__file__).resolve().parent.parent / "media" / "test_source.mp4"


def test_parse_cache_segment_filename_valid():
    parsed = parse_cache_segment_filename(Path("/tmp/cache/front_door@20260706153000+0000.mp4"))
    assert parsed is not None
    assert parsed.camera == "front_door"
    assert parsed.start_time == datetime.datetime(2026, 7, 6, 15, 30, 0, tzinfo=datetime.timezone.utc)


def test_parse_cache_segment_filename_camera_name_with_at_symbols():
    # Camera names shouldn't contain "@", but the regex is greedy on the camera group so a
    # pathological name still parses using the LAST "@" as the separator.
    parsed = parse_cache_segment_filename(Path("weird@cam@20260706153000+0000.mp4"))
    assert parsed is not None
    assert parsed.camera == "weird@cam"


def test_parse_cache_segment_filename_rejects_non_matching():
    assert parse_cache_segment_filename(Path("/tmp/cache/not_a_segment.mp4")) is None
    assert parse_cache_segment_filename(Path("/tmp/cache/preview_cam1-123.mp4")) is None


def test_list_cache_segments_skips_preview_and_invalid_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "cam1@20260706153000+0000.mp4").touch()
        (tmp_path / "cam1@20260706153010+0000.mp4").touch()
        (tmp_path / "preview_cam1-123.mp4").touch()
        (tmp_path / "not_a_segment.mp4").touch()

        segments = list_cache_segments(tmp)
        assert len(segments) == 2
        assert all(s.camera == "cam1" for s in segments)


def test_is_valid_segment_duration():
    assert is_valid_segment_duration(10.0) is True
    assert is_valid_segment_duration(0.001) is True
    assert is_valid_segment_duration(0.0) is False
    assert is_valid_segment_duration(-1.0) is False
    assert is_valid_segment_duration(601.0) is False
    assert is_valid_segment_duration(None) is False


def test_make_recording_id_format():
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    rid = make_recording_id(now)
    ts_part, rand_part = rid.split("-")
    assert float(ts_part) == now.timestamp()
    assert len(rand_part) == 6


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_probe_duration_on_real_video():
    duration = probe_duration(TEST_VIDEO)
    assert duration is not None
    assert 4.5 < duration < 5.5  # test_source.mp4 was generated as a 5s clip


def test_probe_duration_on_missing_file_returns_none():
    assert probe_duration(Path("/nonexistent/path/does-not-exist.mp4")) is None


def test_probe_duration_on_corrupt_file_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        corrupt = Path(tmp) / "corrupt.mp4"
        corrupt.write_bytes(b"not a real mp4 file")
        assert probe_duration(corrupt) is None


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_promote_segment_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        start_time = datetime.datetime(2026, 7, 6, 15, 30, 0, tzinfo=datetime.timezone.utc)
        cache_path = cache_dir / "front_door@20260706153000+0000.mp4"

        # Produce a real, valid mp4 segment via ffmpeg (not just a copy) to exercise the
        # actual remux path meaningfully.
        sp.run(
            ["ffmpeg", "-hide_banner", "-y", "-i", str(TEST_VIDEO), "-c", "copy", str(cache_path)],
            capture_output=True, timeout=30, check=True,
        )
        assert cache_path.exists()

        parsed = parse_cache_segment_filename(cache_path)
        assert parsed is not None
        parsed_with_real_start = parsed.__class__(path=cache_path, camera="front_door", start_time=start_time)

        dest = promote_segment(parsed_with_real_start, str(record_dir))

        assert dest is not None
        assert dest.exists()
        assert dest == record_dir / "2026-07-06" / "15" / "front_door" / "30.00.mp4"
        assert not cache_path.exists(), "cache file should be removed after successful promotion"

        # The promoted file should itself be a valid, probeable video of the same duration.
        duration = probe_duration(dest)
        assert duration is not None
        assert 4.5 < duration < 5.5


def test_promote_segment_failure_leaves_cache_file_in_place():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        record_dir = Path(tmp) / "recordings"
        cache_dir.mkdir()

        cache_path = cache_dir / "front_door@20260706153000+0000.mp4"
        cache_path.write_bytes(b"not a real mp4 file")

        parsed = parse_cache_segment_filename(cache_path)
        assert parsed is not None

        dest = promote_segment(parsed, str(record_dir))

        assert dest is None
        assert cache_path.exists(), "cache file must NOT be deleted when promotion fails"


def test_is_open_by_ffmpeg_false_for_file_no_process_has_open():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "not_open_by_anything.mp4"
        path.write_bytes(b"x")
        assert is_open_by_ffmpeg(path) is False


@pytest.mark.skipif(not TEST_VIDEO.exists(), reason="test_source.mp4 fixture not generated")
def test_is_open_by_ffmpeg_true_while_ffmpeg_holds_the_file_open():
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "held_open.mp4"
        # Start a real, slow-running ffmpeg process that keeps this exact output file
        # open the whole time (looping the source indefinitely so the process itself
        # never exits during this test's short window) -- mirrors exactly the real-world
        # scenario is_open_by_ffmpeg exists to detect (see IMPLEMENTATION_NOTES.md: a
        # cache segment being actively written must not be treated as "invalid" and
        # deleted just because it doesn't have a moov atom yet). Uses a plain single mp4
        # output (not the segment muxer, which requires a %d/strftime numbering
        # placeholder in its output template and rejects a literal path -- confirmed
        # directly: "Invalid segment filename template") since this test only needs
        # "a real ffmpeg process holding a real file open," not segmenting behavior.
        proc = sp.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-re", "-stream_loop", "-1",
             "-i", str(TEST_VIDEO), "-c", "copy", "-f", "mp4",
             "-movflags", "frag_keyframe+empty_moov", str(out_path)],
            stdout=sp.DEVNULL, stderr=sp.DEVNULL,
        )
        try:
            deadline = time.time() + 10
            while time.time() < deadline and not out_path.exists():
                time.sleep(0.2)
            assert out_path.exists(), "ffmpeg never created the output file"

            # Give ffmpeg a moment to actually open (not just create) the file.
            time.sleep(1)
            assert is_open_by_ffmpeg(out_path) is True
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except sp.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

        # After ffmpeg has fully exited, the same path must no longer be reported open.
        assert is_open_by_ffmpeg(out_path) is False
