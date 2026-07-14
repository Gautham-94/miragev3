"""Unit tests for the shared thumbnail-capture helper (mirage/util/thumbnail.py), used by
both EventProcessor and ReviewSegmentMaintainer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mirage.util.thumbnail import capture_thumbnail


def test_capture_thumbnail_writes_file_and_returns_path():
    with tempfile.TemporaryDirectory() as thumb_dir:
        result = capture_thumbnail(lambda cam: b"\xff\xd8jpeg", thumb_dir, "seg1", "cam1")

        assert result == str(Path(thumb_dir) / "seg1.jpg")
        assert Path(result).read_bytes() == b"\xff\xd8jpeg"


def test_capture_thumbnail_no_fetcher_returns_none():
    with tempfile.TemporaryDirectory() as thumb_dir:
        assert capture_thumbnail(None, thumb_dir, "seg1", "cam1") is None


def test_capture_thumbnail_fetcher_returns_empty_bytes_returns_none():
    with tempfile.TemporaryDirectory() as thumb_dir:
        assert capture_thumbnail(lambda cam: b"", thumb_dir, "seg1", "cam1") is None


def test_capture_thumbnail_fetcher_raises_returns_none_not_propagated():
    def failing(cam):
        raise ConnectionError("go2rtc unreachable")

    with tempfile.TemporaryDirectory() as thumb_dir:
        assert capture_thumbnail(failing, thumb_dir, "seg1", "cam1") is None


def test_capture_thumbnail_retries_once_on_transient_empty_response():
    # Regression test: go2rtc's frame.jpeg endpoint can return an empty body during a
    # brief transient window (observed for real during end-to-end validation -- see
    # IMPLEMENTATION_NOTES.md). A single retry should recover a thumbnail that only
    # fails on the very first attempt.
    calls = []

    def flaky(cam):
        calls.append(cam)
        if len(calls) == 1:
            return b""  # transient empty response, matching go2rtc's real behavior
        return b"\xff\xd8jpeg"

    with tempfile.TemporaryDirectory() as thumb_dir:
        result = capture_thumbnail(flaky, thumb_dir, "seg1", "cam1")

        assert len(calls) == 2
        assert result == str(Path(thumb_dir) / "seg1.jpg")
        assert Path(result).read_bytes() == b"\xff\xd8jpeg"


def test_capture_thumbnail_gives_up_after_two_failed_attempts():
    calls = []

    def always_empty(cam):
        calls.append(cam)
        return b""

    with tempfile.TemporaryDirectory() as thumb_dir:
        result = capture_thumbnail(always_empty, thumb_dir, "seg1", "cam1")

        assert len(calls) == 2
        assert result is None
