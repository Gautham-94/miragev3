from __future__ import annotations

import tempfile

from mirage.notify_bus import (
    MAX_LOG_LINES,
    TRIM_CHECK_INTERVAL,
    NotifyEvent,
    NotifyLogWriter,
    read_notify_events_since,
)


def test_append_and_read_since_zero():
    with tempfile.TemporaryDirectory() as cache_dir:
        writer = NotifyLogWriter(cache_dir)
        writer.append(NotifyEvent(table="event", id="e1", op="create"))
        writer.append(NotifyEvent(table="review_segment", id="r1", op="create"))

        entries, offset = read_notify_events_since(cache_dir, 0)
        assert offset == 2
        assert [e["id"] for e in entries] == ["e1", "r1"]
        assert entries[0]["table"] == "event"
        assert entries[0]["op"] == "create"


def test_incremental_read_only_returns_new_entries():
    with tempfile.TemporaryDirectory() as cache_dir:
        writer = NotifyLogWriter(cache_dir)
        writer.append(NotifyEvent(table="event", id="e1", op="create"))

        entries, offset = read_notify_events_since(cache_dir, 0)
        assert len(entries) == 1

        writer.append(NotifyEvent(table="query_match", id="q1", op="create"))
        entries2, offset2 = read_notify_events_since(cache_dir, offset)
        assert [e["id"] for e in entries2] == ["q1"]
        assert offset2 == 2


def test_read_from_nonexistent_file_returns_empty():
    with tempfile.TemporaryDirectory() as cache_dir:
        entries, offset = read_notify_events_since(cache_dir, 0)
        assert entries == []
        assert offset == 0


def test_offset_beyond_file_length_resets_gracefully():
    with tempfile.TemporaryDirectory() as cache_dir:
        writer = NotifyLogWriter(cache_dir)
        writer.append(NotifyEvent(table="event", id="e1", op="create"))

        # since_offset far beyond the file's actual length (e.g. after a trim) --
        # should reset to reading from the top rather than returning nothing/erroring.
        entries, offset = read_notify_events_since(cache_dir, 9999)
        assert len(entries) == 1
        assert offset == 1


def test_trims_past_max_log_lines():
    with tempfile.TemporaryDirectory() as cache_dir:
        writer = NotifyLogWriter(cache_dir)
        total = MAX_LOG_LINES + 250
        for i in range(total):
            writer.append(NotifyEvent(table="event", id=f"e{i}", op="create"))

        # Trimming is only checked every TRIM_CHECK_INTERVAL appends (see
        # NotifyLogWriter.append), so the file can transiently hold up to that many
        # lines past MAX_LOG_LINES between checks -- same bound as ActivityLogWriter.
        lines = writer.path.read_text().splitlines()
        assert len(lines) <= MAX_LOG_LINES + TRIM_CHECK_INTERVAL
