"""Stitches multiple permanent recording segments (mirage.db.models.Recordings rows)
into one continuous video, for playing back "the footage that led to this alert" from a
ReviewSegment -- a review segment's duration (up to 30s+ of continuous activity, per
mirage.events.review.ReviewSegmentMaintainer's cutoff_seconds) almost always spans
MULTIPLE separate fixed-length recording files (10-60s each, per-camera configurable via
RecordConfig.segment_seconds), never exactly one, so there's no single existing file that
already equals a review segment's own time window.

Uses ffmpeg's concat demuxer with -c copy (stream copy, no re-encode) since all of a
single camera's recordings share the same codec/resolution -- fast and lossless, same
"just remux, don't transcode" principle mirage.recording.segments.promote_segment already
applies elsewhere in this codebase.
"""

from __future__ import annotations

import datetime
import logging
import subprocess as sp
import tempfile
from pathlib import Path

from mirage.db.models import Recordings

logger = logging.getLogger(__name__)

STITCH_TIMEOUT_SECONDS = 60


def recordings_overlapping(camera: str, start_time: datetime.datetime, end_time: datetime.datetime) -> list[Recordings]:
    """Every Recordings row for this camera whose own [start_time, end_time] range
    overlaps [start_time, end_time] at all -- not just ones that started inside the
    window, since a recording segment that started slightly BEFORE the review segment
    began can still contain the first moments of it.
    """
    query = (
        Recordings.select()
        .where(
            (Recordings.camera == camera)
            & (Recordings.start_time < end_time)
            & (Recordings.end_time > start_time)
        )
        .order_by(Recordings.start_time.asc())
    )
    return list(query)


def stitch_recordings(recordings: list[Recordings], dest_path: Path, ffmpeg_path: str = "ffmpeg") -> bool:
    """Concatenates `recordings` (already ordered by start_time) into dest_path via
    ffmpeg's concat demuxer. Returns True on success. dest_path's parent directory must
    already exist.
    """
    if not recordings:
        return False

    missing = [r for r in recordings if not Path(r.path).exists()]
    if missing:
        logger.warning("stitch_recordings: %d of %d recording file(s) missing on disk, stitching what remains", len(missing), len(recordings))
    present = [r for r in recordings if Path(r.path).exists()]
    if not present:
        return False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list_path = Path(f.name)
        for rec in present:
            # ffmpeg's concat demuxer list format requires each path single-quoted, with
            # any literal single-quote in the path itself escaped as '\'' -- unlikely in
            # practice for these auto-generated paths, but handled for correctness.
            escaped = str(Path(rec.path).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        result = sp.run(
            [ffmpeg_path, "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list_path),
             "-c", "copy", "-movflags", "+faststart", str(dest_path)],
            capture_output=True, text=True, timeout=STITCH_TIMEOUT_SECONDS,
        )
    except (sp.TimeoutExpired, OSError) as e:
        logger.error("stitch_recordings: ffmpeg failed: %s", e)
        return False
    finally:
        concat_list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error("stitch_recordings: ffmpeg concat failed: %s", result.stderr[-500:])
        return False
    return True
