"""Recording segment validation and promotion from cache to permanent storage.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 8.1.
"""

from __future__ import annotations

import datetime
import logging
import re
import subprocess as sp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

from mirage.const import CACHE_SEGMENT_FORMAT, MAX_SEGMENT_DURATION

logger = logging.getLogger(__name__)

# Matches "<camera>@<strftime CACHE_SEGMENT_FORMAT>.mp4", e.g. "front_door@20260706153000+0000.mp4".
_SEGMENT_FILENAME_RE = re.compile(r"^(?P<camera>.+)@(?P<timestamp>\d{14}[+-]\d{4})\.mp4$")


@dataclass
class CacheSegment:
    path: Path
    camera: str
    start_time: datetime.datetime


def parse_cache_segment_filename(path: Path) -> Optional[CacheSegment]:
    """Parses a raw ffmpeg segment-muxer output filename back into (camera, start_time).
    Returns None if the filename doesn't match the expected pattern (e.g. a stray file).
    """
    m = _SEGMENT_FILENAME_RE.match(path.name)
    if not m:
        return None
    try:
        start_time = datetime.datetime.strptime(m.group("timestamp"), CACHE_SEGMENT_FORMAT)
    except ValueError:
        return None
    return CacheSegment(path=path, camera=m.group("camera"), start_time=start_time)


def list_cache_segments(cache_dir: str) -> list[CacheSegment]:
    segments = []
    for path in sorted(Path(cache_dir).glob("*.mp4")):
        if path.name.startswith("preview_"):
            continue
        parsed = parse_cache_segment_filename(path)
        if parsed is not None:
            segments.append(parsed)
    return segments


def ffmpeg_open_file_paths() -> set[str]:
    """Every file path currently open by any ffmpeg process on the host, in ONE pass
    over the process table -- the expensive part of the "is this cache segment still
    being written" check (see is_open_by_ffmpeg's docstring for why the check itself is
    needed at all). Split out from is_open_by_ffmpeg so a caller checking MULTIPLE
    segments in the same maintainer cycle (RecordingMaintainer.run_once, up to
    MAX_SEGMENTS_IN_CACHE=6 segments PER CAMERA can be pending at once) only pays the
    psutil.process_iter() + proc.open_files() syscall cost ONCE per cycle, not once per
    segment -- confirmed as real, measurable duplicated work: the old per-call version
    re-enumerated the ENTIRE host process table from scratch for every single pending
    segment within the same 5-second maintainer pass (OPTIMIZATION_OPPORTUNITIES.md
    item 3). No cheaper OS-level primitive gives the same guarantee without a false
    negative -- confirmed directly: ffmpeg does not take an flock/fcntl lock on its
    output file, so a lock-based "is this open" check would silently never detect an
    in-progress segment, reintroducing the exact bug this whole check exists to
    prevent (deleting a still-being-written file). Plumbing ffmpeg's own PID through
    from CameraCapture (a separate process from where this runs) would be more
    invasive for the same net effect.
    """
    open_paths: set[str] = set()
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] not in ("ffmpeg", "ffmpeg.exe"):
                continue
            for f in proc.open_files():
                open_paths.add(f.path)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return open_paths


def is_open_by_ffmpeg(path: Path, open_paths: set[str] | None = None) -> bool:
    """Section 8.1 point 3: skip files still being actively written by ffmpeg, rather
    than treating them as corrupt/invalid just because they don't have a moov atom yet
    (a segment-muxer output file is only finalized -- gets its moov atom written -- once
    ffmpeg rolls over to the NEXT segment or exits cleanly; probing it before that point
    always looks "invalid" even though it's simply still in progress). This is a real,
    load-bearing check: without it, an in-flight segment gets deleted by the recording
    maintainer's very first pass over it, before ffmpeg ever gets a chance to finish
    writing it -- confirmed during development (see IMPLEMENTATION_NOTES.md) by tracing
    real segment files through several maintainer poll cycles: files that were
    "invalid" (no moov atom) at t=10s became fully valid (complete, probeable duration)
    only ~5-10s later, once ffmpeg's next segment rollover flushed them.

    `open_paths`: pass the result of ffmpeg_open_file_paths() to check against an
    already-computed snapshot (the normal usage, once per maintainer cycle -- see that
    function's own docstring) instead of re-scanning the whole process table for this
    one path. Left optional (computed fresh if omitted) so this function stays usable
    on its own, e.g. in tests or a one-off call.
    """
    resolved = str(path.resolve())
    if open_paths is None:
        open_paths = ffmpeg_open_file_paths()
    return resolved in open_paths


def probe_duration(path: Path, ffprobe_path: str = "ffprobe") -> Optional[float]:
    """Runs ffprobe to get the segment's duration in seconds. Returns None if the file is
    not a valid video (corrupt/incomplete segment, still being written, etc).
    """
    try:
        result = sp.run(
            [ffprobe_path, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
    except (sp.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def is_valid_segment_duration(duration: Optional[float]) -> bool:
    return duration is not None and 0 < duration <= MAX_SEGMENT_DURATION


def promote_segment(segment: CacheSegment, record_dir: str, ffmpeg_path: str = "ffmpeg") -> Optional[Path]:
    """Remuxes a validated cache segment into permanent storage with +faststart, per spec
    section 8.1. Destination path pattern: RECORD_DIR/<YYYY-MM-DD>/<HH>/<camera>/<MM.SS>.mp4.
    Returns the destination path on success, None on failure (cache file is left in place
    on failure so a subsequent pass can retry or the backpressure/cleanup logic can reap it).
    """
    dest_dir = (
        Path(record_dir)
        / segment.start_time.strftime("%Y-%m-%d")
        / segment.start_time.strftime("%H")
        / segment.camera
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{segment.start_time.strftime('%M.%S')}.mp4"

    try:
        result = sp.run(
            [ffmpeg_path, "-hide_banner", "-y", "-i", str(segment.path),
             "-c", "copy", "-movflags", "+faststart", str(dest_path)],
            capture_output=True, text=True, timeout=60,
        )
    except (sp.TimeoutExpired, OSError) as e:
        logger.error("failed to promote segment %s: %s", segment.path, e)
        return None

    if result.returncode != 0:
        logger.error("ffmpeg remux failed for %s: %s", segment.path, result.stderr[-500:])
        return None

    segment.path.unlink(missing_ok=True)
    return dest_path


def make_recording_id(start_time: datetime.datetime) -> str:
    import random
    import string

    rand6 = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{start_time.timestamp()}-{rand6}"
