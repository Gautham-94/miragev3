"""RecordingMaintainer: moves validated ffmpeg cache segments to permanent storage and
records their metadata in the database.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 8.1, 8.3.

This module owns the "every ~5 seconds" cache-to-permanent-storage loop. It is deliberately
decoupled from the detection pipeline: it only correlates recording segments against
detection activity by timestamp (section 8.3), reading whatever activity stats have been
published for a camera's time range so far, with no other synchronization required.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Callable, Optional

from mirage.config.schema import CameraConfig
from mirage.const import MAX_SEGMENTS_IN_CACHE
from mirage.db.models import Recordings
from mirage.recording.retention import SegmentActivityStats, should_retain_by_base_policy
from mirage.recording.segments import (
    CacheSegment,
    ffmpeg_open_file_paths,
    is_open_by_ffmpeg,
    is_valid_segment_duration,
    list_cache_segments,
    make_recording_id,
    probe_duration,
    promote_segment,
)
from mirage.util.time import as_naive_utc, utcnow

logger = logging.getLogger(__name__)

# Given a camera name and a [start, end) time range, return aggregate detection-activity
# stats for that window. This is the "section 8.3 timestamp correlation" seam -- in the
# full system this reads from a buffer of published (camera, frame_time, tracked_objects,
# motion_boxes) tuples; kept as an injectable callable here so this module is
# independently testable without the detection pipeline.
ActivityStatsProvider = Callable[[str, datetime.datetime, datetime.datetime], SegmentActivityStats]


def default_activity_stats_provider(camera: str, start: datetime.datetime, end: datetime.datetime) -> SegmentActivityStats:
    return SegmentActivityStats()


class RecordingMaintainer:
    def __init__(
        self,
        cameras: dict[str, CameraConfig],
        cache_dir: str,
        record_dir: str,
        activity_stats_provider: Optional[ActivityStatsProvider] = None,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        self.cameras = cameras
        self.cache_dir = cache_dir
        self.record_dir = record_dir
        self.activity_stats_provider = activity_stats_provider or default_activity_stats_provider
        self.ffmpeg_path = ffmpeg_path

    def run_once(self) -> None:
        segments = list_cache_segments(self.cache_dir)
        self._enforce_cache_backpressure(segments)
        # Computed ONCE per cycle (not once per segment) -- see
        # ffmpeg_open_file_paths's own docstring for why re-scanning the whole host
        # process table per segment was real, measurable duplicated work whenever
        # multiple segments are pending in the same pass (OPTIMIZATION_OPPORTUNITIES.md
        # item 3).
        open_paths = ffmpeg_open_file_paths()
        for segment in segments:
            self._process_segment(segment, open_paths)

    def _enforce_cache_backpressure(self, segments: list[CacheSegment]) -> None:
        """Section 8.1 point 4: if more than MAX_SEGMENTS_IN_CACHE unprocessed segments
        have piled up for one camera, force-delete the oldest excess ones.
        """
        by_camera: dict[str, list[CacheSegment]] = {}
        for s in segments:
            by_camera.setdefault(s.camera, []).append(s)

        for camera, camera_segments in by_camera.items():
            camera_segments.sort(key=lambda s: s.start_time)
            excess = len(camera_segments) - MAX_SEGMENTS_IN_CACHE
            if excess <= 0:
                continue
            for s in camera_segments[:excess]:
                logger.warning("%s: cache backpressure, force-deleting stale segment %s", camera, s.path)
                s.path.unlink(missing_ok=True)
                segments.remove(s)

    def _process_segment(self, segment: CacheSegment, open_paths: set[str]) -> None:
        camera_config = self.cameras.get(segment.camera)
        if camera_config is None or not camera_config.enabled:
            segment.path.unlink(missing_ok=True)
            return

        if is_open_by_ffmpeg(segment.path, open_paths):
            return  # still being written -- leave it for a later pass, don't touch it

        duration = probe_duration(segment.path)
        if not is_valid_segment_duration(duration):
            logger.warning("%s: discarding invalid segment %s (duration=%s)", segment.camera, segment.path, duration)
            segment.path.unlink(missing_ok=True)
            return

        start_time = as_naive_utc(segment.start_time)
        end_time = start_time + datetime.timedelta(seconds=duration)
        stats = self.activity_stats_provider(segment.camera, start_time, end_time)

        now = utcnow()
        if not should_retain_by_base_policy(start_time, now, camera_config.record, stats):
            segment.path.unlink(missing_ok=True)
            return

        dest = promote_segment(segment, self.record_dir, self.ffmpeg_path)
        if dest is None:
            return  # left in cache; will be retried next pass or reaped by backpressure

        Recordings.create(
            id=make_recording_id(segment.start_time),
            camera=segment.camera,
            path=str(dest),
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            motion=stats.motion_count,
            objects=stats.object_count,
            dBFS=stats.dBFS,
            segment_size_mb=dest.stat().st_size / (1024 * 1024),
        )


def run_recording_maintainer_loop(maintainer: RecordingMaintainer, stop_event, interval_seconds: float = 5.0) -> None:
    """Runs maintainer.run_once() repeatedly, pacing so each cycle starts roughly
    `interval_seconds` after the previous one started (accounting for how long the
    previous run_once() itself took), until stop_event is set.
    """
    while True:
        loop_start = time.time()
        try:
            maintainer.run_once()
        except Exception:
            logger.exception("recording maintainer loop iteration failed")
        elapsed = time.time() - loop_start
        wait_time = max(0.0, interval_seconds - elapsed)
        if stop_event.wait(wait_time):
            break
