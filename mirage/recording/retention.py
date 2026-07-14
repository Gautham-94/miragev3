"""Retention policy decisions.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 8.2.

Purely time-based retention -- no disk-usage-percentage triggers. This module is pure
decision logic (given activity stats + config, should a segment be kept?); the actual
DB queries/deletes live in recorder.py so this stays independently unit-testable.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from mirage.config.schema import RecordConfig, RetainMode


@dataclass
class SegmentActivityStats:
    motion_count: int = 0
    object_count: int = 0
    dBFS: int = 0

    @property
    def has_any_activity(self) -> bool:
        return self.motion_count > 0 or self.object_count > 0 or self.dBFS > 0


def continuous_expire_date(now: datetime.datetime, record_config: RecordConfig) -> datetime.datetime:
    return now - datetime.timedelta(days=record_config.continuous.days)


def motion_expire_date(now: datetime.datetime, record_config: RecordConfig) -> datetime.datetime:
    # Motion retention can never be shorter than continuous retention (spec section 8.2).
    days = max(record_config.motion.days, record_config.continuous.days)
    return now - datetime.timedelta(days=days)


def should_retain_by_base_policy(
    segment_start: datetime.datetime,
    now: datetime.datetime,
    record_config: RecordConfig,
    stats: SegmentActivityStats,
) -> bool:
    """The base continuous/motion retention decision, BEFORE considering any overlapping
    review segment (which can extend retention further -- see should_retain_with_review).
    """
    if record_config.continuous.days > 0 and segment_start >= continuous_expire_date(now, record_config):
        return True
    if record_config.motion.days > 0 and stats.has_any_activity and segment_start >= motion_expire_date(now, record_config):
        return True
    return False


def should_retain_with_review_overlap(
    stats: SegmentActivityStats,
    review_retain_mode: RetainMode,
) -> bool:
    """Applied when a segment overlaps a review segment's time range -- the review's own
    retention mode governs, per spec section 8.2.
    """
    if review_retain_mode == RetainMode.all:
        return True
    if review_retain_mode == RetainMode.motion:
        return stats.has_any_activity
    if review_retain_mode == RetainMode.active_objects:
        return stats.object_count > 0
    return False


def overlaps(
    a_start: datetime.datetime, a_end: datetime.datetime,
    b_start: datetime.datetime, b_end: datetime.datetime,
) -> bool:
    return a_start < b_end and b_start < a_end
