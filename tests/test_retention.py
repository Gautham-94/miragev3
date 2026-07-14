from __future__ import annotations

import datetime

from mirage.config.schema import RecordConfig, RetainConfig, RetainMode
from mirage.recording.retention import (
    SegmentActivityStats,
    continuous_expire_date,
    motion_expire_date,
    overlaps,
    should_retain_by_base_policy,
    should_retain_with_review_overlap,
)

NOW = datetime.datetime(2026, 7, 6, 12, 0, 0, tzinfo=datetime.timezone.utc)


def test_continuous_expire_date():
    cfg = RecordConfig(continuous=RetainConfig(days=5))
    assert continuous_expire_date(NOW, cfg) == NOW - datetime.timedelta(days=5)


def test_motion_expire_date_never_shorter_than_continuous():
    cfg = RecordConfig(continuous=RetainConfig(days=10), motion=RetainConfig(days=3))
    # motion.days=3 < continuous.days=10 -> motion expire date must use 10, not 3.
    assert motion_expire_date(NOW, cfg) == NOW - datetime.timedelta(days=10)


def test_motion_expire_date_uses_motion_when_longer():
    cfg = RecordConfig(continuous=RetainConfig(days=2), motion=RetainConfig(days=30))
    assert motion_expire_date(NOW, cfg) == NOW - datetime.timedelta(days=30)


def test_should_retain_continuous_within_window():
    cfg = RecordConfig(continuous=RetainConfig(days=5))
    recent = NOW - datetime.timedelta(days=1)
    assert should_retain_by_base_policy(recent, NOW, cfg, SegmentActivityStats()) is True


def test_should_retain_continuous_outside_window_no_activity():
    cfg = RecordConfig(continuous=RetainConfig(days=5))
    old = NOW - datetime.timedelta(days=10)
    assert should_retain_by_base_policy(old, NOW, cfg, SegmentActivityStats()) is False


def test_should_retain_motion_outside_continuous_but_within_motion_window():
    cfg = RecordConfig(continuous=RetainConfig(days=2), motion=RetainConfig(days=30))
    mid_age = NOW - datetime.timedelta(days=10)  # past continuous, within motion window
    stats_with_motion = SegmentActivityStats(motion_count=5)
    stats_without_motion = SegmentActivityStats()

    assert should_retain_by_base_policy(mid_age, NOW, cfg, stats_with_motion) is True
    assert should_retain_by_base_policy(mid_age, NOW, cfg, stats_without_motion) is False


def test_should_retain_no_config_at_all():
    cfg = RecordConfig()  # continuous.days=0, motion.days=0 by default
    assert should_retain_by_base_policy(NOW, NOW, cfg, SegmentActivityStats(motion_count=99)) is False


def test_review_overlap_mode_all_always_keeps():
    assert should_retain_with_review_overlap(SegmentActivityStats(), RetainMode.all) is True


def test_review_overlap_mode_motion_requires_activity():
    assert should_retain_with_review_overlap(SegmentActivityStats(motion_count=1), RetainMode.motion) is True
    assert should_retain_with_review_overlap(SegmentActivityStats(), RetainMode.motion) is False
    assert should_retain_with_review_overlap(SegmentActivityStats(dBFS=10), RetainMode.motion) is True


def test_review_overlap_mode_active_objects_requires_object_count():
    assert should_retain_with_review_overlap(SegmentActivityStats(object_count=1), RetainMode.active_objects) is True
    assert should_retain_with_review_overlap(SegmentActivityStats(motion_count=5), RetainMode.active_objects) is False


def test_overlaps_true_cases():
    a_start = NOW
    a_end = NOW + datetime.timedelta(seconds=10)
    # Review segment fully contains the recording segment.
    assert overlaps(a_start, a_end, a_start - datetime.timedelta(seconds=5), a_end + datetime.timedelta(seconds=5))
    # Partial overlap at the start.
    assert overlaps(a_start, a_end, a_start - datetime.timedelta(seconds=5), a_start + datetime.timedelta(seconds=1))
    # Partial overlap at the end.
    assert overlaps(a_start, a_end, a_end - datetime.timedelta(seconds=1), a_end + datetime.timedelta(seconds=5))


def test_overlaps_false_when_disjoint():
    a_start = NOW
    a_end = NOW + datetime.timedelta(seconds=10)
    b_start = a_end + datetime.timedelta(seconds=1)
    b_end = b_start + datetime.timedelta(seconds=10)
    assert overlaps(a_start, a_end, b_start, b_end) is False


def test_overlaps_false_when_exactly_adjacent():
    a_start = NOW
    a_end = NOW + datetime.timedelta(seconds=10)
    # b starts exactly when a ends -- half-open interval semantics, not an overlap.
    assert overlaps(a_start, a_end, a_end, a_end + datetime.timedelta(seconds=10)) is False
