from __future__ import annotations

from mirage.config.schema import ObjectFilterConfig
from mirage.tracking.lifecycle import ObjectLifecycle, is_object_filtered, should_save_snapshot

FRAME_SHAPE = (480, 640)


def test_new_object_starts_as_false_positive():
    obj = ObjectLifecycle(label="person")
    assert obj.is_false_positive is True


def test_computed_score_is_median_of_history():
    obj = ObjectLifecycle(label="person")
    for s in [0.5, 0.6, 0.7]:
        obj.record_score(s)
    assert obj.computed_score() == 0.6


def test_computed_score_even_length_averages_middle_two():
    obj = ObjectLifecycle(label="person")
    for s in [0.4, 0.6]:
        obj.record_score(s)
    assert obj.computed_score() == 0.5


def test_record_score_none_drags_median_down():
    obj = ObjectLifecycle(label="person")
    for s in [0.9, 0.9, 0.9]:
        obj.record_score(s)
    assert obj.computed_score() == 0.9
    obj.record_score(None)  # not detected this frame -> appends 0.0
    obj.record_score(None)
    # scores now: [0.9, 0.9, 0.9, 0.0, 0.0] -> median = 0.9 still (sorted: 0,0,.9,.9,.9)
    assert obj.computed_score() == 0.9
    for _ in range(3):
        obj.record_score(None)
    # scores now (maxlen=10): [0.9,0.9,0.9,0,0,0,0,0] -> median pulled down
    assert obj.computed_score() < 0.9


def test_false_positive_becomes_true_positive_once_threshold_crossed():
    obj = ObjectLifecycle(label="person")
    filter_config = ObjectFilterConfig(threshold=0.7)
    for s in [0.9] * 6:
        obj.record_score(s)
        obj.update_false_positive_status(filter_config)
    assert obj.is_false_positive is False


def test_true_positive_is_sticky_even_if_score_drops():
    obj = ObjectLifecycle(label="person")
    filter_config = ObjectFilterConfig(threshold=0.7)
    for s in [0.9] * 6:
        obj.record_score(s)
        obj.update_false_positive_status(filter_config)
    assert obj.is_false_positive is False

    # Score crashes afterward -- must NOT revert to false positive.
    for s in [0.1] * 10:
        obj.record_score(s)
        obj.update_false_positive_status(filter_config)
    assert obj.is_false_positive is False


def test_stays_false_positive_below_threshold():
    obj = ObjectLifecycle(label="person")
    filter_config = ObjectFilterConfig(threshold=0.7)
    for s in [0.55] * 10:
        obj.record_score(s)
        obj.update_false_positive_status(filter_config)
    assert obj.is_false_positive is True


def test_is_object_filtered_min_score():
    cfg = ObjectFilterConfig(min_score=0.5)
    assert is_object_filtered("person", 0.4, (0, 0, 100, 100), FRAME_SHAPE, cfg) is True
    assert is_object_filtered("person", 0.6, (0, 0, 100, 100), FRAME_SHAPE, cfg) is False


def test_is_object_filtered_area_bounds_absolute():
    cfg = ObjectFilterConfig(min_score=0.0, min_area=1000, max_area=50000)
    assert is_object_filtered("person", 0.9, (0, 0, 10, 10), FRAME_SHAPE, cfg) is True  # area=100, too small
    assert is_object_filtered("person", 0.9, (0, 0, 100, 100), FRAME_SHAPE, cfg) is False  # area=10000, ok
    assert is_object_filtered("person", 0.9, (0, 0, 1000, 1000), FRAME_SHAPE, cfg) is True  # area=1e6, too big


def test_is_object_filtered_area_bounds_fractional():
    # min_area/max_area given as fractions of frame area (0,1) per spec section 6.2.
    frame_area = FRAME_SHAPE[0] * FRAME_SHAPE[1]
    cfg = ObjectFilterConfig(min_score=0.0, min_area=0.01, max_area=0.5)
    tiny_box_area = frame_area * 0.001
    side = int(tiny_box_area ** 0.5)
    assert is_object_filtered("person", 0.9, (0, 0, side, side), FRAME_SHAPE, cfg) is True


def test_is_object_filtered_ratio_bounds():
    cfg = ObjectFilterConfig(min_score=0.0, min_ratio=0.5, max_ratio=2.0)
    assert is_object_filtered("person", 0.9, (0, 0, 10, 100), FRAME_SHAPE, cfg) is True  # ratio=0.1, too narrow
    assert is_object_filtered("person", 0.9, (0, 0, 50, 50), FRAME_SHAPE, cfg) is False  # ratio=1.0, ok
    assert is_object_filtered("person", 0.9, (0, 0, 300, 10), FRAME_SHAPE, cfg) is True  # ratio=30, too wide


def test_should_save_snapshot_false_when_false_positive():
    obj = ObjectLifecycle(label="person")
    obj.position_changes = 5
    assert should_save_snapshot(obj, snapshots_enabled=True) is False


def test_should_save_snapshot_false_when_disabled():
    obj = ObjectLifecycle(label="person")
    obj._false_positive = False
    obj.position_changes = 5
    assert should_save_snapshot(obj, snapshots_enabled=False) is False


def test_should_save_snapshot_false_when_never_moved():
    obj = ObjectLifecycle(label="person")
    obj._false_positive = False
    obj.position_changes = 0
    assert should_save_snapshot(obj, snapshots_enabled=True) is False


def test_should_save_snapshot_true_when_all_conditions_met():
    obj = ObjectLifecycle(label="person")
    obj._false_positive = False
    obj.position_changes = 3
    assert should_save_snapshot(obj, snapshots_enabled=True) is True
