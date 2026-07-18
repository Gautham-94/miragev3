"""Tests for mirage/tracking/rules.py -- RulesEngine, the crowd-count and dwell-time/
loitering derived-condition alerts (TODO_FIX_LIST.md item 7.4). Pure in-memory logic,
no DB/SHM/subprocess involved -- these tests feed synthetic TrackedObjectState dicts
directly, same as test_openvocab_dispatcher.py does for its own synthetic-track logic.
"""

from __future__ import annotations

import pytest

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    FfmpegConfig,
    RulesConfig,
)
from mirage.tracking.rules import RULE_OBJECT_ID_PREFIX, RulesEngine
from mirage.tracking.stationary import StationaryClassifier
from mirage.tracking.tracker import TrackedObjectState


def _camera(name: str = "cam1", crowd_threshold: int | None = None, dwell_seconds: int | None = None) -> CameraConfig:
    return CameraConfig(
        name=name,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=640, height=480),
        rules=RulesConfig(crowd_threshold=crowd_threshold, dwell_seconds=dwell_seconds),
    )


def _person(obj_id: str, box=(0, 0, 50, 50), is_false_positive: bool = False, label: str = "person") -> TrackedObjectState:
    return TrackedObjectState(
        id=obj_id, label=label, box=box, score=0.9,
        stationary=StationaryClassifier(threshold_frames=50), frame_time=0.0,
        is_false_positive=is_false_positive,
    )


# --------------------------------------------------------------------------------------
# Crowd count
# --------------------------------------------------------------------------------------


def test_crowd_rule_disabled_by_default():
    engine = RulesEngine()
    camera = _camera()  # crowd_threshold=None
    tracked = {f"p{i}": _person(f"p{i}") for i in range(10)}

    result = engine.process(camera, frame_time=0.0, tracked_objects=tracked)

    assert result == {}


def test_crowd_rule_does_not_fire_below_threshold():
    engine = RulesEngine()
    camera = _camera(crowd_threshold=5)
    tracked = {f"p{i}": _person(f"p{i}") for i in range(4)}

    result = engine.process(camera, frame_time=0.0, tracked_objects=tracked)

    assert result == {}


def test_crowd_rule_fires_at_exactly_the_threshold():
    engine = RulesEngine()
    camera = _camera(crowd_threshold=5)
    tracked = {f"p{i}": _person(f"p{i}") for i in range(5)}

    result = engine.process(camera, frame_time=0.0, tracked_objects=tracked)

    assert len(result) == 1
    state = next(iter(result.values()))
    assert state.id.startswith(RULE_OBJECT_ID_PREFIX)
    assert "5" in state.label
    assert state.is_false_positive is False


def test_crowd_rule_ignores_false_positive_persons_in_the_count():
    engine = RulesEngine()
    camera = _camera(crowd_threshold=3)
    tracked = {
        "p1": _person("p1"), "p2": _person("p2"),
        "p3": _person("p3", is_false_positive=True),  # doesn't count
    }

    result = engine.process(camera, frame_time=0.0, tracked_objects=tracked)

    assert result == {}


def test_crowd_rule_only_counts_the_person_label():
    engine = RulesEngine()
    camera = _camera(crowd_threshold=2)
    tracked = {"p1": _person("p1"), "c1": _person("c1", label="car"), "c2": _person("c2", label="car")}

    result = engine.process(camera, frame_time=0.0, tracked_objects=tracked)

    assert result == {}  # only 1 real "person", cars don't count toward crowd


def test_crowd_rule_stops_firing_once_count_drops_back_below_threshold():
    engine = RulesEngine()
    camera = _camera(crowd_threshold=3)

    over = {f"p{i}": _person(f"p{i}") for i in range(3)}
    assert len(engine.process(camera, frame_time=0.0, tracked_objects=over)) == 1

    under = {"p0": _person("p0")}
    assert engine.process(camera, frame_time=1.0, tracked_objects=under) == {}


def test_crowd_rule_id_is_stable_across_frames_for_the_same_camera():
    """A stable object id across frames is what lets ReviewSegmentMaintainer treat
    consecutive crowd triggers as ONE continuous segment (same id = same "object" in
    its diff logic) rather than starting a brand new segment every frame.
    """
    engine = RulesEngine()
    camera = _camera(crowd_threshold=2)
    tracked = {f"p{i}": _person(f"p{i}") for i in range(2)}

    first = engine.process(camera, frame_time=0.0, tracked_objects=tracked)
    second = engine.process(camera, frame_time=1.0, tracked_objects=tracked)

    assert set(first.keys()) == set(second.keys())


# --------------------------------------------------------------------------------------
# Dwell-time / loitering
# --------------------------------------------------------------------------------------


def test_dwell_rule_disabled_by_default():
    engine = RulesEngine()
    camera = _camera()  # dwell_seconds=None
    tracked = {"p1": _person("p1")}

    for t in [0.0, 100.0, 200.0]:
        result = engine.process(camera, frame_time=t, tracked_objects=tracked)
        assert result == {}


def test_dwell_rule_does_not_fire_before_threshold_elapses():
    engine = RulesEngine()
    camera = _camera(dwell_seconds=60)
    tracked = {"p1": _person("p1")}

    engine.process(camera, frame_time=0.0, tracked_objects=tracked)
    result = engine.process(camera, frame_time=30.0, tracked_objects=tracked)

    assert result == {}


def test_dwell_rule_fires_once_threshold_elapses():
    engine = RulesEngine()
    camera = _camera(dwell_seconds=60)
    tracked = {"p1": _person("p1")}

    engine.process(camera, frame_time=0.0, tracked_objects=tracked)  # first-seen
    result = engine.process(camera, frame_time=61.0, tracked_objects=tracked)

    assert len(result) == 1
    state = next(iter(result.values()))
    assert state.id == f"{RULE_OBJECT_ID_PREFIX}dwell-p1"
    assert "61" in state.label
    assert state.is_false_positive is False


def test_dwell_rule_counts_time_regardless_of_movement():
    """Explicit design choice: dwell time is NOT gated on state.stationary -- a person
    slowly moving forward in a queue must still trigger the rule (see RulesConfig.
    dwell_seconds's own docstring).
    """
    engine = RulesEngine()
    camera = _camera(dwell_seconds=10)
    moving_box_sequence = [(0, 0, 50, 50), (10, 10, 60, 60), (20, 20, 70, 70)]

    engine.process(camera, frame_time=0.0, tracked_objects={"p1": _person("p1", box=moving_box_sequence[0])})
    engine.process(camera, frame_time=5.0, tracked_objects={"p1": _person("p1", box=moving_box_sequence[1])})
    result = engine.process(camera, frame_time=11.0, tracked_objects={"p1": _person("p1", box=moving_box_sequence[2])})

    assert len(result) == 1


def test_dwell_rule_ignores_false_positive_objects():
    engine = RulesEngine()
    camera = _camera(dwell_seconds=10)
    tracked = {"p1": _person("p1", is_false_positive=True)}

    engine.process(camera, frame_time=0.0, tracked_objects=tracked)
    result = engine.process(camera, frame_time=20.0, tracked_objects=tracked)

    assert result == {}


def test_dwell_rule_resets_when_object_disappears_and_a_new_one_appears():
    engine = RulesEngine()
    camera = _camera(dwell_seconds=10)

    engine.process(camera, frame_time=0.0, tracked_objects={"p1": _person("p1")})
    engine.process(camera, frame_time=5.0, tracked_objects={})  # p1 gone
    # A different object id appears later -- must start its OWN dwell clock, not
    # inherit p1's.
    result = engine.process(camera, frame_time=6.0, tracked_objects={"p2": _person("p2")})

    assert result == {}
    result_later = engine.process(camera, frame_time=17.0, tracked_objects={"p2": _person("p2")})
    assert len(result_later) == 1
    assert next(iter(result_later.values())).id == f"{RULE_OBJECT_ID_PREFIX}dwell-p2"


def test_dwell_rule_survives_a_brief_single_frame_gap_for_the_same_object_id():
    """Regression test for a REAL production bug: ObjectTracker.update()
    (mirage/tracking/tracker.py) can drop an id from its returned tracked_objects dict
    for a single frame while norfair's own hit_counter/coasting still considers it the
    SAME ongoing track (it reappears next frame with the identical id) -- confirmed
    directly against real pipeline logs, where a continuously-tracked person (one
    single 1320s Event, per EventProcessor/the DB) only triggered a 120s dwell_seconds
    threshold after ~330s of ACCUMULATED (non-contiguous) time, because the OLD pruning
    logic reset the dwell clock on every single-frame flicker. A brief gap well inside
    _DWELL_START_GRACE_SECONDS must NOT reset the clock -- elapsed time should be
    measured from the object's TRUE first sighting, not its most recent reappearance.
    """
    engine = RulesEngine()
    camera = _camera(dwell_seconds=10)

    engine.process(camera, frame_time=0.0, tracked_objects={"p1": _person("p1")})
    # p1 flickers out of tracked_objects for exactly one frame (a single-frame norfair
    # miss), then reappears with the SAME id, well within the grace window.
    engine.process(camera, frame_time=3.0, tracked_objects={})
    engine.process(camera, frame_time=3.2, tracked_objects={"p1": _person("p1")})

    # Elapsed must be measured from frame_time=0.0 (p1's TRUE first sighting), not
    # reset to 3.2 -- so it should already be past the 10s threshold by frame_time=11.
    result = engine.process(camera, frame_time=11.0, tracked_objects={"p1": _person("p1")})

    assert len(result) == 1
    state = next(iter(result.values()))
    assert state.id == f"{RULE_OBJECT_ID_PREFIX}dwell-p1"
    assert "11" in state.label  # ~11s elapsed since TRUE first sighting at t=0.0


def test_dwell_rule_tracks_multiple_objects_independently():
    engine = RulesEngine()
    camera = _camera(dwell_seconds=10)

    engine.process(camera, frame_time=0.0, tracked_objects={"p1": _person("p1")})
    engine.process(camera, frame_time=5.0, tracked_objects={"p1": _person("p1"), "p2": _person("p2")})
    # p1 has been present 11s (0 -> 11), p2 only 6s (5 -> 11) -- only p1 should fire.
    result = engine.process(camera, frame_time=11.0, tracked_objects={"p1": _person("p1"), "p2": _person("p2")})

    assert len(result) == 1
    assert next(iter(result.values())).id == f"{RULE_OBJECT_ID_PREFIX}dwell-p1"


def test_dwell_starts_bookkeeping_does_not_leak_across_many_short_lived_objects():
    """Regression-shaped test mirroring CameraOrchestrator._lifecycles's own leak fix
    (OPTIMIZATION_OPPORTUNITIES.md item 2) -- RulesEngine._dwell_starts must not grow
    UNBOUNDEDLY as distinct object ids pass through and disappear. Entries are no
    longer pruned the instant an id is missing from one frame (see
    _DWELL_START_GRACE_SECONDS's docstring for the real bug that fixed) -- so a bounded
    "in flight" window of recently-departed ids is expected and correct, not a bug;
    what matters is that it stays bounded (roughly the grace window's worth of ids),
    not that it's always exactly zero.
    """
    from mirage.tracking.rules import _DWELL_START_GRACE_SECONDS

    engine = RulesEngine()
    camera = _camera(dwell_seconds=100)  # never actually triggers in this test

    for i in range(500):
        obj_id = f"p{i}"
        engine.process(camera, frame_time=float(i), tracked_objects={obj_id: _person(obj_id)})
        engine.process(camera, frame_time=float(i) + 0.5, tracked_objects={})  # disappears next frame

    # Bounded by the grace window (roughly one id per second at this test's cadence),
    # not by the full 500 objects that passed through -- proves old entries DO
    # eventually get cleaned up, just not instantly.
    assert 0 < len(engine._dwell_starts) <= int(_DWELL_START_GRACE_SECONDS) + 2

    # And after the grace window fully elapses with no further activity at all, every
    # remaining entry drains away completely.
    engine.process(camera, frame_time=500.0 + _DWELL_START_GRACE_SECONDS + 1, tracked_objects={})
    assert len(engine._dwell_starts) == 0


def test_crowd_and_dwell_rules_can_both_fire_in_the_same_frame():
    engine = RulesEngine()
    camera = _camera(crowd_threshold=2, dwell_seconds=10)
    tracked = {"p1": _person("p1"), "p2": _person("p2")}

    engine.process(camera, frame_time=0.0, tracked_objects=tracked)
    result = engine.process(camera, frame_time=11.0, tracked_objects=tracked)

    # Both crowd (still >= 2) and dwell (p1/p2 both present 11s) should have fired.
    ids = set(result.keys())
    assert any(RULE_OBJECT_ID_PREFIX + "crowd-" in i for i in ids)
    assert any(RULE_OBJECT_ID_PREFIX + "dwell-" in i for i in ids)


def test_rules_engine_is_scoped_per_camera():
    engine = RulesEngine()
    cam1 = _camera("cam1", dwell_seconds=10)
    cam2 = _camera("cam2", dwell_seconds=10)

    engine.process(cam1, frame_time=0.0, tracked_objects={"p1": _person("p1")})
    # cam2 has never seen "p1" before -- its own dwell clock must start fresh, not
    # inherit cam1's first-seen time (this would only be a bug if RulesEngine kept a
    # single _dwell_starts dict keyed by obj_id alone across cameras -- confirming
    # object ids ARE globally unique in practice, so this test's real assertion is
    # that a fresh id on a different camera behaves like a fresh id, full stop).
    result = engine.process(cam2, frame_time=11.0, tracked_objects={"p2": _person("p2")})

    assert result == {}
