"""Tests for mirage/openvocab/dispatcher.py -- the main-process piece that gates,
crops, and dispatches confirmed tracked objects to the (real, separate) OpenVocabProcess,
and turns returned matches into QueryMatch DB rows. Uses a real SHM ring segment (same
SharedMemoryFrameManager machinery the real pipeline uses) and real gating, but a fake
in-process queue standing in for the real OpenVocabProcess -- that process itself needs
the real multi-hundred-MB OWLv2 model, out of scope for a fast unit test.
"""

from __future__ import annotations

import queue
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mirage.config.schema import (
    CameraConfig,
    CameraInputConfig,
    DetectConfig,
    DetectorInstanceConfig,
    FfmpegConfig,
    MirageConfig,
    ModelConfig,
    OpenVocabQuery,
)
from mirage.db.database import close_database, init_database
from mirage.db.models import QueryMatch
from mirage.openvocab.dispatcher import OpenVocabDispatcher
from mirage.openvocab.process import OpenVocabMatch, OpenVocabRequest, OpenVocabResult
from mirage.tracking.stationary import StationaryClassifier
from mirage.tracking.tracker import TrackedObjectState
from mirage.util.shm import SharedMemoryFrameManager, frame_name, yuv_frame_size


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


def _camera(name: str = "cam1", width: int = 64, height: int = 48, openvocab_direct_frame: bool = False) -> CameraConfig:
    return CameraConfig(
        name=name,
        ffmpeg=FfmpegConfig(inputs=[CameraInputConfig(path="rtsp://x/y")]),
        detect=DetectConfig(width=width, height=height),
        detector="default",
        openvocab_direct_frame=openvocab_direct_frame,
    )


def _detectors() -> dict:
    return {"default": DetectorInstanceConfig(name="default", model=ModelConfig())}


def _state(obj_id: str, box=(10, 10, 40, 40), is_false_positive: bool = False, frame_time: float = 0.0) -> TrackedObjectState:
    return TrackedObjectState(
        id=obj_id, label="person", box=box, score=0.9,
        stationary=StationaryClassifier(threshold_frames=50), frame_time=frame_time,
        is_false_positive=is_false_positive,
    )


def _write_real_frame(fm: SharedMemoryFrameManager, name: str, width: int, height: int) -> None:
    size = yuv_frame_size(width, height)
    fm.create(name, size)
    buf = fm.write(name, size)
    # A real, non-degenerate YUV420 buffer -- mid-gray luma, neutral chroma -- so
    # crop_yuv_region's color conversion has real pixel data to work with rather than
    # all-zero (which some cv2 color conversions can special-case oddly).
    payload = np.full(size, 128, dtype=np.uint8)
    buf[:] = payload.tobytes()
    fm.close(name)


class FakeQueue:
    """Minimal stand-in for multiprocessing.Queue's put/get/block=False contract, used
    for both request_queue and result_queue in these tests -- avoids spinning up a real
    OS process just to test the dispatcher's own gating/crop/write-back logic.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def put(self, item, block=True):
        self._q.put(item)

    def get(self, block=True, timeout=None):
        if not block:
            return self._q.get_nowait()
        return self._q.get(timeout=timeout)


@pytest.fixture
def frame_manager():
    fm = SharedMemoryFrameManager()
    yield fm
    fm.cleanup()


@pytest.fixture
def thumb_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_camera_with_no_matching_queries_skips_shm_read_entirely(db, frame_manager):
    # No query at all -- process_frame must not even attempt to read the (nonexistent)
    # frame_name, since that would otherwise raise/return None and could look like a
    # different kind of failure.
    camera = _camera()
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    tracked = {"obj1": _state("obj1")}
    dispatcher.process_frame(config, camera, "nonexistent_frame_name", 100.0, tracked)

    assert request_q._q.empty()


def test_unconfirmed_object_is_not_dispatched(db, frame_manager):
    camera = _camera()
    query = OpenVocabQuery(id="q1", text="a red backpack")
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    name = frame_name(camera.name, 0)
    _write_real_frame(frame_manager, name, camera.detect.width, camera.detect.height)

    tracked = {"obj1": _state("obj1", is_false_positive=True)}  # NOT yet confirmed
    dispatcher.process_frame(config, camera, name, 100.0, tracked)

    assert request_q._q.empty()
    frame_manager.delete(name)


def test_confirmed_object_with_matching_query_is_dispatched_with_correct_crop_offset(db, frame_manager):
    camera = _camera(width=64, height=48)
    query = OpenVocabQuery(id="q1", text="a red backpack", cameras=[camera.name])
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    name = frame_name(camera.name, 0)
    _write_real_frame(frame_manager, name, camera.detect.width, camera.detect.height)

    box = (10.0, 5.0, 40.0, 30.0)  # x1, y1, x2, y2
    tracked = {"obj1": _state("obj1", box=box, is_false_positive=False)}
    dispatcher.process_frame(config, camera, name, 100.0, tracked)

    request: OpenVocabRequest = request_q._q.get_nowait()
    assert request.camera_name == camera.name
    assert request.object_id == "obj1"
    assert request.queries == [("q1", "a red backpack")]
    # frame_box is (y1, x1, y2, x2) -- offset used to translate the OWLv2 result's
    # crop-local box back into full-frame coordinates
    assert request.frame_box == (5.0, 10.0, 30.0, 40.0)
    assert len(request.crop_jpeg) > 0  # a real, non-empty JPEG was encoded

    frame_manager.delete(name)


def test_direct_frame_mode_ignores_tracked_objects_and_gates_on_motion(db, frame_manager):
    # Direct-frame mode must dispatch even with ZERO tracked objects, as long as motion
    # is present -- this is the whole point of the mode (checking things the closed-vocab
    # detector never confirms as an object at all).
    camera = _camera(width=64, height=48, openvocab_direct_frame=True)
    query = OpenVocabQuery(id="q1", text="a specific unusual item", cameras=[camera.name])
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    name = frame_name(camera.name, 0)
    _write_real_frame(frame_manager, name, camera.detect.width, camera.detect.height)

    dispatcher.process_frame(config, camera, name, 100.0, tracked_objects={}, motion_boxes=[(0, 0, 10, 10)])

    request: OpenVocabRequest = request_q._q.get_nowait()
    assert request.camera_name == camera.name
    assert request.object_id.startswith("frame-")
    assert request.queries == [("q1", "a specific unusual item")]
    # frame_box spans the whole frame, not a cropped object box
    assert request.frame_box == (0.0, 0.0, float(camera.detect.height), float(camera.detect.width))
    assert len(request.crop_jpeg) > 0

    frame_manager.delete(name)


def test_direct_frame_mode_without_motion_does_not_dispatch(db, frame_manager):
    camera = _camera(width=64, height=48, openvocab_direct_frame=True)
    query = OpenVocabQuery(id="q1", text="a specific unusual item", cameras=[camera.name])
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    name = frame_name(camera.name, 0)
    _write_real_frame(frame_manager, name, camera.detect.width, camera.detect.height)

    dispatcher.process_frame(config, camera, name, 100.0, tracked_objects={}, motion_boxes=[])

    assert request_q._q.empty()
    frame_manager.delete(name)


def test_confirmed_object_mode_ignores_motion_boxes_param(db, frame_manager):
    # A camera NOT in direct-frame mode must still gate on confirmed tracked objects,
    # never on motion_boxes alone -- passing motion_boxes must not change this camera's
    # existing (default) behavior.
    camera = _camera(width=64, height=48, openvocab_direct_frame=False)
    query = OpenVocabQuery(id="q1", text="a red backpack", cameras=[camera.name])
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    name = frame_name(camera.name, 0)
    _write_real_frame(frame_manager, name, camera.detect.width, camera.detect.height)

    # motion present, but no tracked/confirmed objects at all -- must NOT dispatch
    dispatcher.process_frame(config, camera, name, 100.0, tracked_objects={}, motion_boxes=[(0, 0, 10, 10)])

    assert request_q._q.empty()
    frame_manager.delete(name)


def test_direct_frame_mode_gate_is_keyed_per_camera_not_per_object(db, frame_manager):
    # Two consecutive direct-frame dispatches for the SAME (unchanged) camera scene
    # should be deduped by the gate exactly like the confirmed-object mode dedupes per
    # object id -- here keyed by camera name since there's no object id.
    camera = _camera(width=64, height=48, openvocab_direct_frame=True)
    query = OpenVocabQuery(id="q1", text="a specific unusual item", cameras=[camera.name])
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    name = frame_name(camera.name, 0)
    _write_real_frame(frame_manager, name, camera.detect.width, camera.detect.height)

    dispatcher.process_frame(config, camera, name, 100.0, tracked_objects={}, motion_boxes=[(0, 0, 10, 10)])
    assert request_q._q.qsize() == 1

    # immediately again (same unchanged frame, still "in flight" since no result drained yet)
    dispatcher.process_frame(config, camera, name, 100.1, tracked_objects={}, motion_boxes=[(0, 0, 10, 10)])
    assert request_q._q.qsize() == 1  # still just the one -- gated, not a second dispatch

    frame_manager.delete(name)


def test_query_scoped_to_a_different_camera_does_not_apply(db, frame_manager):
    camera = _camera()
    other_camera = _camera(name="cam2")
    query = OpenVocabQuery(id="q1", text="a red backpack", cameras=[other_camera.name])
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera, other_camera.name: other_camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    tracked = {"obj1": _state("obj1", is_false_positive=False)}
    dispatcher.process_frame(config, camera, "irrelevant", 100.0, tracked)

    assert request_q._q.empty()


def test_disabled_query_does_not_apply(db, frame_manager):
    camera = _camera()
    query = OpenVocabQuery(id="q1", text="a red backpack", enabled=False)
    config = MirageConfig(detectors=_detectors(), cameras={camera.name: camera}, queries=[query])
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    tracked = {"obj1": _state("obj1", is_false_positive=False)}
    dispatcher.process_frame(config, camera, "irrelevant", 100.0, tracked)

    assert request_q._q.empty()


def test_draining_a_result_with_matches_creates_query_match_rows(db, frame_manager, thumb_dir):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager, thumb_dir=thumb_dir)

    # simulate a request having been dispatched, so the result can be matched back to it
    dispatcher._pending["req1"] = ("cam1", "obj1", 100.0)

    fake_jpeg = b"\xff\xd8\xff\xe0fake jpeg bytes for testing"
    result = OpenVocabResult(
        request_id="req1", camera_name="cam1", object_id="obj1",
        matches=[OpenVocabMatch(query_id="q1", query_text="a red backpack", score=0.42, box=(5.0, 10.0, 30.0, 40.0))],
        crop_jpeg=fake_jpeg,
    )
    result_q.put(result)

    dispatcher._drain_results()

    rows = list(QueryMatch.select())
    assert len(rows) == 1
    row = rows[0]
    assert row.query_id == "q1"
    assert row.query_text == "a red backpack"
    assert row.camera == "cam1"
    assert row.object_id == "obj1"

    # the exact crop OWLv2 checked was written to disk as this match's thumbnail
    assert row.thumb_path is not None
    assert Path(row.thumb_path).exists()
    assert Path(row.thumb_path).read_bytes() == fake_jpeg


def test_draining_a_result_with_no_crop_leaves_thumb_path_unset(db, frame_manager, thumb_dir):
    # Defensive case: an exception-path OpenVocabResult (see process.py's
    # openvocab_process_main) has crop_jpeg=b"" -- must not crash trying to write an
    # empty thumbnail, and thumb_path should stay None.
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager, thumb_dir=thumb_dir)
    dispatcher._pending["req1"] = ("cam1", "obj1", 100.0)

    result = OpenVocabResult(
        request_id="req1", camera_name="cam1", object_id="obj1",
        matches=[OpenVocabMatch(query_id="q1", query_text="a red backpack", score=0.42, box=(5.0, 10.0, 30.0, 40.0))],
        crop_jpeg=b"",
    )
    result_q.put(result)
    dispatcher._drain_results()

    row = QueryMatch.get()
    assert row.thumb_path is None
    assert abs(row.score - 0.42) < 1e-6
    assert row.box == [5.0, 10.0, 30.0, 40.0]


def test_draining_a_result_with_no_matches_creates_no_rows(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "obj1", 100.0)

    result_q.put(OpenVocabResult(request_id="req1", camera_name="cam1", object_id="obj1", matches=[]))
    dispatcher._drain_results()

    assert list(QueryMatch.select()) == []


def test_drained_result_clears_in_flight_gate_state(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher.gate._in_flight.add("obj1")

    result_q.put(OpenVocabResult(request_id="req1", camera_name="cam1", object_id="obj1", matches=[]))
    dispatcher._drain_results()

    assert "obj1" not in dispatcher.gate._in_flight


# --------------------------------------------------------------------------------------
# Synthetic-track bridge: a real OWLv2 match feeds the SAME dict[str, TrackedObjectState]
# shape EventProcessor/ReviewSegmentMaintainer already consume, so a match for an
# auto-provisioned (or manual) query drives a real Event/ReviewSegment through the exact
# same code path a closed-vocab detection does -- see SyntheticTrack's docstring in
# mirage/openvocab/dispatcher.py.
# --------------------------------------------------------------------------------------


def test_real_match_produces_a_synthetic_tracked_object(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)

    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.42, box=(5.0, 10.0, 30.0, 40.0))],
        )
    )
    dispatcher._drain_results()

    live = dispatcher.synthetic_tracked_objects("cam1", now=100.5)
    assert len(live) == 1
    state = next(iter(live.values()))
    assert state.label == "animals"
    assert abs(state.score - 0.42) < 1e-6
    # OpenVocabMatch.box is (y1, x1, y2, x2); TrackedObjectState.box is (x1, y1, x2, y2)
    assert state.box == (10.0, 5.0, 40.0, 30.0)
    assert state.is_false_positive is False
    assert not state.stationary.is_stationary()


def test_repeated_matches_for_same_query_keep_the_same_synthetic_object_id(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
        )
    )
    dispatcher._drain_results()
    first_id = next(iter(dispatcher.synthetic_tracked_objects("cam1", now=100.0)))

    dispatcher._pending["req2"] = ("cam1", "frame-105.0", 105.0)
    result_q.put(
        OpenVocabResult(
            request_id="req2", camera_name="cam1", object_id="frame-105.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.5, box=(0.0, 0.0, 12.0, 12.0))],
        )
    )
    dispatcher._drain_results()
    second_id = next(iter(dispatcher.synthetic_tracked_objects("cam1", now=105.0)))

    assert first_id == second_id


def test_synthetic_track_expires_after_ttl(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
        )
    )
    dispatcher._drain_results()

    # Still alive well within the TTL window.
    assert len(dispatcher.synthetic_tracked_objects("cam1", now=110.0)) == 1
    # Long past SYNTHETIC_TRACK_TTL_SECONDS (20s) since the last match at frame_time=100.
    assert dispatcher.synthetic_tracked_objects("cam1", now=200.0) == {}
    # Expired entry is actually dropped from internal state, not just filtered on read.
    assert dispatcher._synthetic_tracks.get("cam1") in (None, {})


def test_different_queries_on_same_camera_get_independent_synthetic_objects(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[
                OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0)),
                OpenVocabMatch(query_id="q2", query_text="a car", score=0.6, box=(1.0, 1.0, 11.0, 11.0)),
            ],
        )
    )
    dispatcher._drain_results()

    live = dispatcher.synthetic_tracked_objects("cam1", now=100.0)
    assert len(live) == 2
    assert {state.label for state in live.values()} == {"animals", "a car"}


def test_two_non_overlapping_matches_for_the_same_query_become_two_synthetic_objects(db, frame_manager):
    """The core multi-instance fix: a single OWLv2 call can genuinely return multiple
    simultaneous matches for the same query (e.g. two separate animals both in frame at
    once -- confirmed against real garage footage). These must NOT collapse into one
    synthetic track that just gets overwritten by whichever match arrives -- each
    spatially distinct box needs its own object_id, so each drives its own Event.
    """
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[
                # far-apart boxes (in the (y1,x1,y2,x2) OpenVocabMatch convention) --
                # essentially zero IoU with each other
                OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0)),
                OpenVocabMatch(query_id="q1", query_text="animals", score=0.4, box=(200.0, 200.0, 210.0, 210.0)),
            ],
        )
    )
    dispatcher._drain_results()

    live = dispatcher.synthetic_tracked_objects("cam1", now=100.0)
    assert len(live) == 2
    assert all(state.label == "animals" for state in live.values())
    boxes = {state.box for state in live.values()}
    assert len(boxes) == 2  # two genuinely distinct positions, not the same box twice


def test_overlapping_match_for_same_query_updates_existing_track_not_a_new_one(db, frame_manager):
    """The other half of the fix: a match that clearly overlaps an ALREADY-live track
    for the same query (the same animal, box shifted slightly between OWLv2 calls) must
    update that track in place -- not spawn a redundant second track for what's really
    the same ongoing sighting.
    """
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
        )
    )
    dispatcher._drain_results()
    first_id = next(iter(dispatcher.synthetic_tracked_objects("cam1", now=100.0)))

    # Second call: box shifted by 1 unit -- still heavily overlapping (high IoU).
    dispatcher._pending["req2"] = ("cam1", "frame-105.0", 105.0)
    result_q.put(
        OpenVocabResult(
            request_id="req2", camera_name="cam1", object_id="frame-105.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.5, box=(1.0, 1.0, 11.0, 11.0))],
        )
    )
    dispatcher._drain_results()

    live = dispatcher.synthetic_tracked_objects("cam1", now=105.0)
    assert len(live) == 1
    assert next(iter(live)) == first_id


def test_new_instance_appearing_alongside_an_existing_live_track_gets_its_own_id(db, frame_manager):
    """A second, spatially distinct animal appearing on a LATER OWLv2 call (not in the
    same batch as the first) must still become an independent track alongside the
    still-live first one -- not replace it, and not get merged into it.
    """
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
        )
    )
    dispatcher._drain_results()
    first_id = next(iter(dispatcher.synthetic_tracked_objects("cam1", now=100.0)))

    dispatcher._pending["req2"] = ("cam1", "frame-105.0", 105.0)
    result_q.put(
        OpenVocabResult(
            request_id="req2", camera_name="cam1", object_id="frame-105.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.4, box=(300.0, 300.0, 310.0, 310.0))],
        )
    )
    dispatcher._drain_results()

    live = dispatcher.synthetic_tracked_objects("cam1", now=105.0)
    assert len(live) == 2
    assert first_id in live


def test_stale_expired_track_is_not_matched_against_for_new_association(db, frame_manager):
    """An expired (past TTL) track must not "steal" a new match via IoU just because
    it's spatially close -- it should be treated as if it doesn't exist, so the new
    match becomes its own fresh track (with a NEW object_id, correctly starting a new
    Event rather than reanimating one that already logically ended).
    """
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
        )
    )
    dispatcher._drain_results()
    first_id = next(iter(dispatcher.synthetic_tracked_objects("cam1", now=100.0)))

    # Way past SYNTHETIC_TRACK_TTL_SECONDS (20s) -- the first track is now expired.
    later_time = 100.0 + 100.0
    dispatcher._pending["req2"] = ("cam1", f"frame-{later_time}", later_time)
    result_q.put(
        OpenVocabResult(
            request_id="req2", camera_name="cam1", object_id=f"frame-{later_time}",
            # Same box position as the expired track -- would be a perfect IoU match if
            # the expired track were wrongly still considered a live candidate.
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.5, box=(0.0, 0.0, 10.0, 10.0))],
        )
    )
    dispatcher._drain_results()

    live = dispatcher.synthetic_tracked_objects("cam1", now=later_time)
    assert len(live) == 1
    new_id = next(iter(live))
    assert new_id != first_id  # a genuinely NEW track, not the old (expired) one reanimated


def test_synthetic_tracks_are_scoped_per_camera(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
        )
    )
    dispatcher._drain_results()

    assert dispatcher.synthetic_tracked_objects("cam2", now=100.0) == {}
    assert len(dispatcher.synthetic_tracked_objects("cam1", now=100.0)) == 1


# --------------------------------------------------------------------------------------
# frame_jpeg propagation (TODO_FIX_LIST.md item 9/11) -- direct-frame-mode matches echo
# their whole-frame crop_jpeg back through to synthetic_frame_jpegs() so EventProcessor
# can give the resulting Event a real snapshot with a real box, instead of the
# no-shared-frame fallback (no boxes at all) it used before this was built.
# --------------------------------------------------------------------------------------


def test_direct_frame_mode_match_frame_jpeg_is_available_via_synthetic_frame_jpegs(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)

    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.42, box=(5.0, 10.0, 30.0, 40.0))],
            crop_jpeg=b"fake-whole-frame-jpeg-bytes",
        )
    )
    dispatcher._drain_results()

    live = dispatcher.synthetic_tracked_objects("cam1", now=100.5)
    assert len(live) == 1
    object_id = next(iter(live))

    frame_jpegs = dispatcher.synthetic_frame_jpegs("cam1")
    assert frame_jpegs == {object_id: b"fake-whole-frame-jpeg-bytes"}


def test_confirmed_object_mode_match_does_not_populate_synthetic_frame_jpegs(db, frame_manager):
    """Confirmed-object mode's object_id is the real closed-vocab tracker's id (no
    "frame-" prefix), and its crop_jpeg is a small object CROP, not a whole frame --
    using it as a snapshot background would be wrong, so it must be excluded here.
    """
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "real-tracker-obj-42", 100.0)

    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="real-tracker-obj-42",
            matches=[OpenVocabMatch(query_id="q1", query_text="a red backpack", score=0.42, box=(5.0, 10.0, 30.0, 40.0))],
            crop_jpeg=b"fake-small-crop-jpeg-bytes",
        )
    )
    dispatcher._drain_results()

    assert dispatcher.synthetic_tracked_objects("cam1", now=100.5)  # the track itself is still real
    assert dispatcher.synthetic_frame_jpegs("cam1") == {}


def test_synthetic_frame_jpeg_updates_when_an_existing_track_is_matched_again(db, frame_manager):
    """An overlapping later match updates the SAME synthetic track in place (see
    test_overlapping_match_for_same_query_updates_existing_track_not_a_new_one) -- its
    frame_jpeg must update too, so a later Event write reflects the MOST RECENT
    sighting's frame, not a stale first one.
    """
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)

    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
            crop_jpeg=b"first-frame",
        )
    )
    dispatcher._drain_results()
    first_id = next(iter(dispatcher.synthetic_tracked_objects("cam1", now=100.0)))
    assert dispatcher.synthetic_frame_jpegs("cam1") == {first_id: b"first-frame"}

    dispatcher._pending["req2"] = ("cam1", "frame-105.0", 105.0)
    result_q.put(
        OpenVocabResult(
            request_id="req2", camera_name="cam1", object_id="frame-105.0",
            # Overlapping box -> same track, per SYNTHETIC_TRACK_MATCH_IOU_THRESHOLD.
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.5, box=(0.0, 0.0, 12.0, 12.0))],
            crop_jpeg=b"second-frame",
        )
    )
    dispatcher._drain_results()
    second_id = next(iter(dispatcher.synthetic_tracked_objects("cam1", now=105.0)))

    assert second_id == first_id  # same track, updated in place
    assert dispatcher.synthetic_frame_jpegs("cam1") == {first_id: b"second-frame"}


def test_synthetic_frame_jpegs_excludes_expired_tracks(db, frame_manager):
    request_q, result_q = FakeQueue(), FakeQueue()
    dispatcher = OpenVocabDispatcher(request_q, result_q, frame_manager)
    dispatcher._pending["req1"] = ("cam1", "frame-100.0", 100.0)
    result_q.put(
        OpenVocabResult(
            request_id="req1", camera_name="cam1", object_id="frame-100.0",
            matches=[OpenVocabMatch(query_id="q1", query_text="animals", score=0.3, box=(0.0, 0.0, 10.0, 10.0))],
            crop_jpeg=b"stale-frame",
        )
    )
    dispatcher._drain_results()

    # Past SYNTHETIC_TRACK_TTL_SECONDS -- synthetic_tracked_objects() prunes it, and
    # synthetic_frame_jpegs() must be called AFTER that pruning (see its own docstring)
    # to correctly reflect the same live set, not a stale one.
    later_time = 100.0 + 100.0
    assert dispatcher.synthetic_tracked_objects("cam1", now=later_time) == {}
    assert dispatcher.synthetic_frame_jpegs("cam1") == {}
