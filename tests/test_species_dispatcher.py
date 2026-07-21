"""Tests for mirage/species/dispatcher.py (SpeciesDispatcher) and the request-processing
logic in mirage/species/process.py -- using an in-process fake queue pair rather than a
real multiprocessing.Queue/Process, since these tests only need to exercise the
dispatch()/drain_results() logic and _process_request()'s classification-to-DB-write
mapping, not real cross-process IPC.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from mirage.db.database import close_database, init_database
from mirage.db.models import Event
from mirage.species.dispatcher import SpeciesDispatcher
from mirage.species.process import SpeciesRequest, SpeciesResult, _process_request
from mirage.species.registry import create_classifier
from mirage.config.schema import SpeciesClassifierConfig


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = init_database(str(Path(tmp) / "test.db"))
        yield database
        close_database(database)


class _FakeQueue:
    """Minimal stand-in for multiprocessing.Queue -- a plain list-backed FIFO, since
    these tests run entirely in one process/thread and never need real IPC.
    """

    def __init__(self):
        self._items = []

    def put(self, item, block=True):
        self._items.append(item)

    def get_nowait(self):
        import queue as queue_module

        if not self._items:
            raise queue_module.Empty
        return self._items.pop(0)


def _real_jpeg(width: int = 40, height: int = 30) -> bytes:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _make_event(db, event_id: str = "ev1", **data_overrides) -> Event:
    import datetime

    data = {
        "box": [0, 0, 10, 10], "snapshot_boxes": [], "species": None,
        "species_status": "pending", "species_confidence": None, "species_taxonomy": None, "species_model": None,
    }
    data.update(data_overrides)
    return Event.create(
        id=event_id, label="animal", camera="cam1",
        start_time=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
        end_time=None, data=data,
    )


def test_dispatch_puts_a_request_on_the_queue():
    request_queue = _FakeQueue()
    result_queue = _FakeQueue()
    dispatcher = SpeciesDispatcher(request_queue, result_queue)

    dispatcher.dispatch("ev1", "cam1", "animal", b"\xff\xd8fakejpeg")

    assert len(request_queue._items) == 1
    request: SpeciesRequest = request_queue._items[0]
    assert request.event_id == "ev1"
    assert request.camera_name == "cam1"
    assert request.label == "animal"
    assert request.crop_jpeg == b"\xff\xd8fakejpeg"


def test_dispatch_swallows_a_full_queue_without_raising():
    class FullQueue(_FakeQueue):
        def put(self, item, block=True):
            import queue as queue_module

            raise queue_module.Full

    dispatcher = SpeciesDispatcher(FullQueue(), _FakeQueue())
    dispatcher.dispatch("ev1", "cam1", "animal", b"\xff\xd8fakejpeg")  # must not raise


def test_drain_results_applies_a_complete_result_to_the_event(db):
    _make_event(db, "ev1")
    result_queue = _FakeQueue()
    result_queue.put(SpeciesResult(
        request_id="r1", event_id="ev1", status="complete",
        species_common="White-tailed Deer", species_scientific="Odocoileus virginianus",
        confidence=0.93, taxonomy={"class": "Mammalia"}, model_name="speciesnet",
    ))
    dispatcher = SpeciesDispatcher(_FakeQueue(), result_queue)

    dispatcher.drain_results()

    row = Event.get(Event.id == "ev1")
    assert row.data["species"] == "White-tailed Deer"
    assert row.data["species_status"] == "complete"
    assert row.data["species_confidence"] == 0.93
    assert row.data["species_taxonomy"] == {"class": "Mammalia"}
    assert row.data["species_model"] == "speciesnet"
    # box/snapshot_boxes (owned by EventProcessor) must survive untouched.
    assert row.data["box"] == [0, 0, 10, 10]


def test_drain_results_preserves_other_data_keys_written_by_event_processor(db):
    """The species dispatcher's own read-modify-write must not clobber fields it
    doesn't own -- symmetric with EventProcessor._on_update's fix for the same hazard.
    """
    _make_event(db, "ev1", snapshot_boxes=[{"label": "animal", "box": [1, 2, 3, 4]}])
    result_queue = _FakeQueue()
    result_queue.put(SpeciesResult(request_id="r1", event_id="ev1", status="failed", model_name="speciesnet"))
    dispatcher = SpeciesDispatcher(_FakeQueue(), result_queue)

    dispatcher.drain_results()

    row = Event.get(Event.id == "ev1")
    assert row.data["snapshot_boxes"] == [{"label": "animal", "box": [1, 2, 3, 4]}]
    assert row.data["species_status"] == "failed"


def test_drain_results_for_a_deleted_event_does_not_raise(db):
    result_queue = _FakeQueue()
    result_queue.put(SpeciesResult(request_id="r1", event_id="does-not-exist", status="complete"))
    dispatcher = SpeciesDispatcher(_FakeQueue(), result_queue)

    dispatcher.drain_results()  # must not raise


def test_drain_results_drains_multiple_queued_results():
    request_queue = _FakeQueue()
    result_queue = _FakeQueue()
    dispatcher = SpeciesDispatcher(request_queue, result_queue)
    for i in range(3):
        result_queue.put(SpeciesResult(request_id=f"r{i}", event_id=f"ev{i}", status="skipped"))

    applied = []
    dispatcher._apply_result = lambda result: applied.append(result.event_id)  # type: ignore[method-assign]
    dispatcher.drain_results()

    assert applied == ["ev0", "ev1", "ev2"]


def test_process_request_with_noop_classifier_returns_skipped():
    classifier = create_classifier(SpeciesClassifierConfig(device="noop"))
    request = SpeciesRequest(request_id="r1", event_id="ev1", camera_name="cam1", label="animal", crop_jpeg=_real_jpeg())

    result = _process_request(request, classifier, "noop")

    assert result.status == "skipped"
    assert result.model_name == "noop"


def test_process_request_with_invalid_jpeg_returns_failed():
    classifier = create_classifier(SpeciesClassifierConfig(device="noop"))
    request = SpeciesRequest(request_id="r1", event_id="ev1", camera_name="cam1", label="animal", crop_jpeg=b"not a jpeg")

    result = _process_request(request, classifier, "noop")

    assert result.status == "failed"
