"""SpeciesDispatcher: lives in the main process's result-consumer loop
(mirage.app.MirageApp._result_consumer_loop), alongside EventProcessor and
ReviewSegmentMaintainer -- mirrors mirage.openvocab.dispatcher.OpenVocabDispatcher's
role as the main-process side of a request/result mp.Queue pair, but considerably
simpler: species classification fires exactly ONCE per Event's lifetime (at Event
creation, from EventProcessor._on_start), never per-frame, so there's no synthetic-track
bridge and no perceptual-hash dedup gate to build (OpenVocabDispatcher needs both
because it re-checks the SAME tracked object across many frames as it changes).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as queue_module
import uuid

from mirage.db.models import Event
from mirage.logging_bus import LogEvent
from mirage.species.process import SpeciesRequest, SpeciesResult

logger = logging.getLogger(__name__)


class SpeciesDispatcher:
    def __init__(self, request_queue: "mp.Queue", result_queue: "mp.Queue", activity_log_queue=None) -> None:
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.activity_log_queue = activity_log_queue

    def dispatch(self, event_id: str, camera_name: str, label: str, crop_jpeg: bytes) -> None:
        """Fire-and-forget -- called synchronously from EventProcessor._on_start, so
        this must never block. A full queue (species worker falling behind) just drops
        the request and logs; the Event's species_status stays "pending" forever in
        that case, which is an acceptable, visible degradation (distinguishable from
        "complete"/"failed" in the UI) rather than blocking event creation.
        """
        request = SpeciesRequest(
            request_id=str(uuid.uuid4()), event_id=event_id, camera_name=camera_name, label=label, crop_jpeg=crop_jpeg,
        )
        try:
            self.request_queue.put(request, block=False)
        except queue_module.Full:
            logger.warning("species: request_queue full, dropping species classification request for event %s", event_id)

    def drain_results(self) -> None:
        """Non-blocking drain of every result currently available, applying each via a
        read-modify-write Event.update() -- see mirage.events.processor.EventProcessor.
        _on_update's own comment for why this MUST be read-modify-write, not a blind
        literal: EventProcessor's own throttled/heartbeat updates write to the SAME
        Event.data JSON field at arbitrary, unrelated times, and Event.update(data=...)
        replaces the whole field. Reading the current row here and merging in only the
        species_* keys is symmetric with that fix -- neither writer can clobber the
        other's fields, regardless of interleaving order.
        """
        while True:
            try:
                result: SpeciesResult = self.result_queue.get_nowait()
            except queue_module.Empty:
                break
            except (OSError, EOFError):
                break
            self._apply_result(result)

    def _apply_result(self, result: SpeciesResult) -> None:
        event = Event.get_or_none(Event.id == result.event_id)
        if event is None:
            # Event may have been deleted/pruned by the time classification finished --
            # not an error, just nothing left to enrich.
            return

        current_data = dict(event.data) if event.data else {}
        current_data["species"] = result.species_common or result.species_scientific
        current_data["species_status"] = result.status
        current_data["species_confidence"] = result.confidence
        current_data["species_taxonomy"] = result.taxonomy
        current_data["species_model"] = result.model_name

        Event.update(data=current_data).where(Event.id == result.event_id).execute()
        logger.debug("species: event %s enriched, status=%s", result.event_id, result.status)

        if self.activity_log_queue is not None:
            species_name = result.species_common or result.species_scientific or "unknown"
            try:
                self.activity_log_queue.put_nowait(
                    LogEvent(category="species", message=f"classified as {species_name}", camera=None)
                )
            except Exception:
                pass  # log queue backpressure/full -- never let logging affect enrichment
