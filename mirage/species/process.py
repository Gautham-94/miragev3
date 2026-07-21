"""SpeciesProcess: the single dedicated OS process that loads the configured species
classifier once and serves species-classification requests for every camera's
Animal/Bird events -- mirrors mirage.openvocab.process.OpenVocabProcess's "one process,
one loaded model, shared via a queue" pattern exactly.

Plain multiprocessing.Queue, not SHM, for the same reason OpenVocabProcess's own module
docstring gives: species classification only ever fires once per Event's lifetime (at
Event creation, see mirage.events.processor.EventProcessor._on_start), not per-frame --
infrequent, small JPEG-crop payloads, inference-latency-dominated. SHM's zero-copy
benefit doesn't matter at this call volume/payload size.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import queue as queue_module
import time

from mirage.config.schema import SpeciesClassifierConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpeciesRequest:
    request_id: str
    event_id: str  # the Event row to update -- dispatch happens once, at Event creation
    camera_name: str
    label: str  # "animal" | "bird" -- passed through as a prior/hint for the classifier
    crop_jpeg: bytes


@dataclasses.dataclass
class SpeciesResult:
    request_id: str
    event_id: str
    status: str  # "complete" | "failed" | "skipped"
    species_common: str | None = None
    species_scientific: str | None = None
    confidence: float | None = None
    taxonomy: dict | None = None
    model_name: str | None = None  # the backend's type_key, e.g. "speciesnet" -- which
    # plugin/version produced this result, for future re-classification/debugging


def species_process_main(
    request_queue: "mp.Queue",
    result_queue: "mp.Queue",
    stop_event,
    classifier_config: SpeciesClassifierConfig,
) -> None:
    """Runs as the single dedicated SpeciesProcess OS process. Loads the configured
    classifier backend exactly once here, regardless of how many cameras exist --
    camera tracker processes never load a species model themselves, and the main
    process only ever sends small crop-jpeg requests through request_queue.
    """
    logging.basicConfig(level=logging.INFO)

    from mirage.species.registry import create_classifier

    logger.info("species: loading backend=%s ...", classifier_config.device)
    t0 = time.time()
    classifier = create_classifier(classifier_config)
    logger.info("species: ready in %.1fs", time.time() - t0)

    while not stop_event.is_set():
        try:
            request: SpeciesRequest = request_queue.get(timeout=1)
        except queue_module.Empty:
            continue
        except (OSError, EOFError):
            break

        try:
            result = _process_request(request, classifier, classifier_config.device)
        except Exception:
            logger.exception("species: classification failed for event %s (camera %s)", request.event_id, request.camera_name)
            result = SpeciesResult(
                request_id=request.request_id, event_id=request.event_id, status="failed",
                model_name=classifier_config.device,
            )

        try:
            result_queue.put(result, block=False)
        except queue_module.Full:
            logger.warning("species: result_queue full, dropping result for event %s", request.event_id)

    logger.info("species: stopped")


def _process_request(request: SpeciesRequest, classifier, model_name: str) -> SpeciesResult:
    import cv2
    import numpy as np

    arr = np.frombuffer(request.crop_jpeg, dtype=np.uint8)
    crop_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if crop_bgr is None:
        return SpeciesResult(request_id=request.request_id, event_id=request.event_id, status="failed", model_name=model_name)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    classification = classifier.classify(crop_rgb)
    if classification.species_common is None and classification.species_scientific is None:
        # A deliberate "no confident result" outcome (see SpeciesClassification's own
        # docstring) -- not an error, just nothing worth recording.
        return SpeciesResult(request_id=request.request_id, event_id=request.event_id, status="skipped", model_name=model_name)

    return SpeciesResult(
        request_id=request.request_id,
        event_id=request.event_id,
        status="complete",
        species_common=classification.species_common,
        species_scientific=classification.species_scientific,
        confidence=classification.confidence,
        taxonomy=classification.taxonomy,
        model_name=model_name,
    )


class SpeciesProcess(mp.Process):
    def __init__(
        self,
        request_queue: "mp.Queue",
        result_queue: "mp.Queue",
        stop_event,
        classifier_config: SpeciesClassifierConfig,
    ) -> None:
        super().__init__(name="species")
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.classifier_config = classifier_config

    def run(self) -> None:
        species_process_main(self.request_queue, self.result_queue, self.stop_event, self.classifier_config)
