"""OpenVocabProcess: the single dedicated OS process that loads OWLv2 once and serves
open-vocabulary query-matching requests for every camera, mirroring the "one process,
one loaded model, shared via a queue" pattern mirage.detection.process.DetectorProcess
already uses for the fast YOLOv8n detector (see NVR_PIPELINE_IMPLEMENTATION_SPEC.md
section 3.2) -- NOT the same IPC mechanism, though: DetectorProcess uses SHM + a tiny
ZMQ "done" pub/sub signal tuned for millisecond-scale, fixed-shape, many-times-per-second
calls. OWLv2 calls are multi-second (~1-1.7s measured on this repo's own dev machine,
see TODO_FIX_LIST.md item 6) and happen at most a handful of times a minute (gated by
mirage.openvocab.gating -- only on already-confirmed tracked objects that have visibly
changed since last checked), so a plain multiprocessing.Queue carrying small JPEG-encoded
crops directly is simpler and entirely sufficient; SHM's zero-copy benefit doesn't matter
at this call volume/payload size.

Runs on whichever torch device is configured (cpu/mps/cuda) -- see
mirage.openvocab.device for the resolution logic, deliberately separate from
mirage.detection.execution_providers since that module is onnxruntime-specific and OWLv2
uses PyTorch/transformers, a different runtime with its own device API
(torch.device(...) vs onnxruntime's providers=[...] list).
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import queue as queue_module
import time

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OpenVocabRequest:
    request_id: str
    camera_name: str
    object_id: str
    crop_jpeg: bytes
    # The crop's box in the ORIGINAL full-resolution frame's pixel coords (y1, x1, y2, x2)
    # -- OWLv2 runs on just the cropped region for speed/focus, but a match's box needs
    # to be reported in full-frame coords for display, so this offset is carried through.
    frame_box: tuple[float, float, float, float]
    queries: list[tuple[str, str]]  # (query_id, query_text) pairs to check this crop against


@dataclasses.dataclass
class OpenVocabMatch:
    query_id: str
    query_text: str
    score: float
    box: tuple[float, float, float, float]  # full-frame pixel coords


@dataclasses.dataclass
class OpenVocabResult:
    request_id: str
    camera_name: str
    object_id: str
    matches: list[OpenVocabMatch]
    # Echoes the request's own crop back so a match can be saved as a real thumbnail
    # (mirage.openvocab.dispatcher._save_thumb) without OpenVocabDispatcher needing to
    # hang onto the original crop bytes itself across the request's in-flight lifetime.
    crop_jpeg: bytes = b""


def openvocab_process_main(
    request_queue: "mp.Queue",
    result_queue: "mp.Queue",
    stop_event,
    device: str = "cpu",
    model_id: str = "google/owlv2-base-patch16-ensemble",
    score_threshold: float = 0.15,
) -> None:
    """Runs as the single dedicated OpenVocabProcess OS process. Loads OWLv2 exactly
    once here, regardless of how many cameras/queries exist -- camera tracker processes
    never load this model themselves, and the main process only ever sends small crop +
    query-list requests through request_queue.
    """
    logging.basicConfig(level=logging.INFO)

    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    logger.info("openvocab: loading %s on device=%s ...", model_id, device)
    t0 = time.time()
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)
    logger.info("openvocab: ready in %.1fs", time.time() - t0)

    while not stop_event.is_set():
        try:
            request: OpenVocabRequest = request_queue.get(timeout=1)
        except queue_module.Empty:
            continue
        except (OSError, EOFError):
            break

        try:
            result = _process_request(request, processor, model, torch_device, score_threshold)
        except Exception:
            logger.exception("openvocab: inference failed for camera %s object %s", request.camera_name, request.object_id)
            result = OpenVocabResult(request.request_id, request.camera_name, request.object_id, matches=[], crop_jpeg=b"")

        try:
            result_queue.put(result, block=False)
        except queue_module.Full:
            logger.warning("openvocab: result_queue full, dropping result for %s", request.object_id)

    logger.info("openvocab: stopped")


def _process_request(request: OpenVocabRequest, processor, model, torch_device, score_threshold: float) -> OpenVocabResult:
    import io

    import torch
    from PIL import Image

    image = Image.open(io.BytesIO(request.crop_jpeg)).convert("RGB")
    query_texts = [text for _, text in request.queries]

    inputs = processor(text=[query_texts], images=image, return_tensors="pt").to(torch_device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    processed = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=score_threshold
    )[0]

    crop_width, crop_height = image.size
    crop_y1, crop_x1, _, _ = request.frame_box
    matches: list[OpenVocabMatch] = []
    for score, label_idx, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
        query_id, query_text = request.queries[int(label_idx)]
        x1, y1, x2, y2 = (float(v) for v in box)
        # OWLv2's regressed box coordinates can slightly overshoot the crop's own pixel
        # bounds (normal regression imprecision, not a bug) -- clamp to the crop's own
        # size BEFORE offsetting into full-frame coords, same principle as
        # mirage/tracking/orchestration.py's clamp of the closed-vocab detector's boxes
        # to frame_shape.
        x1 = max(0.0, min(x1, crop_width))
        x2 = max(0.0, min(x2, crop_width))
        y1 = max(0.0, min(y1, crop_height))
        y2 = max(0.0, min(y2, crop_height))
        # box is in the CROP's own pixel coords -- offset back into the full frame using
        # the crop's own top-left corner (frame_box), so downstream display/DB storage
        # doesn't need to know a crop was ever involved.
        full_frame_box = (y1 + crop_y1, x1 + crop_x1, y2 + crop_y1, x2 + crop_x1)
        matches.append(OpenVocabMatch(query_id=query_id, query_text=query_text, score=float(score), box=full_frame_box))

    return OpenVocabResult(request.request_id, request.camera_name, request.object_id, matches=matches, crop_jpeg=request.crop_jpeg)


class OpenVocabProcess(mp.Process):
    def __init__(
        self,
        request_queue: "mp.Queue",
        result_queue: "mp.Queue",
        stop_event,
        device: str = "cpu",
        model_id: str = "google/owlv2-base-patch16-ensemble",
        score_threshold: float = 0.15,
    ) -> None:
        super().__init__(name="openvocab")
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = device
        self.model_id = model_id
        self.score_threshold = score_threshold

    def run(self) -> None:
        openvocab_process_main(
            self.request_queue, self.result_queue, self.stop_event,
            device=self.device, model_id=self.model_id, score_threshold=self.score_threshold,
        )
