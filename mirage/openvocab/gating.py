"""Gating logic deciding WHEN a tracked object's crop actually gets sent to the (slow,
~1-1.7s per call, see TODO_FIX_LIST.md item 6) OWLv2 process -- this is the whole reason
open-vocab matching stays practical: without gating, every confirmed object would be
re-checked every single frame it's tracked, which would immediately overwhelm a
multi-second-per-call model. Two independent gates, both must pass:

Gate 1 (track confirmed): only ever consider an object once its ObjectLifecycle has
flipped `is_false_positive` False -> True (mirrored onto TrackedObjectState, see
mirage/tracking/tracker.py's docstring) -- i.e. it has already survived norfair's
initialization_delay warm-up and the configured score/area filters. This is the SAME
signal that already gates real Event/ReviewSegment creation elsewhere in this codebase,
reused here rather than inventing a second "is this real" heuristic.

Gate 2 (perceptual-hash dedup): even after an object is confirmed, don't re-query OWLv2
every frame it's merely still visible and unchanged (e.g. an animal standing still for a
minute) -- compare a cheap perceptual hash of the object's current crop against the hash
of whichever crop was last actually sent to OWLv2 for this same object id, and only
dispatch again once the Hamming distance exceeds a threshold (pose changed, lighting
shifted, a second object entered the crop, etc). Deliberately NOT a full CLIP embedding
encoder (a third heavy model) -- a simple downsampled average-hash is enough to catch
"this looks meaningfully different," which is all this gate needs to do; see
TODO_FIX_LIST.md item 4's original CLIP suggestion for context on why this is a
deliberately cheaper substitute, agreed with the user in this session.
"""

from __future__ import annotations

import time

import numpy as np

HASH_SIZE = 8  # 8x8 -> 64-bit hash, the same size python-imagehash's average_hash defaults to
DEFAULT_HAMMING_THRESHOLD = 10  # out of 64 bits -- empirically reasonable starting point, not tuned
MIN_RECHECK_INTERVAL_SECONDS = 5.0  # even if the hash looks unchanged, allow a periodic recheck


def average_hash(crop_rgb: np.ndarray) -> int:
    """A minimal from-scratch average-hash (avoids pulling in the `imagehash` package for
    something this small): downsample to HASH_SIZE x HASH_SIZE grayscale, threshold each
    pixel against the mean, pack into a single Python int as a 64-bit bitmask.
    """
    from PIL import Image

    img = Image.fromarray(crop_rgb).convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    mean = pixels.mean()
    bits = pixels > mean
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class OpenVocabGate:
    """Per-object dispatch state, owned by the main process's result-consumer loop (one
    instance shared across all cameras, keyed by TrackedObjectState.id -- object ids are
    already globally unique, see mirage/tracking/tracker.py's _new_id).
    """

    def __init__(
        self,
        hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
        min_recheck_interval: float = MIN_RECHECK_INTERVAL_SECONDS,
    ) -> None:
        self.hamming_threshold = hamming_threshold
        self.min_recheck_interval = min_recheck_interval
        self._last_sent_hash: dict[str, int] = {}
        self._last_sent_time: dict[str, float] = {}
        self._in_flight: set[str] = set()

    def should_dispatch(self, object_id: str, crop_rgb: np.ndarray, now: float | None = None) -> bool:
        """Gate 2 -- call only for objects that already passed Gate 1 (is_false_positive
        is False) in the caller. Returns True if this crop should be sent to OWLv2 now,
        and if so, records it as sent (idempotent within one call -- the caller is
        expected to actually dispatch immediately after a True return).
        """
        if object_id in self._in_flight:
            return False  # a request for this object is already outstanding

        now = now if now is not None else time.time()
        current_hash = average_hash(crop_rgb)

        last_hash = self._last_sent_hash.get(object_id)
        last_time = self._last_sent_time.get(object_id, 0.0)

        if last_hash is not None:
            distance = hamming_distance(current_hash, last_hash)
            changed_enough = distance > self.hamming_threshold
            recheck_due = (now - last_time) >= self.min_recheck_interval
            if not changed_enough and not recheck_due:
                return False

        self._last_sent_hash[object_id] = current_hash
        self._last_sent_time[object_id] = now
        self._in_flight.add(object_id)
        return True

    def mark_result_received(self, object_id: str) -> None:
        self._in_flight.discard(object_id)

    def forget(self, object_id: str) -> None:
        """Called when an object's track ends -- drops its state so a future, unrelated
        object that happens to reuse... well, ids are never reused (see _new_id), but
        this bounds memory for long-running cameras rather than accumulating forever.
        """
        self._last_sent_hash.pop(object_id, None)
        self._last_sent_time.pop(object_id, None)
        self._in_flight.discard(object_id)
