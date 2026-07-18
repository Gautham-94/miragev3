"""ReviewSegmentMaintainer: aggregates tracked-object activity into human-reviewable
"review segments," distinct from individual object Events.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 9.
"""

from __future__ import annotations

import logging
import random
import string
import time
from dataclasses import dataclass, field
from enum import Enum

from mirage.config.schema import CameraConfig
from mirage.const import REVIEW_THUMB_DIR
from mirage.db.models import ReviewSegment
from mirage.tracking.rules import RULE_OBJECT_ID_PREFIX
from mirage.tracking.tracker import TrackedObjectState
from mirage.util.thumbnail import ThumbnailFetcher, capture_thumbnail
from mirage.util.time import utc_from_timestamp

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    alert = "alert"
    detection = "detection"
    # A camera-level derived-condition alert (crowd count / dwell-time -- see
    # mirage.tracking.rules.RulesEngine), distinct from `alert` above: `alert`/
    # `detection` are classified PER OBJECT LABEL (classify_severity), but a rule
    # trigger isn't about any one object's label -- it's a condition computed from the
    # current SET of tracked objects (a count, a duration). Always the highest
    # severity (see classify_severity's ordering) since a rule firing is, by
    # definition, something the user explicitly configured a threshold for.
    rule = "rule"


# Ordering used by ReviewSegmentMaintainer.process()'s highest-severity-wins logic and
# _upgrade_segment -- a segment only ever moves UP this ranking within its lifetime
# (detection -> alert -> rule), never back down, same as the pre-existing detection ->
# alert upgrade behavior this generalizes.
_SEVERITY_RANK = {Severity.detection: 0, Severity.alert: 1, Severity.rule: 2}


def _segment_id(start_time: float) -> str:
    rand6 = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{start_time}-{rand6}"


@dataclass
class PendingReviewSegment:
    id: str
    camera: str
    severity: Severity
    start_time: float
    last_activity_time: float
    detections: dict[str, str] = field(default_factory=dict)  # obj_id -> label
    # Historical set of every distinct label seen from a REAL (non-rule) object this
    # segment's lifetime -- correctly persists after that object leaves frame (e.g.
    # "person" stays in the list even once they're gone, if a "car" shows up later,
    # both are shown). A rule-triggered object's label is handled SEPARATELY (see
    # rule_labels below) and never added here -- its label text embeds a constantly
    # changing elapsed-seconds count (e.g. "loitering (120s)"), so treating it the same
    # way would accumulate a near-infinite set of near-identical strings, one per
    # second, instead of a single live-updating entry (a real bug the user caught live:
    # a genuine screenshot showing dozens of "Loitering (120s), Loitering (121s), ...").
    objects: set[str] = field(default_factory=set)  # labels
    # obj_id -> latest label, for RULE-triggered objects only (mirage.tracking.rules,
    # ids prefixed with RULE_OBJECT_ID_PREFIX) -- exactly one entry per distinct rule
    # trigger (e.g. one per loitering person, one for the camera's crowd condition),
    # always holding the MOST RECENT label text so the displayed count keeps updating
    # in place rather than growing a new entry every time it changes.
    rule_labels: dict[str, str] = field(default_factory=dict)
    zones: set[str] = field(default_factory=set)
    thumb_path: str | None = None
    # obj_id -> (label, [x1, y1, x2, y2] normalized 0-1) -- captured ONCE, at the exact
    # moment the thumbnail itself was captured (in _start_segment, from that same
    # process() call's tracked_objects), never updated afterward. This is deliberate:
    # the thumbnail is a single go2rtc snapshot taken at segment-start and never
    # refreshed, so a box drawn from a LATER frame's state.box would show the object in
    # a position it had already moved away from by the time this frame was captured
    # -- normalized (not raw pixel) coordinates so the frontend can scale them onto
    # the thumbnail's actual rendered size regardless of the snapshot's real resolution,
    # which can differ from camera.frame_shape's detect-resolution box coordinate space.
    thumb_boxes: dict[str, tuple[str, list[float]]] = field(default_factory=dict)

    def data(self) -> dict:
        return {
            "detections": list(self.detections.keys()),
            "objects": sorted(self.objects) + sorted(self.rule_labels.values()),
            "zones": sorted(self.zones),
            "thumb_boxes": {
                obj_id: {"label": label, "box": box} for obj_id, (label, box) in self.thumb_boxes.items()
            },
        }


def classify_severity(label: str, camera: CameraConfig, obj_id: str | None = None) -> Severity | None:
    """Section 9: an object is "alert"-worthy if its label is in the configured
    alert-label list; else "detection"-worthy under the (typically broader) detections
    list; else it doesn't qualify for review at all.

    A rule-triggered synthetic object (obj_id prefixed with mirage.tracking.rules.
    RULE_OBJECT_ID_PREFIX -- see that module) is always Severity.rule instead,
    regardless of its (dynamic, e.g. "crowd (12)") label text -- a rule condition
    firing is unconditionally alert-worthy by construction (the user explicitly
    configured a threshold for it), and its label was never meant to be matched
    against the alerts/detections lists at all.
    """
    if obj_id is not None and obj_id.startswith(RULE_OBJECT_ID_PREFIX):
        return Severity.rule
    if label in camera.review.alerts.labels:
        return Severity.alert
    if label in camera.review.detections.labels:
        return Severity.detection
    return None


def qualifies_for_review(state: TrackedObjectState, obj_id: str | None = None) -> bool:
    """Excludes stationary/never-moved/false-positive objects from counting toward
    review activity (section 9) -- EXCEPT a rule-triggered synthetic object (see
    classify_severity's docstring), which is exempt from the stationary check: dwell-
    time/loitering is specifically about objects that may not be moving much, so
    excluding them here would defeat the rule that's supposed to catch exactly that.
    """
    if state.is_false_positive:
        return False
    if obj_id is not None and obj_id.startswith(RULE_OBJECT_ID_PREFIX):
        return True
    if state.stationary.is_stationary():
        return False
    return True


def _normalized_boxes_at_thumb_time(
    camera: CameraConfig, tracked_objects: dict[str, TrackedObjectState]
) -> dict[str, tuple[str, list[float]]]:
    """Snapshots each REVIEW-QUALIFYING object's current box, normalized to [0, 1] by
    camera.frame_shape (the detect-resolution coordinate space state.box is already
    expressed in) -- called exactly once, from _start_segment, at the same instant the
    segment's thumbnail itself is captured. Normalized rather than raw pixel coordinates
    specifically because the thumbnail is a go2rtc live snapshot, not a frame decoded at
    detect resolution -- its actual JPEG dimensions can differ from frame_shape, so the
    frontend needs a resolution-independent box to scale onto whatever size it actually
    renders the thumbnail at. Uses the SAME qualifies_for_review/classify_severity gate
    as process()'s own `contributing` filter, so a stationary or non-alert-worthy object
    in frame doesn't get a box drawn on it too.
    """
    height, width = camera.frame_shape
    boxes: dict[str, tuple[str, list[float]]] = {}
    for obj_id, state in tracked_objects.items():
        if not qualifies_for_review(state, obj_id):
            continue
        if classify_severity(state.label, camera, obj_id) is None:
            continue
        x1, y1, x2, y2 = state.box
        boxes[obj_id] = (state.label, [x1 / width, y1 / height, x2 / width, y2 / height])
    return boxes


class ReviewSegmentMaintainer:
    def __init__(
        self,
        cutoff_seconds: float = 30.0,
        thumbnail_fetcher: ThumbnailFetcher | None = None,
        thumb_dir: str = REVIEW_THUMB_DIR,
    ) -> None:
        self.cutoff_seconds = cutoff_seconds
        self.thumbnail_fetcher = thumbnail_fetcher
        self.thumb_dir = thumb_dir
        self._pending: dict[str, PendingReviewSegment] = {}  # camera -> pending segment

    def process(
        self,
        camera: CameraConfig,
        frame_time: float,
        tracked_objects: dict[str, TrackedObjectState],
        frame_jpeg: bytes | None = None,
    ) -> None:
        # frame_jpeg (the exact detected frame, see
        # mirage.tracking.camera_tracker._maybe_encode_frame) is accepted for call-site
        # symmetry with EventProcessor.process, which DOES use it (see
        # mirage.events.processor's write_boxed_snapshot fix) -- ReviewSegmentMaintainer
        # itself still uses the older live-go2rtc-fetch thumbnail path unchanged
        # (capture_thumbnail below), since Review-page boxes are intentionally not
        # shown at all right now (user request).
        pending = self._pending.get(camera.name)

        contributing: list[tuple[str, str, Severity]] = []  # (obj_id, label, severity)
        for obj_id, state in tracked_objects.items():
            if not qualifies_for_review(state, obj_id):
                continue
            severity = classify_severity(state.label, camera, obj_id)
            if severity is None:
                continue
            contributing.append((obj_id, state.label, severity))

        if contributing:
            highest_severity = max((s for _, _, s in contributing), key=_SEVERITY_RANK.__getitem__)

            if pending is None:
                pending = self._start_segment(camera, frame_time, highest_severity, tracked_objects)
            elif _SEVERITY_RANK[highest_severity] > _SEVERITY_RANK[pending.severity]:
                self._upgrade_segment(pending, frame_time, highest_severity)

            for obj_id, label, _ in contributing:
                pending.detections[obj_id] = label
                # A rule-triggered object's label embeds a constantly-changing count
                # (e.g. "loitering (120s)" -> "(121s)" -> ...) -- keyed by obj_id here
                # so the SAME entry updates in place, instead of pending.objects (a
                # plain set) accumulating a new, never-deduplicated string every time
                # the count changes (the real bug this replaces -- see
                # PendingReviewSegment.rule_labels's own docstring).
                if obj_id.startswith(RULE_OBJECT_ID_PREFIX):
                    pending.rule_labels[obj_id] = label
                else:
                    pending.objects.add(label)
            pending.last_activity_time = frame_time
            self._save(pending, end_time=None)

        elif pending is not None:
            elapsed = frame_time - pending.last_activity_time
            if elapsed >= self.cutoff_seconds:
                self._end_segment(pending, frame_time)

    def _start_segment(
        self,
        camera: CameraConfig,
        frame_time: float,
        severity: Severity,
        tracked_objects: dict[str, TrackedObjectState],
    ) -> PendingReviewSegment:
        segment = PendingReviewSegment(
            id=_segment_id(frame_time), camera=camera.name, severity=severity,
            start_time=frame_time, last_activity_time=frame_time,
        )
        segment.thumb_path = capture_thumbnail(self.thumbnail_fetcher, self.thumb_dir, segment.id, camera.name)
        segment.thumb_boxes = _normalized_boxes_at_thumb_time(camera, tracked_objects)
        self._pending[camera.name] = segment
        logger.debug("%s: review segment %s started (%s)", camera.name, segment.id, severity)
        return segment

    def _upgrade_segment(self, pending: PendingReviewSegment, frame_time: float, new_severity: Severity) -> None:
        """Section 9: a segment can be upgraded in place to a higher severity if a
        higher-ranked contributor appears (detection -> alert, or either -> rule) --
        never downgraded, same as the original detection -> alert-only behavior this
        generalizes to a third rank (see _SEVERITY_RANK).
        """
        logger.debug("%s: review segment %s upgraded to %s", pending.camera, pending.id, new_severity.value)
        pending.severity = new_severity

    def _end_segment(self, pending: PendingReviewSegment, frame_time: float) -> None:
        self._save(pending, end_time=frame_time)
        del self._pending[pending.camera]
        logger.debug("%s: review segment %s ended", pending.camera, pending.id)

    def _save(self, pending: PendingReviewSegment, end_time: float | None) -> None:
        defaults = {
            "camera": pending.camera,
            "start_time": utc_from_timestamp(pending.start_time),
            "end_time": utc_from_timestamp(end_time) if end_time is not None else None,
            "severity": pending.severity.value,
            "thumb_path": pending.thumb_path,
            "data": pending.data(),
        }
        existing = ReviewSegment.get_or_none(ReviewSegment.id == pending.id)
        if existing is None:
            ReviewSegment.create(id=pending.id, **defaults)
        else:
            ReviewSegment.update(**defaults).where(ReviewSegment.id == pending.id).execute()

    def close_all_pending(self, frame_time: float | None = None) -> None:
        """Called on shutdown: force-close any still-open review segments (analogous to
        section 10.2's dangling-Event cleanup).
        """
        # NOTE: datetime.timestamp() on a NAIVE datetime assumes it represents LOCAL
        # time, not UTC -- utcnow() deliberately returns a naive-but-UTC datetime for
        # safe DB storage (see mirage/util/time.py), so calling .timestamp() on it here
        # would silently apply the wrong (local) offset. Use time.time() directly
        # instead, which is already a UTC-based Unix timestamp by definition.
        ts = frame_time if frame_time is not None else time.time()
        for camera_name in list(self._pending.keys()):
            self._end_segment(self._pending[camera_name], ts)
