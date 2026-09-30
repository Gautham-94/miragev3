"""EventProcessor: consumes detected-frame results and maintains Event DB rows across
each tracked object's start/update/end lifecycle.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 6.3.
"""

from __future__ import annotations

import datetime
import logging
import random
import string

from mirage.config.schema import CameraConfig
from mirage.const import EVENT_SNAPSHOT_DIR
from mirage.db.models import Event
from mirage.logging_bus import LogEvent
from mirage.notify_bus import NotifyEvent
from mirage.tracking.tracker import TrackedObjectState
from mirage.util.thumbnail import ThumbnailFetcher, capture_thumbnail, crop_jpeg_to_box, write_clean_snapshot
from mirage.util.time import utc_from_timestamp, utcnow

logger = logging.getLogger(__name__)

UPDATE_PUBLISH_THROTTLE_SECONDS = 5.0
HEARTBEAT_FORCE_SECONDS = 60.0

# Mirage V3: only these labels are ever pushed to the (optional) species classifier --
# Person/Vehicle events are never enrichable, so they're stamped "not_applicable" once
# at creation and never touched again. Deliberately a fixed constant, not user-facing
# config -- see NVR V3 architecture doc / species classification plan.
SPECIES_ENRICHABLE_LABELS = frozenset({"animal", "bird"})


def _event_id(start_time: float) -> str:
    rand6 = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{start_time}-{rand6}"


def _to_datetime(ts: float) -> datetime.datetime:
    return utc_from_timestamp(ts)


def _confirmed_snapshot_boxes(tracked_objects: dict[str, TrackedObjectState]) -> list[dict]:
    """Every Gate-1-confirmed (is_false_positive is False) object's box in this frame,
    as plain JSON-serializable dicts for Event.data["snapshot_boxes"] -- same bar
    CameraTracker._has_qualifying_object already uses to decide whether to encode the
    frame at all, and the same bar used everywhere else in the pipeline (open-vocab
    dispatch, Review severity) for "is this a real detection, not noise." A still-
    unconfirmed object (mid initialization_delay) is deliberately excluded -- it might
    still turn out to be a false positive, so it shouldn't get boxed as if it were a
    real confirmed detection. Also excludes any object whose box isn't currently "live"
    (TrackedObjectState.is_live -- both corners matched a real detection recently, not
    just Kalman-coasting): the object stays tracked so it doesn't flicker away during a
    brief gap, but its box is then a motion prediction, not an observation, and can
    visibly drift from the object's real position in this exact frame -- confirmed live
    via real stored snapshot_boxes with inverted (x2<x1/y2<y1) coordinates, a symptom
    of exactly this. Rendered on demand at request time by
    mirage/api/routers/events.py, via mirage.util.thumbnail.draw_boxes_on_jpeg_bytes --
    see that module's own docstring for why boxes are stored as data, not burned into
    the saved file (mirrors Frigate's own confirmed design, frigate/util/image.py).
    """
    return [
        {"label": state.label, "box": list(state.box)}
        for state in tracked_objects.values()
        if not state.is_false_positive and state.is_live
    ]


class EventProcessor:
    """Section 6.3: diffs the tracker's current confirmed object-id set against the
    previous frame's set, per camera, and drives Event DB row creation/update/close.
    """

    def __init__(self, thumbnail_fetcher: ThumbnailFetcher | None = None, thumb_dir: str = EVENT_SNAPSHOT_DIR) -> None:
        # obj_id -> (event_id, last_published_time, last_top_score)
        self._active: dict[str, dict] = {}
        self._previous_ids_by_camera: dict[str, set[str]] = {}
        self.thumbnail_fetcher = thumbnail_fetcher
        self.thumb_dir = thumb_dir
        # Set by MirageApp AFTER construction, once _start_species_worker() has decided
        # whether a species classifier is configured/enabled -- see mirage.app.MirageApp
        # for why this can't be passed in at __init__ time (EventProcessor is
        # constructed before the species worker is started). None means "no species
        # classification available," in which case _on_start skips dispatch entirely
        # and stamps species_status="skipped" instead of "pending".
        self.species_dispatcher = None
        # Set by MirageApp AFTER construction, same as species_dispatcher above -- an
        # mp.Queue for the Logs page (see mirage.logging_bus). None (e.g. in tests that
        # construct EventProcessor directly) means logging is silently skipped.
        self.activity_log_queue = None
        # Set by MirageApp AFTER construction, same pattern -- an mp.Queue feeding the
        # frontend's live SSE stream (see mirage.notify_bus). None means notification is
        # silently skipped (e.g. direct-construction tests).
        self.notify_queue = None

    def _log(self, category: str, message: str, camera_name: str | None = None) -> None:
        if self.activity_log_queue is None:
            return
        try:
            self.activity_log_queue.put_nowait(LogEvent(category=category, message=message, camera=camera_name))
        except Exception:
            pass  # log queue backpressure/full -- never let logging affect event processing

    def _notify(self, table: str, row_id: str, op: str) -> None:
        if self.notify_queue is None:
            return
        try:
            self.notify_queue.put_nowait(NotifyEvent(table=table, id=row_id, op=op))
        except Exception:
            pass  # notify queue backpressure/full -- never let SSE affect event processing

    def process(
        self,
        camera: CameraConfig,
        frame_time: float,
        tracked_objects: dict[str, TrackedObjectState],
        frame_jpeg: bytes | None = None,
    ) -> None:
        previous_ids = self._previous_ids_by_camera.get(camera.name, set())
        current_ids = set(tracked_objects.keys())

        new_ids = current_ids - previous_ids
        updated_ids = current_ids & previous_ids
        removed_ids = previous_ids - current_ids

        for obj_id in new_ids:
            self._on_start(camera, obj_id, tracked_objects[obj_id], frame_time, frame_jpeg, tracked_objects)

        for obj_id in updated_ids:
            self._on_update(camera, obj_id, tracked_objects[obj_id], frame_time)

        for obj_id in removed_ids:
            # `removed_ids` objects are no longer in tracked_objects (that's precisely
            # why they're "removed"), so their FINAL is_false_positive/score reading was
            # whatever showed up the last time they WERE in tracked_objects -- captured
            # into self._active below, in _on_update/_on_start, specifically so _on_end
            # can persist it even though it has no state object of its own to read here.
            self._on_end(obj_id, frame_time)

        self._previous_ids_by_camera[camera.name] = current_ids

    def _on_start(
        self,
        camera: CameraConfig,
        obj_id: str,
        state: TrackedObjectState,
        frame_time: float,
        frame_jpeg: bytes | None = None,
        tracked_objects: dict[str, TrackedObjectState] | None = None,
    ) -> None:
        event_id = _event_id(frame_time)
        # The saved file is always the CLEAN frame -- boxes are stored as data
        # (snapshot_boxes below) and rendered on demand at request time (see
        # mirage.util.thumbnail's own module docstring for the full reasoning, and why
        # this replaced an earlier "burn the box in once, here" approach). frame_jpeg
        # (passed synchronously from CameraTracker's own process, see
        # mirage.tracking.camera_tracker._maybe_encode_frame) is preferred when
        # available -- it's the ACTUAL frame the closed-vocab detector just ran on, so
        # snapshot_boxes can correctly include EVERY Gate-1-confirmed object in that
        # frame, not just the one that triggered this Event (matches what a human
        # reviewing the snapshot expects to see -- confirmed by the user after noticing
        # only one of several visible people had a box in a real snapshot). Falls back
        # to the older live go2rtc fetch (capture_thumbnail) when no frame_jpeg is
        # available at all (e.g. an open-vocab synthetic track, produced by a
        # completely separate process with no shared frame -- gets a snapshot with no
        # boxes rather than a wrong one).
        if frame_jpeg is not None:
            snapshot_path = write_clean_snapshot(frame_jpeg, self.thumb_dir, event_id)
            snapshot_boxes = _confirmed_snapshot_boxes(tracked_objects if tracked_objects is not None else {obj_id: state})
        else:
            snapshot_path = capture_thumbnail(self.thumbnail_fetcher, self.thumb_dir, event_id, camera.name)
            snapshot_boxes = []

        # Mirage V3 async species classification: Person/Vehicle/etc. are never
        # enrichable (species_status stays "not_applicable" forever, matching the
        # design doc's "species classification only for detected animals"). Animal/Bird
        # events get "pending" IF a crop is actually dispatched below, else "skipped"
        # (no species_dispatcher configured, or no frame_jpeg to crop from -- e.g. the
        # capture_thumbnail live-fetch fallback path has no synchronously-available
        # frame to crop). Detection/tracking/recording/alerting are never blocked on
        # any of this -- the Event row above is already fully created and reviewable.
        species_enrichable = state.label in SPECIES_ENRICHABLE_LABELS
        species_status = "not_applicable"
        if species_enrichable:
            species_status = "skipped"
            if self.species_dispatcher is not None and frame_jpeg is not None:
                crop_jpeg = crop_jpeg_to_box(frame_jpeg, state.box)
                if crop_jpeg is not None:
                    self.species_dispatcher.dispatch(event_id, camera.name, state.label, crop_jpeg)
                    species_status = "pending"
                    self._log("species", f"{state.label} image queued for species classification", camera.name)

        self._active[obj_id] = {
            "event_id": event_id,
            "last_published": frame_time,
            "top_score": state.score,
            "false_positive": state.is_false_positive,
            # Fixed once, here, at creation time, and re-included VERBATIM on every
            # later Event.update() call in _on_update -- never recomputed from a
            # LATER frame's box position. The snapshot image itself is captured only
            # once (right above) and never changes again, so re-deriving
            # snapshot_boxes from a later frame would reintroduce the exact
            # box/pixel-mismatch bug the whole render-on-request design fixes (a
            # moving object's later box position would no longer match where it
            # actually was in this frozen image).
            "snapshot_boxes": snapshot_boxes,
        }
        Event.create(
            id=event_id,
            label=state.label,
            camera=camera.name,
            start_time=_to_datetime(frame_time),
            end_time=None,
            score=state.score,
            top_score=state.score,
            false_positive=state.is_false_positive,
            has_snapshot=snapshot_path is not None,
            snapshot_path=snapshot_path,
            data={
                "box": list(state.box),
                "snapshot_boxes": snapshot_boxes,
                "species": None,
                "species_status": species_status,
                "species_confidence": None,
                "species_taxonomy": None,
                "species_model": None,
            },
        )
        logger.debug("%s: event %s started for %s", camera.name, event_id, state.label)
        self._log("detect", f"{state.label} detected", camera.name)
        self._notify("event", event_id, "create")

    def _on_update(self, camera: CameraConfig, obj_id: str, state: TrackedObjectState, frame_time: float) -> None:
        entry = self._active.get(obj_id)
        if entry is None:
            return  # shouldn't happen, but don't crash on an inconsistent state

        previous_top_score = entry["top_score"]
        top_score = max(previous_top_score, state.score)

        elapsed = frame_time - entry["last_published"]
        should_update_db = self._should_update_db(previous_top_score, top_score, elapsed)
        # Always keep these current on `entry`, whether or not we write to the DB this
        # call -- `entry` is what `_on_end` reads to persist the FINAL status, since by
        # the time an object is removed there's no `state` for it to read from anymore
        # (see `process`'s comment on `removed_ids`). Without this, a false_positive ->
        # true_positive flip that happens on a throttled-out frame right before the
        # object disappears would never reach the database at all -- this was a real
        # bug: a person clearly and repeatedly detected (top_score up to 0.92) stayed
        # false_positive=True forever, because Event.update() only ever ran on this
        # throttled schedule and _on_end never wrote false_positive itself.
        entry["top_score"] = top_score
        entry["false_positive"] = state.is_false_positive
        if not should_update_db:
            return

        entry["last_published"] = frame_time

        # Event.update(data=...) REPLACES the whole JSON field -- writing {"box": ...,
        # "snapshot_boxes": ...} alone would silently wipe out any OTHER key already in
        # `data`. This used to only matter for snapshot_boxes (real bug caught live: a
        # fresh Event's snapshot_boxes was present at creation, then gone by the time
        # the object's track ended). Mirage V3 makes this a live hazard for a second,
        # more subtle reason: the (optional) SpeciesDispatcher can write species/
        # species_status/species_confidence/species_taxonomy into this SAME row, at ANY
        # time, from a completely different call site (mirage.species.dispatcher,
        # draining results in the main-process result-consumer loop) -- fully
        # asynchronously with respect to this throttled update. `entry` only ever holds
        # a stale snapshot of species fields from _on_start time (never updated after),
        # so re-including entry's copy here (the way snapshot_boxes does) would
        # RE-CLOBBER a genuine just-completed species classification back to
        # "pending"/None. The only correct fix is read-modify-write off the CURRENT row
        # instead of a blind literal -- read whatever is in the DB right now, overlay
        # just the fields this call owns (box/snapshot_boxes), and write the merged
        # result back, leaving any species_* keys exactly as the species dispatcher
        # last left them.
        current = Event.get_or_none(Event.id == entry["event_id"])
        current_data = dict(current.data) if current is not None and current.data else {}
        current_data["box"] = list(state.box)
        current_data["snapshot_boxes"] = entry["snapshot_boxes"]
        Event.update(
            score=state.score,
            top_score=top_score,
            false_positive=state.is_false_positive,
            data=current_data,
        ).where(Event.id == entry["event_id"]).execute()

    def _should_update_db(self, previous_top_score: float, top_score: float, elapsed: float) -> bool:
        """Section 6.3: throttle DB writes to at most once per
        UPDATE_PUBLISH_THROTTLE_SECONDS, except force at least a heartbeat update once
        per HEARTBEAT_FORCE_SECONDS, or immediately if top_score just increased.
        """
        if top_score > previous_top_score:
            return True
        if elapsed >= HEARTBEAT_FORCE_SECONDS:
            return True
        return elapsed >= UPDATE_PUBLISH_THROTTLE_SECONDS

    def _on_end(self, obj_id: str, frame_time: float) -> None:
        entry = self._active.pop(obj_id, None)
        if entry is None:
            return
        # Write the final false_positive/top_score here too, not just end_time -- the
        # last _on_update call for this object may have been throttled out (see
        # _should_update_db) even though `entry` itself was kept current, so this is the
        # last chance to persist a true final status before the row is done being
        # touched at all.
        Event.update(
            end_time=_to_datetime(frame_time),
            top_score=entry["top_score"],
            false_positive=entry["false_positive"],
        ).where(Event.id == entry["event_id"]).execute()
        logger.debug("event %s ended", entry["event_id"])

    def close_dangling_events(self) -> None:
        """Section 10.2: on startup, any Event rows left with end_time IS NULL (from an
        unclean previous shutdown) should be force-closed.
        """
        now = utcnow()
        dangling = Event.select().where(Event.end_time.is_null())
        for event in dangling:
            event.end_time = max(event.start_time + datetime.timedelta(seconds=30), now)
            event.save()
