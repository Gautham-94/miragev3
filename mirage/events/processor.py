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
from mirage.tracking.tracker import TrackedObjectState
from mirage.util.thumbnail import ThumbnailFetcher, capture_thumbnail
from mirage.util.time import utc_from_timestamp, utcnow

logger = logging.getLogger(__name__)

UPDATE_PUBLISH_THROTTLE_SECONDS = 5.0
HEARTBEAT_FORCE_SECONDS = 60.0


def _event_id(start_time: float) -> str:
    rand6 = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{start_time}-{rand6}"


def _to_datetime(ts: float) -> datetime.datetime:
    return utc_from_timestamp(ts)


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

    def process(
        self,
        camera: CameraConfig,
        frame_time: float,
        tracked_objects: dict[str, TrackedObjectState],
    ) -> None:
        previous_ids = self._previous_ids_by_camera.get(camera.name, set())
        current_ids = set(tracked_objects.keys())

        new_ids = current_ids - previous_ids
        updated_ids = current_ids & previous_ids
        removed_ids = previous_ids - current_ids

        for obj_id in new_ids:
            self._on_start(camera, obj_id, tracked_objects[obj_id], frame_time)

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

    def _on_start(self, camera: CameraConfig, obj_id: str, state: TrackedObjectState, frame_time: float) -> None:
        event_id = _event_id(frame_time)
        self._active[obj_id] = {
            "event_id": event_id,
            "last_published": frame_time,
            "top_score": state.score,
            "false_positive": state.is_false_positive,
        }
        snapshot_path = capture_thumbnail(self.thumbnail_fetcher, self.thumb_dir, event_id, camera.name)
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
            data={"box": list(state.box)},
        )
        logger.debug("%s: event %s started for %s", camera.name, event_id, state.label)

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
        Event.update(
            score=state.score,
            top_score=top_score,
            false_positive=state.is_false_positive,
            data={"box": list(state.box)},
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
