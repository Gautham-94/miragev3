"""RulesEngine: camera-level derived-condition alerts -- crowd count and dwell-time/
loitering (TODO_FIX_LIST.md item 7.4's "rules engine" idea, scoped down to these two
rule types after discussion with the user; a third -- weapon detection -- was
deliberately deferred, since it has no existing detection capability to build on and
needs a real accuracy check against the open-vocab OWLv2 path before being shipped as
an alert type).

Distinct from mirage.events.review.classify_severity, which classifies alert/detection
severity PER OBJECT LABEL: a rule condition is computed from the CURRENT SET of tracked
objects (a count, a duration), not any single object's label -- there is no "the label
that means crowd." Lives alongside mirage.openvocab.dispatcher.OpenVocabDispatcher as a
second producer of SYNTHETIC TrackedObjectState entries fed into the same
dict[str, TrackedObjectState] shape EventProcessor/ReviewSegmentMaintainer already
consume from the closed-vocab pipeline (see OpenVocabDispatcher's own module docstring
for the pattern this follows) -- called once per camera per frame from
mirage.app._result_consumer_loop, merged into that frame's tracked_objects BEFORE
EventProcessor/ReviewSegmentMaintainer run, so a rule firing drives a real
ReviewSegment (severity="rule", see mirage.events.review.Severity) through the exact
same start/end/cutoff diff logic a closed-vocab detection does, rather than being a
separate, disconnected alerting path.

Every synthetic object id this module produces is prefixed with RULE_OBJECT_ID_PREFIX
-- mirage.events.review.classify_severity/qualifies_for_review specifically check for
this prefix (not the label, which is dynamic human-readable text like "crowd (12)") to
recognize a rule trigger and treat it as Severity.rule, exempt from the stationary-
object exclusion that normally applies to review activity (dwell-time/loitering is
specifically about objects that may not be moving much).
"""

from __future__ import annotations

from dataclasses import dataclass

from mirage.config.schema import CameraConfig
from mirage.tracking.stationary import StationaryClassifier
from mirage.tracking.tracker import TrackedObjectState

RULE_OBJECT_ID_PREFIX = "rule-"
_CROWD_OBJECT_ID_PREFIX = f"{RULE_OBJECT_ID_PREFIX}crowd-"
_DWELL_OBJECT_ID_PREFIX = f"{RULE_OBJECT_ID_PREFIX}dwell-"

# The only closed-vocab label crowd counting considers -- matches ReviewConfig's own
# default alert label ("person"), and keeps the rule's meaning unambiguous (a crowd of
# cars, or a mix of every tracked label, isn't what "crowd" means in the ask this was
# built for). Not user-configurable yet -- RulesConfig.crowd_threshold is the only
# knob; revisit if a camera genuinely needs to count a different label as "occupants."
CROWD_COUNTED_LABEL = "person"


@dataclass
class _DwellStart:
    first_seen: float
    last_seen: float


# How long an object id can be MISSING from tracked_objects before its dwell-start
# bookkeeping is forgotten -- deliberately generous relative to norfair's own
# max_disappeared coasting window (mirage.tracking.config.max_disappeared = fps * 5,
# i.e. up to 5s of tolerance for a genuinely momentary miss before norfair itself
# considers the track gone). This exists because of a REAL bug found in production: the
# very first version of this pruning logic deleted an id's dwell-start the instant it
# was absent from even ONE frame's tracked_objects dict -- but ObjectTracker.update()
# (mirage/tracking/tracker.py) removes an id from its returned dict for any frame where
# it isn't re-confirmed in `active_ids_this_frame`, even while norfair's own
# hit_counter/coasting still considers it the SAME ONGOING TRACK (it can reappear next
# frame with the same id). A live person tracked continuously for 1300+ real seconds
# only triggered a 120s dwell_seconds threshold after ~330s of ACCUMULATED time,
# because the dwell clock kept getting silently reset by these single-frame flickers --
# confirmed directly against real pipeline logs (a continuous 1320s Event, but
# "loitering (331s)" as the first triggered label, nowhere near the configured 120s).
# A generous grace period, rather than trusting per-frame presence, is what actually
# matches what "the same track" means everywhere else in this codebase.
_DWELL_START_GRACE_SECONDS = 15.0


class RulesEngine:
    """One instance per MirageApp (not per camera) -- all per-camera state is keyed by
    camera name internally, same pattern OpenVocabDispatcher already uses.
    """

    def __init__(self) -> None:
        # obj_id -> (first frame_time this id was seen, most recent frame_time it was
        # seen) with a qualifying (non-false-positive) state -- deliberately
        # INDEPENDENT bookkeeping from EventProcessor._active's own start-time
        # tracking (that's private internal state of a different module); the two
        # will be within a fraction of a second of each other in practice, since both
        # first observe an object on the same frame. See _DWELL_START_GRACE_SECONDS
        # for why `last_seen` exists at all (it isn't just `first_seen` -- a real,
        # confirmed bug fix).
        self._dwell_starts: dict[str, _DwellStart] = {}

    def process(
        self, camera: CameraConfig, frame_time: float, tracked_objects: dict[str, TrackedObjectState],
    ) -> dict[str, TrackedObjectState]:
        """Returns synthetic rule-trigger entries for THIS frame only -- unlike
        OpenVocabDispatcher.synthetic_tracked_objects (which has its own multi-frame
        TTL-based liveness state, since OWLv2 matches arrive asynchronously seconds
        apart), a rule condition is fully re-evaluated fresh every single frame from
        real, already-current tracked_objects: if the condition no longer holds this
        frame, the synthetic entry simply isn't in the returned dict, and
        ReviewSegmentMaintainer's existing cutoff-timer logic (same as any other
        object disappearing) closes out the segment after review.cutoff_seconds of no
        further activity -- no separate expiry mechanism needed here.
        """
        synthetic: dict[str, TrackedObjectState] = {}

        crowd_entry = self._evaluate_crowd(camera, frame_time, tracked_objects)
        if crowd_entry is not None:
            synthetic[crowd_entry.id] = crowd_entry

        synthetic.update(self._evaluate_dwell(camera, frame_time, tracked_objects))

        self._prune_dwell_starts(frame_time)
        return synthetic

    def _evaluate_crowd(
        self, camera: CameraConfig, frame_time: float, tracked_objects: dict[str, TrackedObjectState],
    ) -> TrackedObjectState | None:
        threshold = camera.rules.crowd_threshold
        if not threshold:
            return None

        count = sum(
            1 for state in tracked_objects.values()
            if not state.is_false_positive and state.label == CROWD_COUNTED_LABEL
        )
        if count < threshold:
            return None

        return TrackedObjectState(
            id=f"{_CROWD_OBJECT_ID_PREFIX}{camera.name}",
            label=f"crowd ({count})",
            # No single box represents "the whole crowd" -- the full frame is the
            # most honest placeholder; nothing currently renders a box for a
            # rule-severity ReviewSegment's thumbnail (review boxes are disabled
            # entirely per an earlier user request, same as every other synthetic
            # track -- see OpenVocabDispatcher.SyntheticTrack's own note on this).
            box=(0.0, 0.0, float(camera.detect.width), float(camera.detect.height)),
            score=1.0,
            stationary=StationaryClassifier(threshold_frames=10**9),
            frame_time=frame_time,
            is_false_positive=False,
        )

    def _evaluate_dwell(
        self, camera: CameraConfig, frame_time: float, tracked_objects: dict[str, TrackedObjectState],
    ) -> dict[str, TrackedObjectState]:
        threshold = camera.rules.dwell_seconds
        if not threshold:
            return {}

        triggered: dict[str, TrackedObjectState] = {}
        for obj_id, state in tracked_objects.items():
            if state.is_false_positive:
                continue

            start = self._dwell_starts.get(obj_id)
            if start is None:
                self._dwell_starts[obj_id] = _DwellStart(first_seen=frame_time, last_seen=frame_time)
                continue

            start.last_seen = frame_time
            elapsed = frame_time - start.first_seen
            if elapsed < threshold:
                continue

            dwell_id = f"{_DWELL_OBJECT_ID_PREFIX}{obj_id}"
            triggered[dwell_id] = TrackedObjectState(
                id=dwell_id,
                label=f"loitering ({int(elapsed)}s)",
                box=state.box,
                score=state.score,
                stationary=StationaryClassifier(threshold_frames=10**9),
                frame_time=frame_time,
                is_false_positive=False,
            )
        return triggered

    def _prune_dwell_starts(self, frame_time: float) -> None:
        """Forgets an object id's dwell-start bookkeeping only once it's been ABSENT
        for longer than _DWELL_START_GRACE_SECONDS -- NOT the instant it's missing
        from a single frame's tracked_objects (see _DWELL_START_GRACE_SECONDS's own
        docstring for the real bug this fixes: ObjectTracker can drop an id from its
        returned dict for one frame while norfair still considers it the SAME ongoing
        track, and pruning on that single-frame absence silently reset the dwell
        clock over and over, making a 120s threshold take 1000+ real seconds to
        actually fire). Still bounded -- an id that's genuinely gone for good (the
        real per-frame absence the OLD logic was trying to detect) is cleaned up
        within one grace window of its track actually ending, same order of
        magnitude as the leak this replaces (OPTIMIZATION_OPPORTUNITIES.md item 2 /
        TODO_FIX_LIST.md's CameraOrchestrator._lifecycles fix), just not instantaneous.
        """
        stale = [
            obj_id for obj_id, start in self._dwell_starts.items()
            if frame_time - start.last_seen > _DWELL_START_GRACE_SECONDS
        ]
        for obj_id in stale:
            del self._dwell_starts[obj_id]
