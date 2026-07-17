# Optimization opportunities

A code-grounded audit of shortcomings in the current architecture beyond what's already
tracked in `TODO_FIX_LIST.md` and `DETECTOR_SCALING.md` (single detector process,
frame-drop-on-full, no config hot-reload, no rules engine, no INT8/batching, static
frame rate, no notification service — see those files for those items). Every finding
below is grounded in an actual code path, not a hypothetical.

Ordered roughly by severity — items 1-3 are the ones most likely to actually bite you
on a long-running deployment; the rest are real but smaller.

---

## 1. No retention sweep ever deletes old `Recordings`/`Event`/`ReviewSegment`/`QueryMatch` rows or files — the DB and disk grow forever

**Files:** `mirage/recording/maintainer.py`, `mirage/recording/retention.py`,
`mirage/db/models.py`, `mirage/events/processor.py`, `mirage/events/review.py`,
`mirage/openvocab/dispatcher.py`

`should_retain_by_base_policy()` (`retention.py`) is only ever called once, at the
moment a *new* cache segment is about to be promoted to permanent storage
(`RecordingMaintainer._process_segment`, `maintainer.py`) — it decides whether the new
segment should be kept at all. Nothing anywhere re-evaluates already-promoted
recordings once they age past `retain_days`. Confirmed via an exhaustive grep for
`.delete()` / `DELETE FROM` / `os.remove` / `unlink()` across `mirage/`: the only file
deletions are cache-segment force-deletes under backpressure and the raw segment file
after a successful remux — neither ever touches the permanent `record_dir` tree or the
`Recordings` table. `Recordings` is only ever `.create()`d or `.select()`ed, never
`.delete()`d, anywhere in the codebase.

The exact same gap applies to `Event` (`EventProcessor._on_start`/`Event.create`),
`ReviewSegment`, and `QueryMatch` (`OpenVocabDispatcher._drain_results`, which also
writes a thumbnail file to disk on every real match) — every one of these tables (and
their associated snapshot/thumbnail directories) only ever grows.

**Impact:** correctness-adjacent + real resource cost. Configuring `retain_days` only
controls what gets promoted *going forward* — it silently does nothing to expire
anything already on disk or in the DB. Over weeks/months this means unbounded disk
usage (video files, thumbnails) and an unbounded SQLite table, which in turn degrades
every list query over time (see item 12). This is the single highest-impact gap found —
worse than anything already tracked, since "set a retention policy" is a reasonable
user expectation this silently doesn't fulfill.

---

## 2. [FIXED] `CameraOrchestrator._lifecycles` grows unbounded for the life of each camera-tracker process

**Status:** fixed. `_update_lifecycles` now deletes a closed-out track's entry outright
instead of just marking `end_time` and leaving it in the dict forever — object ids are
never reused (`tracker.py`'s `_new_id` is timestamp-based), so once a track ends
nothing can reference that id again, and nothing reads a lifecycle after its track ends
anyway (`get_lifecycle()` has no remaining callers; the only thing that ever left this
process was `is_false_positive`, mirrored onto `TrackedObjectState` while the track was
still live). Two regression tests added in `tests/test_orchestration.py`:
`test_lifecycle_entries_are_evicted_once_a_track_ends` and
`test_lifecycles_dict_does_not_grow_across_many_distinct_short_lived_objects` (500
simulated short-lived objects, asserts the dict returns to empty). Full suite: 418
passed.

**File:** `mirage/tracking/orchestration.py`

```python
self._lifecycles: dict[str, ObjectLifecycle] = {}
...
def _update_lifecycles(self, tracked, frame_time):
    for obj_id, state in tracked.items():
        lifecycle = self._lifecycles.get(obj_id)
        if lifecycle is None:
            lifecycle = ObjectLifecycle(label=state.label, start_time=frame_time)
            self._lifecycles[obj_id] = lifecycle
        ...
    for obj_id in list(self._lifecycles.keys()):
        if obj_id not in tracked:
            self._lifecycles[obj_id].end_time = frame_time
```

When a tracked object's track ends, its `ObjectLifecycle` gets `end_time` set — but is
never removed from `self._lifecycles`. Every distinct object id a camera has ever seen
(a fresh id per track, per `tracker.py`'s `_new_id`) accumulates a permanent dict entry
for as long as that camera's `CameraTracker` process runs.

**Impact:** genuine unbounded memory growth in a long-running per-camera process that
isn't expected to restart on its own. On a busy camera (continuous foot/vehicle
traffic) this grows continuously over days/weeks of uptime. A real leak, not
hypothetical.

---

## 3. [FIXED] `is_open_by_ffmpeg` scans every process on the host, every 5 seconds, per cache segment

**Status:** fixed. The mechanism itself (enumerating ffmpeg processes' open files) is
still necessary -- confirmed directly that ffmpeg takes no flock/fcntl lock on its
output file, so a lock-based check would silently never detect an in-progress segment,
reintroducing the exact bug this check exists to prevent. The actual waste was
re-running that whole-host scan once PER PENDING SEGMENT within the same maintainer
cycle -- split into `ffmpeg_open_file_paths()` (one scan, returns a `set[str]` of every
currently-open path) called ONCE per `RecordingMaintainer.run_once()`, with
`is_open_by_ffmpeg(path, open_paths)` now just a set-membership check against that
snapshot. New regression test in `tests/test_recording_maintainer.py`:
`test_run_once_scans_ffmpeg_open_files_only_once_per_cycle_not_per_segment` (4 pending
segments across 2 cameras, asserts exactly 1 scan). Full suite: 419 passed.

**File:** `mirage/recording/segments.py`

```python
def is_open_by_ffmpeg(path: Path) -> bool:
    resolved = str(path.resolve())
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] not in ("ffmpeg", "ffmpeg.exe"):
                continue
            for f in proc.open_files():
                if f.path == resolved:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False
```

Called from `RecordingMaintainer._process_segment` for every pending cache segment, on
every 5-second maintainer loop tick. `psutil.process_iter()` enumerates every process
on the host; `proc.open_files()` per ffmpeg process is an OS-level syscall (reads
`/proc/<pid>/fd/*` and resolves each symlink on Linux; similarly expensive on macOS) to
list every open file descriptor. With N cameras (N ffmpeg processes) and M pending
segments, this is O(processes × open_fds) work, repeated indefinitely every 5 seconds,
just to answer "is this one file open."

**Impact:** real, recurring CPU/syscall cost that scales with camera count and never
stops. A cheaper check (non-blocking exclusive lock attempt, or leaning on the
`probe_duration` moov-atom validity check the code already does as a secondary signal)
would avoid enumerating the whole process table.

---

## 4. `MirageConfig.from_db()` re-reads and re-validates the entire config tree on every API request that touches config

**Files:** `mirage/api/app.py`, `mirage/config/schema.py`, `mirage/config/store.py`,
`mirage/api/routers/cameras.py`, `mirage/api/routers/config.py`

```python
app.state.get_config = MirageConfig.from_db
```

Every router handler needing config calls `request.app.state.get_config()` — including
cheap read-only endpoints like `GET /api/cameras` (hit on every Live-page load) and
`GET /api/config/detectors`. `from_db()` does a full SQLite round trip
(`AppConfig.get_or_none`) followed by a full Pydantic `model_validate()` pass over the
*entire* nested config tree (every camera, detector, query) — on every single request,
even ones that only need a handful of fields.

**Impact:** since config only changes via the config-write endpoints (and even then
needs a pipeline restart to take effect), this is trivially cacheable with invalidation
on write. Today every read pays the full DB+deserialization cost regardless. Scales
with config size (camera/detector/query count) and API request volume — not
catastrophic yet, but a clear, easy, avoidable per-request cost.

---

## 5. [FIXED] Camera-tile snapshot polling never stops once a real video stream connects

**Status:** fixed. `refreshSnapshot()`'s interval subscription now starts/stops through
a single `setMode()` helper that every `mode()` transition goes through: entering
`'mse'`/`'webrtc'` stops the poll subscription outright (matching the template's own
gating -- the `<img class="poster">` only renders while `mode()` is NOT `'mse'`/
`'webrtc'`); falling back to `'connecting'`/`'poster'`/`'error'` restarts it, so a
stream that later fails still gets a live poster again.

**File:** `frontend/src/app/pages/live/camera-tile/camera-tile.ts`

```typescript
const SNAPSHOT_POLL_MS = 1000;
...
ngAfterViewInit(): void {
    interval(SNAPSHOT_POLL_MS)
      .pipe(startWith(0), takeUntilDestroyed(this.destroyRef))
      .subscribe(() => this.refreshSnapshot());
    this.startVideoTiers();
}
```

The code's own comment acknowledges this is deliberate ("keeps running continuously
regardless of video-tier state"), but once `mode()` becomes `'mse'` or `'webrtc'` (a
live, continuously-updating video stream is already rendering), the component keeps
firing a full HTTP GET + JPEG decode every second per tile, forever, for a snapshot
image that's no longer even displayed.

**Impact:** real, continuous, unconditional background HTTP traffic that scales
directly with camera count on the Live grid — competing with the actual video streams
and other polling (Events/Review/Recordings pages) for the same backend. Not cosmetic.

---

## 6. [PARTIALLY FIXED] Events/Review/Recordings pages fully re-fetch and re-render on every poll, no visibility check, no virtualization

**Status:** the tab-visibility half is fixed; virtualization/incremental-fetch are
still open (real, but separate, larger changes). New shared utility
`frontend/src/app/core/rxjs/visible-interval.ts` (`visibleInterval(ms)`) replaces
`interval(POLL_MS).pipe(startWith(0), ...)` on all three pages -- pauses entirely while
`document.visibilityState !== 'visible'`, resumes (restarting from tick 0) the instant
the tab becomes visible again. Getting this right took a real correction: an initial
version `filter()`ed the visibilitychange stream down to just "became visible" events
before `switchMap`-ing, which meant `switchMap` never received a new outer emission on
the "became hidden" transition -- so it never got a chance to unsubscribe the
still-running inner `interval`, and polling silently continued regardless (confirmed
with a real RxJS subscription trace: tick count kept growing well after
`visibilityState` was set to `'hidden'`). Fixed by having `switchMap` choose between a
fresh `interval(ms)` (visible) or `NEVER` (hidden) on EVERY visibilitychange event, so
`switchMap`'s own "unsubscribe the previous inner observable" behavior is what actually
tears down the ticking interval. Verified live in a real headless Chrome session
against the actual running Events page, tracing real `/api/events` network requests via
CDP's `Network.requestWillBeSent` across three phases: visible (4 requests/11s
baseline) → simulated hidden via `Object.defineProperty(document, 'visibilityState',
...)` + a dispatched `visibilitychange` event (0 requests/11s -- confirmed genuinely
paused) → visible again (1 immediate request, confirming instant resume from tick 0).
Full suite/build: `ng build` clean, no TS errors.

**Files:** `frontend/src/app/pages/events/events-page.ts`,
`frontend/src/app/pages/review/review-page.ts`,
`frontend/src/app/pages/recordings/recordings-page.ts`

All three poll on a fixed interval (`interval(POLL_MS).pipe(startWith(0), switchMap(...))`,
5-10s) and fully re-fetch the same top-N window (limit 100-200) every cycle — even when
the browser tab is backgrounded (no `document.visibilityState` check), and even though
nothing may have changed. No cursor/"since"-based incremental fetch, no
`cdk-virtual-scroll-viewport` (confirmed absent from all three templates) — up to 200
DOM rows get diffed/re-rendered every poll.

**Impact:** modest but real, bounded network/DB cost that scales with the number of
open browser tabs. A push-based (WebSocket) or conditional (ETag / "since" cursor)
approach would avoid the redundant full re-fetch.

---

## 7. [FIXED] `motion/detector.py` uses `scipy.ndimage.gaussian_filter` instead of `cv2.GaussianBlur` in the unconditional per-frame hot path

**Status:** fixed, with real measurement before committing to it. Directly compared
`scipy.ndimage.gaussian_filter(sigma=1, radius=blur_radius)` against
`cv2.GaussianBlur((2*radius+1, 2*radius+1), sigmaX=1, sigmaY=1)` on a real random test
frame: cv2's DEFAULT border mode (`BORDER_REFLECT_101`) diverges from scipy's default
`'reflect'` edge handling by up to 61/255 at the frame edges -- a real behavior change,
not noise -- but `cv2.BORDER_REFLECT` matches scipy's convention far more closely (max
diff 2/255 everywhere, ordinary `uint8` rounding). With that border mode, measured
~7.15x faster (0.109ms vs 0.015ms/call at a realistic 108x192 motion-detection frame
size, 2000-iteration benchmark). `scipy` import removed from this file entirely (was
its only non-cv2 call in an otherwise all-cv2 hot path). All 10 existing
`tests/test_motion.py` tests re-verified passing unchanged, confirming no observable
motion-detection behavior change.

**File:** `mirage/motion/detector.py`

```python
from scipy.ndimage import gaussian_filter
...
resized_frame = gaussian_filter(resized_frame, sigma=1, radius=self.blur_radius)
```

Runs unconditionally on every single frame, from every camera with motion detection
enabled (`MotionDetector.detect()` is called unconditionally every frame per
`CameraOrchestrator.process_frame`). Every other operation in the same function is a
`cv2.*` call (`resize`, `absdiff`, `threshold`, `dilate`, `findContours`,
`accumulateWeighted`) — mixing in one scipy call adds an array format/dtype
negotiation plus a generally slower pure-numpy-backed blur compared to OpenCV's
SIMD/multi-threaded C++ `GaussianBlur`.

**Impact:** modest but real per-frame CPU overhead, directly in the most
latency-sensitive stage of the pipeline, multiplied by every camera/frame. No
algorithmic reason found for preferring scipy here (unlike the deliberate,
commented-on avoidance of a heavier dependency elsewhere in the codebase, e.g.
`openvocab/gating.py`'s from-scratch perceptual hash instead of `imagehash`).

---

## 8. `StationaryClassifier.update()` computes a second `trimmed_box()` that's never read

**File:** `mirage/tracking/stationary.py`

```python
self._previous_trimmed_box: Box | None = None
...
def update(self, box: Box) -> None:
    current_trimmed = self.trimmed_box()          # 4x np.percentile
    self._xmins.append(x1); ...
    if current_trimmed is None:
        self._previous_trimmed_box = self.trimmed_box()
        return
    ...
    self._previous_trimmed_box = self.trimmed_box()   # 4x np.percentile -- never read
```

`_previous_trimmed_box` is written in three places but never read anywhere in the class
or the rest of the codebase (confirmed by grep) — it's dead state. Each `trimmed_box()`
call does 4 separate `np.percentile()` calls. `update()` runs once per tracked object,
per frame (`ObjectTracker.update()` → `state.stationary.update(box)`), so every
confirmed+candidate object, every frame, pays for a second full trimmed-box computation
purely to populate a field nothing reads.

**Impact:** pure waste — literally dead computation, ~50% of this function's
percentile cost is thrown away. Easy to verify, easy to remove.

---

## 9. `OpenVocabDispatcher._pending` and `OpenVocabGate._in_flight` have no timeout — a lost result leaks state and can permanently starve a gate key

**Files:** `mirage/openvocab/dispatcher.py`, `mirage/openvocab/gating.py`

```python
self._pending: dict[str, tuple[str, str, float]] = {}
...
self._pending[request_id] = (camera.name, object_id, frame_time)
...
pending = self._pending.pop(result.request_id, None)   # only removed on a result arriving
```

Every dispatched OWLv2 request adds a `_pending` entry, removed only when its
`OpenVocabResult` is drained. If `OpenVocabProcess` crashes/restarts or a request is
otherwise lost, that entry is never cleaned up — no timeout-based sweep exists. Since
`OpenVocabGate._in_flight` is also only cleared by `mark_result_received()` (called
from the same drain path), a lost result doesn't just leak a small dict entry — it
permanently blocks that object/camera gate key from ever being re-checked again for the
life of the process.

**Impact:** slow leak plus a correctness-adjacent starvation bug. Small in isolation,
but compounds with the already-tracked "no restart-recovery for OpenVocabProcess" gap —
a single lost request silently and permanently disables open-vocab checking for that
object/camera until the whole pipeline restarts.

---

## 10. Open-vocab match draining does synchronous disk + DB writes inline in the single shared result-consumer loop

**File:** `mirage/openvocab/dispatcher.py`

```python
def _drain_results(self) -> None:
    while True:
        ...
        for match in result.matches:
            match_id = f"{time.time()}-{uuid.uuid4().hex[:6]}"
            thumb_path = self._save_thumb(result, match_id)   # blocking disk write
            QueryMatch.create(...)                             # blocking DB write
```

`process_frame()` calls `_drain_results()` unconditionally at the top, for every
camera, every frame, inside the main process's single-threaded `_result_consumer_loop`
— the same loop that drains every camera's `detected_frames_queue` results. A blocking
thumbnail write plus a blocking SQLite write here directly delays processing of every
*other* camera's queued results, since this loop is shared and single-threaded.

**Impact:** real, not hypothetical — worse the more simultaneous OWLv2 matches occur
across cameras, since the I/O cost of draining one camera's matches adds latency to the
whole pipeline's result consumption, not just that camera's.

---

## 11. [FIXED] `Event.false_positive` has no index despite being the default filter on the most common Events query

**Status:** fixed -- `index=True` added to the field. Confirmed (not assumed) this
actually creates the index retroactively on an already-existing table, not just fresh
ones: `init_database`'s `create_tables(ALL_MODELS, safe=True)` issues idempotent
`CREATE INDEX IF NOT EXISTS` DDL on every startup regardless of whether the table
predates the model change -- verified directly by creating a DB with the old schema,
then re-running `init_database` after adding `index=True` and confirming
`event_false_positive` appears in `sqlite_master`'s index list with no migration
script needed.

**Files:** `mirage/db/models.py`, `mirage/api/routers/events.py`

```python
false_positive = BooleanField(default=True)   # no index=True
...
if not include_false_positive:  # False is the default, and the frontend never sets it True
    query = query.where(Event.false_positive == False)
query = query.order_by(Event.start_time.desc()).limit(limit).offset(offset)
```

`camera`, `label`, `start_time`, and `end_time` are all indexed on `Event`;
`false_positive` — applied on essentially every `GET /api/events` call (the frontend
polls this every 5s, never opting into `include_false_positive=true`) — is not.

**Impact:** modest today, but compounds directly with item 1 (the table never shrinks)
— as row count grows unbounded, this filter increasingly can't be satisfied by an index
alone. A cheap, easy fix that becomes more valuable the longer the system runs
uninterrupted.

---

## Suggested priority if picked up

1. **Retention sweep (item 1)** — highest impact, and the current behavior silently
   contradicts a real user-facing config setting (`retain_days`). Should be a dedicated
   periodic job (mirroring the existing `RecordingMaintainer` loop pattern) that deletes
   expired `Recordings` rows + files, and equivalent sweeps for `Event`/`ReviewSegment`/
   `QueryMatch` + their thumbnail files.
2. **`CameraOrchestrator._lifecycles` leak (item 2)** — small, contained fix (evict on
   the same `_update_lifecycles` pass instead of just marking `end_time`).
3. **`is_open_by_ffmpeg` (item 3)** and **config caching (item 4)** — both cheap,
   isolated fixes with real recurring-cost payoff.
4. Everything else is real but lower urgency — worth batching into a cleanup pass
   rather than fixing individually.
