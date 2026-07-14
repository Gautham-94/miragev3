# Running mirage locally

This is the practical "how do I start it and see something" guide. For architecture and
implementation details, see `NVR_PIPELINE_IMPLEMENTATION_SPEC.md`,
`NVR_EXTENSION_MULTI_MODEL_ROUTING.md`, and `IMPLEMENTATION_NOTES.md`.

You'll run **four processes**, each in its own terminal:

1. A test video source (stands in for a real camera)
2. `python -m mirage` — the capture/detect/track/record pipeline + go2rtc (live view)
3. `python -m mirage.api` — the read API the frontend talks to
4. The Angular frontend dev server

## 0. One-time setup

```
cd /Users/gauthamkrishna/Documents/Projects/nvr/frigate_nvr/mirage
python3 -m venv .venv        # if .venv doesn't already exist
source .venv/bin/activate
pip install -r requirements.txt
```

```
cd frontend
npm install
```

You need `ffmpeg` on your PATH (`brew install ffmpeg` on macOS). The first time
`python -m mirage` runs, it auto-downloads go2rtc to `mirage/bin/` — no manual setup
needed there.

### Optional: use a real video instead of a test pattern

Drop any `.mp4` at:

```
mirage/test_video_source/test.mp4
```

If present, the test stream loops it (so the detector has real objects to find). If
absent, it falls back to a synthetic moving-shapes pattern, which works for testing the
pipeline mechanics but never triggers real object detections.

## 1. Terminal 1 — start the test video source

```
cd /Users/gauthamkrishna/Documents/Projects/nvr/frigate_nvr/mirage
./scripts/run_test_stream.sh
```

Leave this running. It serves the test video on two ports (19500 for mirage's own
capture, 19501 for go2rtc's live-view restream — real cameras don't need this split,
it's only because the synthetic source can't serve two clients on one port). Stop with
Ctrl+C when you're done with everything (or `pkill -9 -f run_test_stream; pkill -9 -f 19500; pkill -9 -f 19501` if Ctrl+C doesn't stick — this can happen since it's a retry-loop).

## 2. Terminal 2 — start the mirage pipeline

Config now lives in the database (`config/mirage.db`), not in `config/mirage.yaml` —
`--config` is only a **one-time import**: the very first time you run against a fresh
database with no config in it yet, pass `--config` to seed the DB from that YAML file.
Every run after that reads from the DB and ignores `--config` entirely, so you can
(and should) drop it from the command once the DB has been seeded once.

First run (seeds the DB from `config/mirage.yaml`):

```
cd /Users/gauthamkrishna/Documents/Projects/nvr/frigate_nvr/mirage
source .venv/bin/activate
python3 -m mirage --config config/mirage.yaml --go2rtc-source "test_cam=tcp://127.0.0.1:19501" -v
```

Every run after that (DB already has config, no YAML needed at all):

```
python3 -m mirage --go2rtc-source "test_cam=tcp://127.0.0.1:19501" -v
```

This starts capture, motion detection, ONNX object detection, tracking, recording, and
go2rtc (for live view). `-v` gives debug-level logs; drop it for quieter output. The
`--go2rtc-source` flag is only needed for this synthetic dual-port test setup — with a
real camera you'd omit it entirely (see "Using a real camera" below).

Stop with Ctrl+C — shutdown is graceful (tears down all subprocesses, closes dangling DB
rows) and normally takes a few seconds.

**Adding/editing cameras after that first run**: don't re-edit `config/mirage.yaml` and
expect it to do anything — it's only read on that first, DB-empty run. Add cameras via
the wizard/config API instead (see below), then restart `python -m mirage` to pick up
the change (config changes aren't hot-reloaded into an already-running pipeline yet).

## 3. Terminal 3 — start the API

```
cd /Users/gauthamkrishna/Documents/Projects/nvr/frigate_nvr/mirage
source .venv/bin/activate
python3 -m mirage.api --db-path config/mirage.db --port 8000 --cors-origin "http://localhost:4200"
```

This reads the same SQLite DB the pipeline (terminal 2) is writing to, and always reads
config **fresh from that DB** on every request — so it immediately reflects any camera
you add via the wizard, with no restart of the API itself needed (only the pipeline in
terminal 2 needs a restart to actually start capturing the new camera). It's a separate,
independent process. `--cors-origin` needs to match wherever the frontend dev server runs.

Sanity check it's working:

```
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/cameras
```

Auto-generated API docs (Swagger UI): **http://127.0.0.1:8000/docs**

## 4. Terminal 4 — start the frontend

```
cd /Users/gauthamkrishna/Documents/Projects/nvr/frigate_nvr/mirage/frontend
npm start
```

Then open **http://localhost:4200** in a browser.

## Where to see results

| Page | URL | What it shows |
|---|---|---|
| Live | `/live` | Camera tiles with real live video (MSE/WebRTC), falling back to a polled JPEG snapshot if the video tiers can't connect |
| Review | `/review` | Alert/detection activity feed with thumbnails, filterable by camera/severity, click a card for details |
| Events | `/events` | Every individual tracked-object detection, filterable by camera/label/time range, with a small thumbnail per row |
| Recordings | `/recordings` | Stored 10-second clips grouped by date, filterable by camera, hover a row for an inline video preview |

You can also inspect the database directly:

```
sqlite3 config/mirage.db "select camera, label, top_score, false_positive from event order by rowid desc limit 10;"
sqlite3 config/mirage.db "select camera, severity, start_time from reviewsegment order by rowid desc limit 10;"
sqlite3 config/mirage.db "select camera, start_time, duration, segment_size_mb from recordings order by rowid desc limit 10;"
```

Recorded clips land in `media/recordings/YYYY-MM-DD/HH/<camera>/MM.SS.mp4`. Captured
thumbnails land in `media/clips/events/` and `media/clips/review/`.

## What to expect with the shipped test config

`config/mirage.yaml` tracks `person, car, bus, dog, cat`. If you're using the fallback
synthetic pattern (no `test_video_source/test.mp4`), you'll see recordings accumulate but
**no** events/review activity — there's nothing recognizable in a moving-shapes test
pattern. Drop in a real video with people/vehicles/animals in it to see the detection
side actually populate. Real object detections can take a few minutes of runtime to
appear (confidence has to clear a threshold, and review segments specifically only
happen for labels listed under `review.alerts`/`review.detections` in the camera config —
by default just `person`/`car` for alerts, nothing for detections).

## Adding a camera (the normal way, once mirage/mirage.api are already running)

Once you've got the first-run seed done (see below) and `mirage.api` + the frontend are
running, adding a camera doesn't need any YAML editing at all:

1. Open **http://localhost:4200/add-camera** (or click the dashed "Add camera" button in
   the sidebar).
2. Choose **Scan network** to find ONVIF-compliant cameras automatically (enter that
   camera's ONVIF username/password when prompted, so mirage can look up its real RTSP
   stream URL), or **Enter RTSP URL** to type one in directly for a non-ONVIF camera.
3. Fill in the camera's name, detector, tracked objects, resolution/fps, and recording
   settings, then save.

The camera is saved to the database immediately (you'll see it in `GET
http://127.0.0.1:8000/api/cameras` right away) — but it won't actually start capturing
until you **restart `python -m mirage`** (terminal 2). Config changes aren't hot-reloaded
into an already-running pipeline yet.

You can also do this directly against the API without the UI, e.g. for scripting:

```
curl -X POST http://127.0.0.1:8000/api/config/cameras \
  -H "Content-Type: application/json" \
  -d '{"name":"front_door","rtsp_url":"rtsp://user:pass@192.168.1.50:554/stream1","detector":"general","track_objects":["person","car"]}'
```

`PUT /api/config/cameras/{name}` edits a camera, `DELETE /api/config/cameras/{name}`
removes one — same "restart to apply" rule for both.

## Using a real camera for the very first run (before the DB has any config yet)

Skip terminal 1 entirely. If you haven't run `python -m mirage` yet at all (fresh
database, nothing seeded), edit `config/mirage.yaml` before that first run and it'll be
imported automatically:

```yaml
cameras:
  front_door:
    name: front_door
    detector: general
    ffmpeg:
      inputs:
        - path: rtsp://user:pass@192.168.1.50:554/stream1
    detect:
      width: 640
      height: 480
      fps: 5
    record:
      enabled: true
      continuous:
        days: 7
    objects:
      track: [person, car]
```

Then run terminal 2 **without** `--go2rtc-source` (real RTSP cameras accept multiple
concurrent connections, so mirage's capture and go2rtc can both connect directly — no
port-splitting needed):

```
python3 -m mirage --config config/mirage.yaml -v
```

Terminals 3 and 4 stay exactly the same.

## Cleaning up between runs

If you want a clean slate (empty DB, no old recordings):

```
cd /Users/gauthamkrishna/Documents/Projects/nvr/frigate_nvr/mirage
rm -f config/mirage.db config/mirage.db-shm config/mirage.db-wal
rm -rf media/recordings media/cache media/clips/events media/clips/review
```

Do this while everything is stopped (not while the pipeline is running).

## If something looks stuck

- **`ERROR ... Connection refused` / `capture thread died; restarting ffmpeg` repeating
  in terminal 2's logs**: this is expected noise with the synthetic test source, not a
  real problem. `run_test_stream.sh` only accepts one client at a time and restarts the
  instant the previous one disconnects, so mirage's own reconnect logic can race that
  gap and log a handful of failed attempts before it settles. Check
  `sqlite3 config/mirage.db "select count(*) from recordings;"` — if the count is going
  up, recording/detection are working fine underneath the noise. A real camera's RTSP
  server doesn't have this limitation, so this is test-harness-only noise.
- **Port already in use**: `lsof -nP -iTCP:19500 -sTCP:LISTEN` (swap the port number) to
  find the process, then `kill -9 <pid>`.
- **Ctrl+C doesn't stop `run_test_stream.sh`**: it's a retry loop; use
  `pkill -9 -f run_test_stream; pkill -9 -f 19500; pkill -9 -f 19501`.
- **`python -m mirage` hangs on shutdown**: give it up to ~30-60s (it's waiting on
  ffmpeg/detector processes to exit cleanly) before force-killing. If it's still stuck
  after that, `pkill -9 -f "python.*-m mirage"`.
- **go2rtc port conflicts**: go2rtc listens on `1984` (API) and `8555` (WebRTC) by
  default — make sure nothing else on your machine uses those.
