# Live view & capture stability fixes

Four independent defects found while bringing the Docker stack up against a real RTSP
camera. Each is recorded with the evidence that identified it, because in three of the
four cases the obvious suspect turned out to be wrong.

Summary of the end state: capture drops went from ~1/second to zero, Live view runs on
WebRTC at 1920x1080 / 25fps, a dropped stream now recovers on its own, and a deployed
frontend fix actually reaches the browser.

---

## 1. Capture dropped the camera every 2-4 seconds

**Symptom.** `capture thread died; restarting ffmpeg` roughly once a second. Recordings
arrived as 2-3s fragments instead of clean 10s segments, and detection only ever saw
part of any scene.

**What it was not.** The camera looked like the obvious culprit, and it was not:

- Not a concurrent-session limit. With the entire stack stopped and exactly one ffmpeg
  connected, six sequential attempts still died at 2.1s, 1.6s, 3.2s, 1.8s, 4.3s, 2.6s --
  every one exiting cleanly (code 0) with the *input* hitting EOF.
- Not bandwidth. The 640x360 sub-stream failed identically to the 1080p main stream.
- Not mirage's code. Raw ffmpeg outside the stack failed exactly the same way.

**Root cause.** ffmpeg's RTSP client cannot hold a session to this camera; go2rtc's can.
Measured back to back against the same camera at the same time:

| Client | Session length |
|---|---|
| ffmpeg -> camera directly | 1.6s, 1.8s, 2.1s, 2.6s, 3.2s, 4.3s |
| go2rtc -> camera | 3 x 60s, ~13.4 MB each, **zero gaps >0.7s** |
| ffmpeg -> go2rtc's restream | 31.0s, 30.8s, 30.9s (full duration, clean exit) |

**Fix.** Let go2rtc own the single connection to the camera and have mirage capture from
its restream. go2rtc's RTSP server already listens on `:8554` by default, so no code
change was needed:

1. The camera's configured `rtsp_url` points at `rtsp://127.0.0.1:8554/cam1`.
2. `docker-compose.yml` passes `--go2rtc-source cam1=$CAM1_RTSP_URL` so go2rtc keeps
   using the *physical* camera. Without this, mirage generates go2rtc's stream list from
   the camera's own `rtsp_url` and go2rtc ends up pointed at its own restream -- a loop.

Side benefit: one connection to the camera instead of two, with go2rtc handling reconnects.

**Verified.** 0 drops / 0 short reads / 0 inference failures across multiple independent
90-180s windows.

**Known residual.** Exactly one capture blip ~90ms after each pipeline start -- mirage's
ffmpeg reaches for the restream a fraction of a second before go2rtc is listening. The
watchdog retries once and runs clean afterwards. A readiness check on the restream before
starting capture would remove it.

---

## 2. Live view stuck on "STILL", never showing video

**Symptom.** Tiles showed the 1fps JPEG poster (`STILL` badge) instead of live video.

**Root cause -- two faults that compound.**

*Fault A: WebRTC could not work in Docker.* It is tried first. go2rtc's generated config
had `ice_servers: []` and no `candidates`, so inside a container it advertises only
bridge-network addresses (172.x.x.x) that a browser cannot route to. Compounding it,
`"8555:8555"` publishes **tcp only** -- WebRTC needs udp. ICE failed every time.

*Fault B: the MSE fallback was dead on arrival.* `WebRtcPlayer` assigns
`videoEl.srcObject` when the track arrives, but `destroy()` never cleared it. `MsePlayer`
then assigns `videoEl.src = blobUrl`, and per the HTML spec **`srcObject` takes precedence
over `src`**, so the blob was ignored. Demonstrated in-browser:

```
srcObjectStillSet:     true
srcAttr:               blob(MSE)
currentSrc:            (empty -> using srcObject, ignoring src)
mediaSourceReadyState: "closed"   <- never opened
```

Because the MediaSource never opened, `sourceopen` never fired, so `MsePlayer` never even
opened its WebSocket. It also never errored, so no failure was reported -- the tile simply
sat in `connecting` forever. Note the `track` event fires when the remote description is
applied, *before* ICE completes, so `srcObject` was set even though WebRTC was doomed:
Fault A guaranteed Fault B triggered.

**Fix.**

- `webrtc-player.ts`: clear `videoEl.srcObject` in `destroy()`.
- `mirage/go2rtc/config.py`: emit `webrtc.candidates` from `MIRAGE_GO2RTC_WEBRTC_CANDIDATES`.
- `docker-compose.yml`: publish `8555/tcp` **and** `8555/udp`; set the candidate env var.

**Verified.** Real signaling handshake from a browser: ICE `connected`, both tcp and udp
candidates advertised, 1.69 MB of video received at 25fps, 1920x1080.

---

## 3. Live view froze permanently after switching away

**Symptom.** Worked with no latency, then stuck on the last frame after changing screens,
never recovering.

**Root cause.** `onTierFailed('mse')` called `setMode('error')` and stopped. There was no
retry anywhere in the tile. The first time the WebSocket dropped -- go2rtc restarting, a
camera blip, a laptop sleeping, a backgrounded tab's socket being torn down -- the tile
fell back to the snapshot poller permanently. Only navigating away and back recreated it.

Confirmed from the API log: the WebSocket connected, went silent, and only `snapshot.jpg`
requests continued (37 in 60s -- Chrome throttling the timer in a background tab), while
go2rtc still held a perfectly healthy producer to the camera.

**Fix** (`camera-tile.ts`): retry with backoff (2s doubling to 15s) instead of giving up;
reconnect immediately on `visibilitychange` -> visible; tear down stale players before each
attempt so two can't race the same `<video>`; clear the timer and listener on destroy.
WebRTC is skipped on retries only if it *never once* reached playback -- a transient blip
on a working WebRTC connection should not permanently demote the tile to MSE.

---

## 4. Deployed frontend fixes never reached the browser

**Symptom.** After all of the above shipped, the browser still showed `STILL`. The same
server, driven from a fresh browser session, showed `LIVE` at 1920x1080.

**Root cause.** nginx served `index.html` with **no `Cache-Control` header** -- only
`ETag`/`Last-Modified`. With no directive, browsers apply heuristic freshness and can serve
it from cache without revalidating. `index.html` is the one file whose URL is constant while
its contents change every deploy: it maps the app to the current content-hashed bundle
names. A stale copy pins the browser to a previous build's chunks indefinitely, so a
deployed fix silently never arrives and only a manual hard-refresh recovers.

**Fix** (`docker/nginx.conf`): `index.html` -> `no-cache, must-revalidate` (still a 304 on
revalidation, so it costs one conditional request); content-hashed assets ->
`public, max-age=31536000, immutable`, which is safe precisely because a new build produces
new filenames.

---

## Files changed

| File | Change |
|---|---|
| `frontend/.../players/webrtc-player.ts` | Clear `srcObject` in `destroy()` so the MSE fallback is reachable |
| `frontend/.../camera-tile/camera-tile.ts` | Reconnect with backoff + visibility re-arm; tier teardown; cleanup |
| `mirage/go2rtc/config.py` | `webrtc.candidates` via `MIRAGE_GO2RTC_WEBRTC_CANDIDATES` |
| `docker/nginx.conf` | Cache headers for `index.html` and hashed assets |
| `docker-compose.yml` | udp 8555, candidate env var, `--go2rtc-source` override |
| `.env.example` | Template for `CAM1_RTSP_URL` (`.env` is gitignored) |

## Configuration applied at runtime (not code)

These live in the config DB, applied through the API rather than in this repo:

- Detector `general` repointed to `/data/models/...`. A fresh DB seeds *relative* paths
  (`models/yolov8n.onnx`), but the pipeline image deliberately does not bake in `models/`
  -- it is bind-mounted at `/data/models`, so the detector crash-loops until repointed.
- `execution_provider: auto`, not `cuda`. The API validates the provider against its *own*
  CPU-only onnxruntime and rejects `cuda` outright even though the pipeline has GPU;
  `auto` is resolved inside the pipeline and correctly selects CUDA there.
- Switching a detector's model does **not** update its input dimensions. Moving to
  `zilodetector.onnx` (768x768) while the config still said 320x320 made *every* inference
  fail with `INVALID_ARGUMENT ... Got: 320 Expected: 768`.

Both of the first two are arguably upstream bugs: every fresh Docker install hits the
relative-path one, and the provider validation checks the wrong machine.
