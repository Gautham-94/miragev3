# Linux (Ubuntu) direct install guide

A production-style install of mirage on Ubuntu, running directly on the host (no
Docker) via systemd. Assumes a fresh Ubuntu 22.04 or 24.04 server/desktop with a
regular (non-root) user account and sudo access.

This mirrors what `Dockerfile.pipeline`/`Dockerfile.api`/`Dockerfile.frontend` already
do for the containerized deployment (same system packages, same CPU-only torch wheel
concern, same env-var-driven paths from `mirage/const.py`) — just laid out as native
systemd services instead of containers.

---

## 0. System packages

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip ffmpeg curl ca-certificates git nginx
```

- **`ffmpeg`** — required at runtime by every camera's capture/record subprocess
  (`mirage/capture/`, `mirage/recording/`) and by the review-clip stitcher
  (`mirage/recording/stitch.py`). Not a Python dependency — comes from apt, exactly as
  `Dockerfile.pipeline` installs it.
- **`curl`/`ca-certificates`** — `mirage/go2rtc/download.py` auto-downloads the correct
  go2rtc binary for `linux/amd64` or `linux/arm64` on first run; no manual binary swap
  needed (confirmed against go2rtc's own release asset naming in that module).
- **`nginx`** — reverse-proxies the built frontend + `/api/*` to `mirage.api` on one
  origin (see step 6) — the production frontend build assumes same-origin
  (`environment.prod.ts`: `apiBaseUrl: ''`), so *something* has to do this; nginx is
  what the existing `Dockerfile.frontend` uses too.
- **Python 3.12**: if your Ubuntu release doesn't ship `python3.12` in its default
  repos (check with `apt-cache policy python3.12` first), add the deadsnakes PPA:
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update && sudo apt install -y python3.12 python3.12-venv
  ```
  `pyproject.toml` only requires Python `>=3.11`, so a native `python3.11` also works if
  your distro already has it and you'd rather skip the PPA.

If you'll run YOLOv8/MegaDetector on an NVIDIA GPU instead of CPU, install the NVIDIA
driver + CUDA toolkit separately first (`nvidia-smi` should work before you continue) —
covered in step 4's GPU note, not repeated here since it's entirely distro/driver-version
specific.

---

## 1. Get the code

```bash
sudo mkdir -p /opt/mirage
sudo chown "$USER":"$USER" /opt/mirage
git clone <your-repo-url> /opt/mirage
cd /opt/mirage/mirage   # the actual app lives in the mirage/ subdirectory
```

Everywhere below assumes you're inside that `mirage/` directory (the one containing
`requirements.txt`, `mirage/` the package, `frontend/`, etc.) unless stated otherwise.

---

## 2. Main Python environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
```

**Install torch as CPU-only first**, before the rest of `requirements.txt` — otherwise
pip's default index resolves `torch==2.12.*` to CUDA-bundled wheels
(`nvidia-cudnn-cu13`/`nvidia-nccl-cu13`/etc., 1GB+ of libraries you don't need unless
you're actually running OWLv2 on an NVIDIA GPU). This is exactly what
`Dockerfile.pipeline`/`Dockerfile.api` already do:

```bash
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/pip install -e .
```

If you DO have an NVIDIA GPU and want OWLv2 (open-vocabulary search) accelerated on it,
skip the `--extra-index-url` line and let pip install the default CUDA-enabled torch
build instead — but see the GPU note in step 4 first, since object detection
(onnxruntime) needs a separate GPU package regardless of which torch you install.

Confirm the venv is healthy:

```bash
.venv/bin/python -c "import cv2, onnxruntime, torch, norfair; print('ok')"
```

---

## 3. (Optional) SpeciesNet — isolated second venv

Only needed if you want real species classification (`config.species_classifier`).
Skip this whole section if not — every Animal/Bird Event's `species_status` just stays
`"skipped"` forever, which is a completely valid, working state.

The real `speciesnet` PyPI package's transitive deps (`yolov5`/`ultralytics`/`sahi`/
`roboflow`) require `numpy>=2` and the GUI `opencv-python` package — both genuinely,
unresolvably conflict with the main venv's pinned `norfair` (`numpy<2.0`) and
`opencv-python-headless` (confirmed by directly attempting the install and hitting real
pip dependency-resolver errors, not just warnings). SpeciesNet therefore runs in its
own, completely separate venv, talking to the main pipeline over a small
stdin/stdout JSON protocol (see `mirage/species/plugins/speciesnet.py`'s own module
docstring) — not shared with the main venv at all:

```bash
python3.12 -m venv .venv-speciesnet
.venv-speciesnet/bin/pip install -r requirements-speciesnet.txt
```

Model weights are auto-downloaded on first real use via `kagglehub`/`huggingface_hub`
(same lazy-download pattern OWLv2 already uses) — no manual model file placement
needed. First classification will be slow (weight download); subsequent ones are fast.

If you later export/convert any other `ultralytics`-based `.pt` checkpoint to ONNX
(as this session did for a custom detector), do that export from
**`.venv-speciesnet`** too (it already has `ultralytics` installed as SpeciesNet's own
transitive dependency) — never install `ultralytics` into the main venv, for the exact
same numpy/opencv conflict reason.

---

## 4. GPU acceleration (optional)

This machine's dev sessions ran on Apple Silicon (CoreML) — **CoreML does not exist on
Linux**. On Ubuntu, your realistic accelerator options are:

- **NVIDIA GPU via CUDA** — install `onnxruntime-gpu` instead of plain `onnxruntime` in
  the main venv (`pip uninstall onnxruntime && pip install onnxruntime-gpu==1.22.*`,
  matching the pinned version), and set each detector's `execution_provider` to
  `cuda` via the Config/Detectors page. `mirage/detection/execution_providers.py`
  already has full CUDA support wired in (`available_execution_providers()`/
  `resolve_providers()`) — this is a real, tested code path, not a stub.
- **CPU only** — the default and always-available fallback (`execution_provider: cpu`
  or `auto`, which degrades to CPU automatically when no accelerator is installed).
  Every detector this session used was benchmarked at both 320×320 and 1280×1280 on
  CPU-comparable numbers are in `DETECTOR_SCALING.md`/`PROCESS_AUDIT.md` if you want a
  sizing baseline before committing to a GPU purchase.

Do NOT install both plain `onnxruntime` and `onnxruntime-gpu` in the same venv — they
conflict at the package level (same `onnxruntime` Python module namespace, similar to
the `opencv-python`/`opencv-python-headless` conflict already documented in this repo).

---

## 5. Frontend build

```bash
cd frontend
sudo apt install -y nodejs npm   # or use nvm if you need a specific Node version
npm ci
npm run build
```

This produces `frontend/dist/frontend/browser/` — a static site, served by nginx (step
6). There is no frontend dev server in a production install — `ng serve` is a dev-only
tool.

---

## 6. Reverse proxy (nginx)

The production build expects the API at the **same origin** it's served from
(`environment.prod.ts`: `apiBaseUrl: ''`) — nginx needs to serve the static frontend
files AND proxy `/api/*` (including the `/api/live/*/ws` WebSocket route) to
`mirage.api`. This repo already has a working config for the containerized deployment
at `docker/nginx.conf` — reuse it nearly as-is:

```bash
cat docker/nginx.conf
```

Adapt it into `/etc/nginx/sites-available/mirage`:

```nginx
server {
    listen 80;
    server_name _;

    root /opt/mirage/mirage/frontend/dist/frontend/browser;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;  # /api/live/*/ws is long-lived
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/mirage /etc/nginx/sites-enabled/mirage
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Adjust `proxy_pass`'s port if you changed `--port` for `mirage.api` in step 7, and add
TLS (certbot/Let's Encrypt) separately if this is reachable outside your LAN — this
guide doesn't cover that since it's identical to any other nginx-fronted app.

---

## 7. Data directories and environment variables

`mirage/const.py` resolves every storage path via env vars, defaulting to paths under
the repo itself if unset (fine for the dev flow in `HOW_TO_RUN.md`, not ideal for a
real server). Point them at real storage instead — `MIRAGE_CACHE_DIR` in particular
should be tmpfs (`/dev/shm`-backed) per the original spec, since it holds the
short-lived ffmpeg segment cache that gets read/written constantly:

```bash
sudo mkdir -p /var/lib/mirage/{config,media,cache} /dev/shm/mirage-cache
sudo chown -R "$USER":"$USER" /var/lib/mirage /dev/shm/mirage-cache
```

Create `/opt/mirage/mirage/.env` (sourced by the systemd units in step 8, not read by
Python directly):

```bash
MIRAGE_CACHE_DIR=/dev/shm/mirage-cache
MIRAGE_BASE_DIR=/var/lib/mirage/media
MIRAGE_CONFIG_DIR=/var/lib/mirage/config
MIRAGE_DB_PATH=/var/lib/mirage/config/mirage.db
MIRAGE_MODEL_CACHE_DIR=/opt/mirage/mirage/models
MIRAGE_GO2RTC_BIN_DIR=/var/lib/mirage/go2rtc-bin
```

`MIRAGE_MODEL_CACHE_DIR` stays inside the repo checkout (`models/`) since that's where
you'll actually place/export `.onnx` files, matching the workflow this session used
(model registration stores a path like `models/zilodetector.onnx`, resolved relative to
wherever the process's CWD is — see step 8's `WorkingDirectory` for why this matters).

---

## 8. systemd services

Three units: the pipeline (via `mirage.supervisor`, exactly as `Dockerfile.pipeline`
runs it — NOT `python -m mirage` directly, since the supervisor is what makes the
frontend's "Apply changes" restart button work at all) and the API. The frontend has no
process of its own in this layout — nginx serves its static build directly (step 6).

`/etc/systemd/system/mirage-pipeline.service`:

```ini
[Unit]
Description=mirage NVR pipeline (capture/detect/track/record)
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/opt/mirage/mirage
EnvironmentFile=/opt/mirage/mirage/.env
ExecStart=/opt/mirage/mirage/.venv/bin/python -m mirage.supervisor -v
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/mirage-api.service`:

```ini
[Unit]
Description=mirage read API
After=network.target mirage-pipeline.service

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/opt/mirage/mirage
EnvironmentFile=/opt/mirage/mirage/.env
ExecStart=/opt/mirage/mirage/.venv/bin/python -m mirage.api --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Replace `YOUR_USERNAME` with the real account that owns `/opt/mirage` and
`/var/lib/mirage`. `WorkingDirectory` matters: relative paths in config (model paths,
labelmap paths — e.g. `models/zilodetector.onnx`, as registered via the Config page in
this session) resolve relative to the process's CWD, so both units must run from the
same directory you'd `cd` into when testing manually.

`Restart=on-failure` gives you an OS-level safety net under `mirage.supervisor`'s own
in-process watchdog (`mirage/watchdog.py`, `mirage/capture/watchdog.py`) — if the
entire Python process itself dies (not just one subprocess it manages), systemd brings
it back.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mirage-pipeline.service
sudo systemctl enable --now mirage-api.service
```

Check status/logs:

```bash
sudo systemctl status mirage-pipeline mirage-api
journalctl -u mirage-pipeline -f
journalctl -u mirage-api -f
```

`journalctl` here fully replaces the `/tmp/mirage_supervisor.log`-style file redirection
used during ad-hoc `nohup` testing on this Mac dev machine — systemd captures stdout/
stderr automatically.

---

## 9. First-run config seed

The very first time `mirage-pipeline` starts against a brand-new, empty
`MIRAGE_DB_PATH`, it has no cameras/detectors configured yet. Either:

- Seed from the sample `config/mirage.yaml` once (`ExecStart=... -m mirage.supervisor -v --config config/mirage.yaml`, then remove `--config` from the unit file and
  `daemon-reload` + restart after that first successful boot — matching
  `HOW_TO_RUN.md`'s "only read on that first, DB-empty run" behavior), or
- Just start with the empty DB and add your first camera/detector entirely through the
  frontend's Add Camera wizard / Detectors page — the cleaner path for a real
  deployment, since you're not carrying over this Mac's dev-only sample YAML.

Either way, every config change (new camera, detector toggle, `num_workers`, etc.)
still requires the explicit "Apply changes" restart (`POST /api/system/restart`, wired
to the same `mirage.supervisor` restart-request file mechanism) — nothing hot-reloads,
by design (see `mirage/supervisor.py`'s own docstring on why that was tried and
reverted).

---

## 10. Verify

```bash
curl -s http://127.0.0.1:8000/api/system/status
curl -s http://127.0.0.1:8000/api/config/detectors
```

Then open `http://<server-ip>/` in a browser — nginx serves the frontend, which talks
to the API through the same origin via the `/api/` proxy path configured in step 6.

---

## Notes / things that differ from this repo's macOS dev environment

- **No CoreML** — every "coreml"-configured detector from Mac-side testing must be
  re-pointed at `cpu`, `cuda` (if you added an NVIDIA GPU), or `auto` via the
  Config/Detectors page before/after migrating; `coreml` simply isn't a valid
  `ExecutionProvider` choice `available_execution_providers()` will ever report on
  Linux, so a camera left pointed at it will silently fail detector registration
  validation (`mirage/api/routers/config.py`'s own check) rather than degrading
  quietly.
- **`bin/go2rtc`** (the macOS binary checked into this repo for local dev) is never
  used on Linux — `mirage/go2rtc/download.py` downloads the correct Linux binary into
  `MIRAGE_GO2RTC_BIN_DIR` on first run automatically. No action needed, just don't
  expect the committed macOS binary to be relevant here.
- **mock_cameras** (the synthetic ONVIF camera emulator used for local testing this
  session) is a separate, standalone repo/venv with its own setup — not part of this
  guide, since a real Linux deployment is presumably pointed at real cameras. Its own
  README covers running it if you still want a synthetic test source on Linux.
- **hardware camera network reachability**: real RTSP cameras (like the Hikvision one
  used this session) need this server to actually be on the same LAN/VLAN as the
  camera — this is an infra/network concern independent of the mirage install itself.
