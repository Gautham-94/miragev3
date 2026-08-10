# Docker Prerequisites (Windows)

What you need installed/configured on a Windows machine to run this stack
(`pipeline`, `api`, `frontend` services). Applies whether you're building the images
yourself (`docker-compose.yml`, `docker compose up --build`) or running prebuilt images
someone shipped you (see "Running from prebuilt images" below) -- the runtime
requirements (WSL2, GPU driver, ports, volumes) are identical either way.

## 1. WSL2

Docker Desktop on Windows requires the WSL2 backend (not the legacy Hyper-V backend) --
this is also what makes GPU passthrough possible.

```
wsl --install
```

Reboot if prompted. Verify:

```
wsl --status
```

should report WSL version 2 as default.

## 2. Docker Desktop

Install Docker Desktop for Windows and select the **WSL2 based engine** during setup
(Settings > General > "Use the WSL 2 based engine" should be checked). Enable
integration with your WSL distro under Settings > Resources > WSL Integration if not
already on.

## 3. NVIDIA GPU support (only if you want GPU-accelerated detection)

The `pipeline` service's image (`Dockerfile.pipeline`) is CUDA-based and
`docker-compose.yml` reserves a GPU for it. To actually use your NVIDIA GPU:

- Install a current NVIDIA driver on Windows itself (not inside WSL) -- get it from
  [nvidia.com/drivers](https://www.nvidia.com/drivers), any driver version with WSL/CUDA
  support (game-ready or studio driver, both work). Do **not** install a separate Linux
  NVIDIA driver inside WSL2 -- the Windows driver is shared through automatically.
- Docker Desktop picks up GPU access automatically once the driver is installed and the
  WSL2 backend is active -- no separate `nvidia-container-toolkit` install needed on
  Windows (that's only required for native Linux Docker hosts).
- Verify the GPU is visible to containers:
  ```
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  This should print your GPU's name/driver/CUDA version. If it fails, GPU passthrough
  isn't working yet -- recheck the driver install and Docker Desktop's WSL2 setting
  before trying to run the mirage stack.

If you don't have an NVIDIA GPU, or don't want to use it, delete the `deploy:` block
under the `pipeline` service in `docker-compose.yml` -- the same image still runs, just
CPU-only (mirage's execution-provider resolution already falls back to CPU
automatically; see `mirage/detection/execution_providers.py`).

## 4. Disk space

The `pipeline` image (CUDA runtime + cuDNN + onnxruntime-gpu, plus a second venv for
SpeciesNet) is large -- budget several GB free for the image layers alone, plus however
much you plan to record into the `media` volume.

## 5. Model files

No model files are baked into any image. Create a `models/` folder next to
`docker-compose.yml` (bind-mounted to `/data/models` in both `pipeline` and `api`) and
put your `.onnx` model + labelmap files there before or after first startup -- point
each detector's `model_path`/`labelmap_path` at the container-side path (e.g.
`/data/models/best.onnx`) via the frontend's Manage Detectors page.

## 6. SpeciesNet (optional -- species classification for Animal/Bird events)

`Dockerfile.pipeline` bakes in a second, isolated Python venv (`/app/.venv-speciesnet`)
with the `speciesnet` package installed, so this feature works out of the box once
enabled -- no separate setup needed inside the container. It stays fully inactive
(`species_status` = "skipped" on every event) until you turn it on via
`config.species_classifier.enabled` (Manage Detectors / species config in the frontend).

If you do enable it:

- **First classification needs outbound internet access** from the `pipeline`
  container -- SpeciesNet downloads its model weights via `kagglehub` on first use
  (can take a few minutes). `docker-compose.yml`'s `speciesnet-cache` volume persists
  that download across container restarts/recreations, so it only happens once.
- If Kaggle ever requires authentication for the underlying dataset, set
  `KAGGLE_USERNAME` / `KAGGLE_KEY` as environment variables on the `pipeline` service in
  `docker-compose.yml` (see kagglehub's own docs).
- If you never plan to use species classification, you can ignore all of the above --
  it costs some extra image build time/size but nothing at runtime.

## 7. Ports

Make sure nothing else on the Windows host is already using:

- `8080` -- frontend (nginx)
- `1984` -- go2rtc API/HTTP
- `8555` -- go2rtc WebRTC

## After prerequisites are met (building from source)

```
docker compose up --build
```

Frontend: http://localhost:8080. First run starts with no cameras configured -- add
them through the frontend's Add Camera wizard. If you enabled GPU support, also set the
relevant detector's execution provider to `cuda` (or `auto`) on the Manage Detectors
page -- it defaults to `cpu` even when the container has GPU access.

## Running from prebuilt images (no source code, no build step)

If you received `.tar` image files instead of the source repo (e.g.
`mirage-pipeline-<version>.tar`, `mirage-api-<version>.tar`,
`mirage-frontend-<version>.tar`) plus `docker-compose.client.yml`, you don't need to
build anything -- just load the images and start the stack:

```
docker load -i mirage-pipeline-<version>.tar
docker load -i mirage-api-<version>.tar
docker load -i mirage-frontend-<version>.tar

docker compose -f docker-compose.client.yml up -d
```

All prerequisites above (WSL2, Docker Desktop, NVIDIA driver, ports, the `models/`
folder) still apply exactly the same -- only the "build the image yourself" step is
skipped. `docker load` prints the image name/tag it loaded; make sure those match the
`image:` lines in `docker-compose.client.yml` (edit that file if the version tag
differs from `latest`).

Verify the images loaded before starting the stack:

```
docker images | findstr mirage
```

Frontend: http://localhost:8080, same as the from-source flow. To update to a newer
version later, get the new `.tar` files, `docker load` them (this replaces the old
tagged image), then `docker compose -f docker-compose.client.yml up -d` again to
recreate the containers from the new images.
