"""Resolves where the packaged app reads bundled (read-only) assets from and where it
writes persistent (user) state to, then stamps both into the MIRAGE_* env vars that
mirage/const.py reads at import time.

CRITICAL ordering constraint: configure_environment() must run before ANY `mirage.*`
module is imported anywhere in the process (including transitively -- e.g. importing
mirage.app pulls in mirage.const at module scope). mirage/const.py resolves every path
constant (CACHE_DIR, DB_PATH, RECORD_DIR, ...) once, at import time, from these env vars
-- setting them after the fact does nothing. This module itself imports nothing from
`mirage` for exactly that reason: it has to be safe to import and call first.

Two distinct roots, for two different reasons:
  - bundle_root(): read-only assets PyInstaller packaged alongside the exe (the Angular
    build, vendored ffmpeg/go2rtc binaries, the ONNX models). Resolved via sys._MEIPASS,
    which PyInstaller sets correctly in both --onedir and --onefile modes -- never
    hand-derived from sys.executable, whose relationship to bundled data differs across
    PyInstaller versions/modes.
  - user_data_root(): everything the app WRITES at runtime (the SQLite DB, recordings,
    clips, ffmpeg cache segments, go2rtc's own runtime config). This must NOT live
    inside the bundle -- a --onedir install directory is commonly under Program Files,
    not writable by a standard user, and gets wiped/replaced on every app update/
    reinstall regardless. %LOCALAPPDATA% is the standard writable-without-admin,
    survives-an-update location on Windows.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_DIR_NAME = "Mirage"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Where PyInstaller put this build's data files (see packaging/mirage_app.spec's
    `datas`). Falls back to the repo root when running from source (`python -m
    mirage.desktop.launcher` during development, not yet frozen).
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(meipass)
    # mirage/desktop/paths.py -> mirage/desktop -> mirage -> repo root
    return Path(__file__).resolve().parent.parent.parent


def user_data_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    # Non-Windows fallback so this module stays importable for local dev/testing of the
    # launcher itself -- the shipped client build is Windows-only (see packaging/).
    return Path.home() / f".{APP_DIR_NAME.lower()}"


def _copy_bundled_dir_once(src: Path, dest: Path) -> Path:
    """Copies a bundled read-only asset dir out to the writable user_data_root the
    FIRST time it's needed, then reuses that copy on every later launch (checked via a
    simple existence test, not a hash -- these are versioned by the app version, not
    live-updated at runtime). Three assets need this, for three different reasons:
      - models/: onnxruntime only reads from here today, but pointing it at the bundle
        forever would break the moment ANY future feature wants to cache a downloaded
        model next to the built-in ones.
      - bin/: ensure_go2rtc_binary()'s own dest.exists() short-circuit (mirage/go2rtc/
        download.py) is what actually matters here -- pre-populating this dir with the
        vendored go2rtc.exe is what makes it skip the network download entirely on a
        client machine that may have no internet access.
      - kagglehub/: same reasoning as bin/, for SpeciesNet's model weights -- see
        configure_environment()'s KAGGLEHUB_CACHE handling below. A writable copy (rather
        than pointing KAGGLEHUB_CACHE straight at the read-only bundle) is used out of
        caution: kagglehub's cache-hit path is a read-only check in practice, but nothing
        guarantees a future kagglehub version never touches the cache dir itself (a lock
        file, a checksum sidecar) the way go2rtc's own bin dir already needs to be
        writable for.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
    return dest


def configure_environment() -> Path:
    """Sets every MIRAGE_* env var mirage/const.py reads, and prepends the bundled
    ffmpeg's directory to PATH. Returns user_data_root() for callers that also want it
    directly (e.g. to place a single-instance lock file).

    Must be called exactly once, before the first `import mirage.<anything else>`.
    """
    bundle = bundle_root()
    user_root = user_data_root()

    os.environ.setdefault("MIRAGE_BASE_DIR", str(user_root / "media"))
    os.environ.setdefault("MIRAGE_CACHE_DIR", str(user_root / "cache"))
    os.environ.setdefault("MIRAGE_CONFIG_DIR", str(user_root / "config"))

    models_dest = _copy_bundled_dir_once(bundle / "models", user_root / "models")
    os.environ.setdefault("MIRAGE_MODEL_CACHE_DIR", str(models_dest))

    go2rtc_bin_dest = _copy_bundled_dir_once(bundle / "vendor" / "go2rtc", user_root / "bin")
    os.environ.setdefault("MIRAGE_GO2RTC_BIN_DIR", str(go2rtc_bin_dest))

    # SpeciesNet's model weights (~500MB, downloaded once via kagglehub during this
    # project's own development -- see packaging/README.md's vendoring step) are shipped
    # inside the bundle under vendor/kagglehub, laid out exactly as kagglehub's own cache
    # (models/<owner>/<model>/<framework>/<version>/<rev>/...) so kagglehub's own "already
    # cached" check finds it and skips the network download entirely. This is what lets
    # species classification work out of the box on a client machine with no internet
    # access, and keeps the "your video/data never leaves your network" claim actually
    # true instead of contradicted by a first-use download to Kaggle's servers.
    #
    # KAGGLEHUB_CACHE only needs to be set in THIS process: mirage/species/plugins/
    # speciesnet.py's subprocess.Popen call for the worker doesn't pass env=, so it
    # inherits the full environment block, including this var, from whichever process
    # constructs SpeciesNetClassifier (the SpeciesProcess child, itself spawned by
    # MirageApp, which in turn inherits it from this top-level process -- env vars
    # propagate down every subprocess/multiprocessing layer on Windows by default).
    kagglehub_bundle_dir = bundle / "vendor" / "kagglehub"
    if kagglehub_bundle_dir.is_dir():
        kagglehub_dest = _copy_bundled_dir_once(kagglehub_bundle_dir, user_root / "kagglehub")
        os.environ.setdefault("KAGGLEHUB_CACHE", str(kagglehub_dest))

    ffmpeg_dir = bundle / "vendor" / "ffmpeg"
    if ffmpeg_dir.is_dir():
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")

    _add_cuda_dll_directories_to_path()

    for d in (user_root, user_root / "media", user_root / "cache", user_root / "config"):
        Path(d).mkdir(parents=True, exist_ok=True)

    return user_root


# Self-contained duplicate of mirage/const.py's ensure_cuda_dll_directories_on_path()
# -- NOT imported from there, deliberately (see this module's own top-of-file docstring:
# importing mirage.const here, before configure_environment() has set the MIRAGE_* env
# vars it reads at import time, would bake in the wrong paths for the rest of the
# process). Keep the component list in sync with const.py's _CUDA_DLL_COMPONENTS if it
# ever changes.
#
# Two cases, two different sources for the same DLLs, but the SAME shape: each
# component gets added to PATH as its own separate directory, never flattened together.
#   - unfrozen (dev, `python -m mirage.desktop.launcher` against .venv): read straight
#     out of site-packages/nvidia/*/bin via sys.prefix, same as the source-tree
#     entrypoints.
#   - frozen (packaged build): packaging/mirage_app.spec copies each component's DLLs
#     into its own bundle_root()/"vendor"/"cuda"/<component>/ subdirectory (see that
#     spec's own CUDA-bundling comment for why: an earlier flat vendor/cuda/ layout with
#     everything mixed together LOOKED harmless -- no filename collisions between
#     components -- but broke cuDNN 9.x's own internal lazy-loading of its sub-DLLs,
#     confirmed live via a real CUDAExecutionProvider inference call failing with
#     `Entry Point Not Found` for a cudnn-internal symbol, while the identical
#     onnxruntime-gpu+cudnn version combination worked fine in dev-tree, which keeps
#     this same per-component separation).
_CUDA_DLL_COMPONENTS = ("cublas", "cuda_nvrtc", "cuda_runtime", "cudnn", "cufft", "nvjitlink")


def _add_cuda_dll_directories_to_path() -> None:
    if sys.platform != "win32":
        return

    if is_frozen():
        cuda_root = bundle_root() / "vendor" / "cuda"
        for component in _CUDA_DLL_COMPONENTS:
            bin_dir = cuda_root / component
            if bin_dir.is_dir():
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        return

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for component in _CUDA_DLL_COMPONENTS:
        bin_dir = nvidia_root / component / "bin"
        if bin_dir.is_dir():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def frontend_dist_dir() -> Path:
    """The Angular production build's browser/ output (see packaging/mirage_app.spec's
    `datas` -- built by `ng build` and copied in under this exact name). Running from
    source, this points at the repo's own frontend/dist/frontend/browser so `python -m
    mirage.desktop.launcher` works during development too, as long as `ng build` has
    been run at least once (there is no dev-server fallback here -- use `ng serve` +
    mirage.api's plain CORS mode for iterative frontend work instead).
    """
    return bundle_root() / "frontend_dist"


def speciesnet_worker_path() -> Path | None:
    """The frozen SpeciesNet worker exe (packaging/speciesnet_worker.spec), if this
    build vendors one. None means species classification stays unavailable unless the
    user points SpeciesModelConfig.venv_python_path at their own venv manually (the
    existing, unchanged dev-time path) -- species classification is optional (see
    SpeciesClassifierConfig.enabled), so a missing worker is never fatal to startup.

    This is an onedir build (a folder: the exe plus its own _internal/ deps), NOT a
    single onefile exe -- onefile was the original choice here but was reverted after
    it added several extra minutes to an already-slow PyTorch/CUDA cold start, since
    onefile re-extracts its entire ~2.8GB payload to a fresh temp directory on EVERY
    launch (this worker relaunches on every app open/pipeline restart, not a rare
    one-time event). mirage_app.spec's datas entry copies the whole
    dist_speciesnet/mirage_speciesnet_worker/ folder to vendor/mirage_speciesnet_worker/
    to match.
    """
    candidate = bundle_root() / "vendor" / "mirage_speciesnet_worker" / "mirage_speciesnet_worker.exe"
    return candidate if candidate.exists() else None
