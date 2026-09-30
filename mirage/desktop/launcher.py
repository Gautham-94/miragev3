"""The desktop app's actual orchestration: start the API + pipeline supervisor, wait for
the API to answer, open a pywebview window pointed at it, and tear everything down
cleanly when that window closes.

Import-time ordering: mirage.desktop.paths.configure_environment() MUST have already run
(see packaging/entrypoint.py, the real PyInstaller entry script) before this module is
imported -- everything below imports mirage.* modules that read path constants from
mirage.const at THEIR import time.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("mirage.desktop")

API_HOST = "127.0.0.1"
API_PORT = 8000
HEALTH_URL = f"http://{API_HOST}:{API_PORT}/api/health"
HEALTH_POLL_TIMEOUT_SECONDS = 30.0
HEALTH_POLL_INTERVAL_SECONDS = 0.25


def _configure_logging(log_dir: Path) -> None:
    """File logging only -- a --noconsole/--windowed PyInstaller build has sys.stdout
    set to None on Windows (no console to attach to), so a StreamHandler would raise on
    first emit. A single rotating-by-launch file under user_data_root/logs is enough for
    a client to attach when reporting a problem; this app has no in-window log viewer.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "desktop.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def _wait_for_api_healthy(timeout: float = HEALTH_POLL_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
    return False


def run_desktop_app() -> None:
    from mirage.desktop import single_instance

    # Must be the very first thing that runs -- exits immediately if another instance
    # already holds the lock, before touching the database, logging, or anything else.
    single_instance.acquire_or_exit()

    from mirage.const import CACHE_DIR, DB_PATH
    from mirage.desktop import job_object, paths

    user_root = paths.user_data_root()
    _configure_logging(user_root / "logs")
    logger.info("mirage desktop starting (user_data_root=%s)", user_root)

    # Also must run before anything spawns a subprocess (the API/supervisor threads
    # started just below both do, transitively) -- see job_object.py's own docstring for
    # why every child process needs to inherit this process's job membership from the
    # very first one onward, not just most of them.
    job_object.enable_kill_on_exit()

    import uvicorn

    from mirage.api.app import create_app
    from mirage.supervisor import PipelineSupervisor

    # MirageConfig.from_db() (called by both the API and the pipeline worker) already
    # self-seeds sensible defaults -- a "general" ONNX detector, zero cameras -- the
    # first time it finds no AppConfig row (see MirageConfig.from_db's own docstring).
    # Nothing to seed manually here; the add-camera wizard is how a client adds their
    # first camera, exactly like the dev flow documented in HOW_TO_RUN.md.

    api_app = create_app(
        db_path=DB_PATH, cache_dir=CACHE_DIR,
        static_dir=str(paths.frontend_dist_dir()),
        cors_origins=[f"http://{API_HOST}:{API_PORT}"],
    )
    uvicorn_config = uvicorn.Config(api_app, host=API_HOST, port=API_PORT, log_level="info")
    uvicorn_server = uvicorn.Server(uvicorn_config)
    api_thread = threading.Thread(target=uvicorn_server.run, name="mirage-api", daemon=True)
    api_thread.start()

    # Re-invokes THIS SAME frozen exe with a special first arg (see
    # packaging/entrypoint.py) instead of the source-tree default `[sys.executable, "-m",
    # "mirage", ...]` -- a frozen exe has no `-m` module-runner, it only understands
    # running itself. Source-tree (unfrozen) runs fall back to the normal default so
    # `python -m mirage.desktop.launcher` still works for local development.
    import sys

    pipeline_command = (
        [sys.executable, "--mirage-pipeline-worker"] if paths.is_frozen() else None
    )
    supervisor = PipelineSupervisor(pipeline_args=[], pipeline_command=pipeline_command)
    supervisor_thread = threading.Thread(target=supervisor.run, name="mirage-supervisor", daemon=True)
    supervisor_thread.start()

    if not _wait_for_api_healthy():
        logger.error("API did not become healthy within %ss; opening window anyway", HEALTH_POLL_TIMEOUT_SECONDS)

    import webview

    window = webview.create_window(
        "Mirage", f"http://{API_HOST}:{API_PORT}/", width=1360, height=860, min_size=(960, 600),
    )

    def _on_closed() -> None:
        logger.info("window closed, shutting down")
        supervisor.request_stop()
        uvicorn_server.should_exit = True
        supervisor_thread.join(timeout=30)
        api_thread.join(timeout=10)
        logger.info("shutdown complete")

    window.events.closed += _on_closed

    # Explicit backend, not auto-detected: Windows 10/11 ships the WebView2 runtime by
    # default, so this needs no bundled Chromium/CEF -- the whole point of choosing
    # pywebview over Electron. Pinning it also means a build-time backend mismatch
    # fails loudly at packaging/test time instead of silently falling back to a
    # different (untested) backend on some client machine.
    webview.start(gui="edgechromium")


if __name__ == "__main__":
    # Local-development entrypoint: `python -m mirage.desktop.launcher` from the repo
    # root (after `ng build` has produced frontend/dist/frontend/browser at least once).
    # The real packaged entrypoint is packaging/entrypoint.py, which additionally calls
    # multiprocessing.freeze_support() and handles the --mirage-pipeline-worker argv
    # dispatch before configure_environment() is even safe to call from a frozen exe.
    from mirage.desktop.paths import configure_environment

    configure_environment()
    run_desktop_app()
