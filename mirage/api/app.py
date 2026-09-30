"""FastAPI app factory for the mirage read API.

Runs as its own process, separate from MirageApp -- it only reads the same SQLite DB
(safe under WAL mode, see mirage/db/database.py) and config/disk paths that MirageApp
writes to; it does not own or supervise any of the capture/detect/track/record pipeline.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response

from mirage.api.routers import (
    cameras,
    config as config_router,
    events,
    live,
    notifications,
    onvif,
    ptz,
    recordings,
    review,
    system,
)
from mirage.config.schema import MirageConfig
from mirage.const import CACHE_DIR as DEFAULT_CACHE_DIR
from mirage.const import EXPORT_DIR as DEFAULT_EXPORT_DIR
from mirage.db.database import close_database, init_database
from mirage.go2rtc.config import DEFAULT_API_PORT


class _SPAStaticFiles(StaticFiles):
    """Serves the Angular build like a normal static dir, except any GET that doesn't
    resolve to a real file (a client-side route like /live or /review, not an actual
    asset) falls back to index.html so Angular's own router can take over -- same
    fallback every SPA host (nginx try_files, Vercel, etc.) needs. StaticFiles' own
    html=True only covers directory-index lookups, not arbitrary unmatched paths.

    This mount is registered LAST, so it only ever sees requests no /api/* router
    already claimed -- but that still includes a MISTYPED or removed /api/* path (no
    router matches it either), which must surface as a real 404, not silently serve
    the frontend's index.html and mask a broken API call as a successful page load.
    """

    async def get_response(self, path: str, scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            is_api_path = scope["path"].startswith("/api/") or scope["path"] == "/api"
            # Only a real "no such file" (404) on a non-API path falls back to
            # index.html -- a 405 (e.g. POST to a static path), anything else, or any
            # /api/* miss propagates as a genuine error instead of being silently
            # masked as a successful page load.
            if exc.status_code != 404 or scope["method"] not in ("GET", "HEAD") or is_api_path:
                raise
            return FileResponse(Path(self.directory) / "index.html")


def create_app(
    config: MirageConfig | None = None,
    db_path: str | None = None,
    cors_origins: list[str] | None = None,
    go2rtc_host: str = "127.0.0.1",
    go2rtc_api_port: int = DEFAULT_API_PORT,
    export_dir: str = DEFAULT_EXPORT_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    static_dir: str | None = None,
) -> FastAPI:
    """`config`, if given, is used as a FIXED override for the lifetime of this app --
    only ever passed by tests that construct a MirageConfig directly without a real DB.
    In real use (config=None, the CLI default), every route reads the CURRENT config
    fresh from the DB on each request via request.app.state.get_config(), since config
    can change at any time now (the add-camera wizard writes to the same DB this API
    reads) -- baking a single static config into app.state at startup would silently go
    stale the moment a camera is added or edited.

    `go2rtc_host` defaults to 127.0.0.1 for the normal single-host layout (mirage.api and
    go2rtc/mirage.supervisor on the same machine); Docker's multi-container topology runs
    go2rtc inside a separate pipeline container, so that deployment overrides this to the
    pipeline container's Docker network hostname (see Dockerfile.api/docker-compose.yml).

    `export_dir` is injectable (same reasoning as go2rtc_api_port above) so tests can
    point stitched review clips at a throwaway tmp_path instead of the real module-level
    EXPORT_DIR constant -- avoids monkeypatching an env var and reloading mirage.const,
    which would leak global state across tests.

    `static_dir`, if given, mounts the Angular production build (the `browser/`
    subfolder of `ng build`'s output) at `/` with SPA fallback, so this one process can
    serve both the API and the UI -- what the packaged desktop app (mirage/desktop/
    launcher.py) points pywebview at instead of running a separate `ng serve`. None
    (the default) leaves routing exactly as it is today: pure API, frontend served by
    its own dev server with CORS bridging the two origins.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.database = init_database(db_path) if db_path else None
        app.state.go2rtc_host = go2rtc_host
        app.state.go2rtc_api_port = go2rtc_api_port
        app.state.export_dir = export_dir
        app.state.cache_dir = cache_dir
        if config is not None:
            app.state.get_config = lambda: config
        else:
            app.state.get_config = MirageConfig.from_db
        app.state.notify_task = asyncio.create_task(notifications.notify_tailer_loop(cache_dir))
        yield
        app.state.notify_task.cancel()
        try:
            await app.state.notify_task
        except asyncio.CancelledError:
            pass
        if app.state.database is not None:
            close_database(app.state.database)

    app = FastAPI(title="mirage API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(cameras.router)
    app.include_router(config_router.router)
    # notifications.router's GET /stream must be registered BEFORE events.router --
    # both share the /api/events prefix, and events.router's GET /{event_id} would
    # otherwise greedily match "/api/events/stream" first (event_id="stream", a real
    # route match that just 404s on lookup) since FastAPI matches in registration order.
    app.include_router(notifications.router)
    app.include_router(events.router)
    app.include_router(onvif.router)
    app.include_router(ptz.router)
    app.include_router(recordings.router)
    app.include_router(review.router)
    app.include_router(live.router)
    app.include_router(system.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # Mounted LAST and at the root: FastAPI matches routes in registration order, so
    # every /api/* router above still wins its own path before this catch-all static
    # mount ever sees a request.
    if static_dir is not None:
        app.mount("/", _SPAStaticFiles(directory=static_dir, html=True), name="frontend")

    return app
