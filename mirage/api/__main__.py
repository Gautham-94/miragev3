"""CLI entrypoint: `python -m mirage.api --db-path path/to/mirage.db`.

Starts the read API as its own uvicorn server, separate from `python -m mirage` (the
capture/detect/track/record pipeline). Both point at the same db-path, and both read
config from that same database (see mirage/config/store.py) -- so the API always reflects
whatever the wizard/config-write endpoints have most recently saved, not a stale snapshot
taken at API startup.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from mirage.api.app import create_app
from mirage.const import CACHE_DIR, DB_PATH, ensure_cuda_dll_directories_on_path
from mirage.go2rtc.config import DEFAULT_API_PORT


def main() -> None:
    # See mirage/__main__.py's identical call for why: onnxruntime-gpu's
    # CUDAExecutionProvider needs its DLL directories on PATH before any session that
    # might request it is created. A no-op on non-Windows / without those wheels
    # installed.
    ensure_cuda_dll_directories_on_path()

    parser = argparse.ArgumentParser(prog="mirage.api", description="Run the mirage read API")
    parser.add_argument("--db-path", default=DB_PATH, help="SQLite database path (must match the running mirage instance)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--go2rtc-host", default="127.0.0.1", help="host/hostname of the running go2rtc instance (e.g. a Docker service name if go2rtc runs in a separate container)")
    parser.add_argument("--go2rtc-api-port", type=int, default=DEFAULT_API_PORT, help="port the running go2rtc instance's API listens on")
    parser.add_argument(
        "--cache-dir", default=CACHE_DIR,
        help="must match the cache_dir mirage.supervisor is running with, so GET/POST "
             "/api/system/* can find its status/restart-request files",
    )
    parser.add_argument("--cors-origin", action="append", dest="cors_origins", help="allowed CORS origin (repeatable); default: allow all")
    parser.add_argument(
        "--static-dir", default=None,
        help="serve the Angular production build (ng build's browser/ output dir) at / "
             "with SPA fallback, instead of requiring a separate `ng serve`",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    app = create_app(
        db_path=args.db_path, cors_origins=args.cors_origins,
        go2rtc_host=args.go2rtc_host, go2rtc_api_port=args.go2rtc_api_port,
        cache_dir=args.cache_dir, static_dir=args.static_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
