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
from mirage.const import DB_PATH
from mirage.go2rtc.config import DEFAULT_API_PORT


def main() -> None:
    parser = argparse.ArgumentParser(prog="mirage.api", description="Run the mirage read API")
    parser.add_argument("--db-path", default=DB_PATH, help="SQLite database path (must match the running mirage instance)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--go2rtc-api-port", type=int, default=DEFAULT_API_PORT, help="port the running go2rtc instance's API listens on")
    parser.add_argument("--cors-origin", action="append", dest="cors_origins", help="allowed CORS origin (repeatable); default: allow all")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    app = create_app(db_path=args.db_path, cors_origins=args.cors_origins, go2rtc_api_port=args.go2rtc_api_port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
