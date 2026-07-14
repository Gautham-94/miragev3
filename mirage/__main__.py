"""CLI entrypoint: `python -m mirage --db-path path/to/mirage.db [--config path/to/seed.yaml]`.

Config now lives in the database (see mirage/config/store.py), not a hand-edited YAML
file -- this is what lets the add-camera wizard write config changes that actually take
effect on the next run. `--config` is now a one-time IMPORT flag: it only has any effect
on a genuinely fresh database (no AppConfig row yet), where it seeds the DB from that
YAML file instead of the built-in defaults. On every subsequent run, `--config` is
ignored entirely and the DB is always the source of truth -- edit cameras/detectors via
the API/wizard, not by re-editing the YAML file and expecting it to take effect again.

Starts MirageApp, and blocks until interrupted (Ctrl+C / SIGTERM), then shuts down
gracefully.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from mirage.app import MirageApp
from mirage.config.schema import MirageConfig
from mirage.const import CACHE_DIR, DB_PATH, RECORD_DIR
from mirage.db.database import close_database, init_database
from mirage.db.models import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="mirage", description="Run the mirage NVR pipeline")
    parser.add_argument(
        "--config", default=None,
        help="path to a MirageConfig YAML file to import ONE TIME ONLY, if the database "
             "has no config yet (fresh install). Ignored on every subsequent run -- the "
             "database is the source of truth once it has a config. Omit entirely to "
             "start from the built-in defaults (general ONNX detector, no cameras) "
             "instead of importing a file.",
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="scratch dir for ffmpeg cache segments + IPC sockets")
    parser.add_argument("--record-dir", default=RECORD_DIR, help="permanent recording storage dir")
    parser.add_argument("--db-path", default=DB_PATH, help="SQLite database path")
    parser.add_argument(
        "--go2rtc-source", action="append", default=[], metavar="CAMERA=URL",
        help="override the stream URL go2rtc uses for CAMERA (repeatable). Real cameras "
             "(RTSP) accept many concurrent clients so this is never needed in production; "
             "it exists for local testing against a synthetic single-client TCP source "
             "where mirage's own capture process and go2rtc can't share one port.",
    )
    parser.add_argument("--no-go2rtc", action="store_true", help="disable go2rtc (no live view restreaming)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger = logging.getLogger("mirage.cli")

    go2rtc_stream_overrides = {}
    for entry in args.go2rtc_source:
        camera_name, _, url = entry.partition("=")
        if not url:
            parser.error(f"--go2rtc-source must be CAMERA=URL, got {entry!r}")
        go2rtc_stream_overrides[camera_name] = url

    # Config now lives in the DB -- open it here (rather than waiting for
    # MirageApp.start() to do so) specifically to decide what config to run with.
    # MirageApp.start() re-initializes the database on the same path later, which is
    # safe (same underlying WAL file, just a fresh connection object).
    database = init_database(args.db_path)
    if args.config is not None and AppConfig.select().count() == 0:
        logger.info("no existing config in %s, importing %s (one-time)", args.db_path, args.config)
        config = MirageConfig.from_yaml_file(args.config)
        config.save_to_db()
    else:
        config = MirageConfig.from_db()
    close_database(database)

    app = MirageApp(
        config, cache_dir=args.cache_dir, record_dir=args.record_dir, db_path=args.db_path,
        enable_go2rtc=not args.no_go2rtc, go2rtc_stream_overrides=go2rtc_stream_overrides or None,
    )

    stop_requested = False

    def _handle_signal(signum, frame):
        nonlocal stop_requested
        logger.info("received signal %s, shutting down...", signal.Signals(signum).name)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    app.start()
    try:
        while not stop_requested:
            time.sleep(0.5)
    finally:
        app.stop()


if __name__ == "__main__":
    sys.exit(main() or 0)
