"""Database connection setup.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 10.

WAL mode, synchronous=NORMAL (safe under WAL, much faster than FULL), a generous page
cache. `db_proxy` (in models.py) is bound to a concrete database instance at startup via
`init_database()`, so model classes can be imported/defined before any actual file path is
known -- this is what lets multiple processes share the same model definitions while each
opening its own connection to the same on-disk file.
"""

from __future__ import annotations

from pathlib import Path

from playhouse.sqlite_ext import SqliteExtDatabase

from mirage.db.models import ALL_MODELS, db_proxy


def init_database(db_path: str) -> SqliteExtDatabase:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    database = SqliteExtDatabase(
        db_path,
        pragmas={
            "journal_mode": "wal",
            "synchronous": "normal",  # safe under WAL, much faster than "full"
            "cache_size": -1 * 512 * 1000,  # 512MB page cache, negative = KB
            "foreign_keys": 1,
        },
    )
    db_proxy.initialize(database)
    database.connect(reuse_if_open=True)
    database.create_tables(ALL_MODELS, safe=True)
    return database


def close_database(database: SqliteExtDatabase) -> None:
    if not database.is_closed():
        database.close()
