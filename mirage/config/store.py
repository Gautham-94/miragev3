"""Reads/writes the singleton AppConfig DB row that backs MirageConfig.from_db()/
.save_to_db(). Kept in its own module (rather than inline in mirage/config/schema.py) so
the Pydantic schema module itself has no direct peewee/DB dependency -- mirrors the
existing split between mirage/db/models.py (schema) and mirage/db/database.py (connection
lifecycle).

Requires init_database() to already have been called by the caller (same assumption
every other DB access in this codebase makes -- AppConfig shares the same db_proxy
connection as Event/Recordings/ReviewSegment/etc., there is no separate path/connection
for config specifically).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mirage.db.models import AppConfig
from mirage.util.time import utcnow

if TYPE_CHECKING:
    from mirage.config.schema import MirageConfig

_SINGLETON_ROW_ID = 1


def load_or_seed_config() -> "MirageConfig":
    from mirage.config.schema import MirageConfig

    row = AppConfig.get_or_none(AppConfig.id == _SINGLETON_ROW_ID)
    if row is None:
        config = MirageConfig.default()
        save_config(config)
        return config
    return MirageConfig.model_validate(row.data)


def save_config(config: "MirageConfig") -> None:
    payload = config.model_dump(mode="json")
    AppConfig.insert(id=_SINGLETON_ROW_ID, data=payload, updated_at=utcnow()).on_conflict(
        conflict_target=[AppConfig.id],
        update={AppConfig.data: payload, AppConfig.updated_at: utcnow()},
    ).execute()
