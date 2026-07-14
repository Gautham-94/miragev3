"""Datetime helpers for values that will be written to the database.

IMPORTANT: peewee's `DateTimeField.formats` (the strptime patterns it uses to parse a
value read back out of SQLite) has no timezone-offset pattern -- see
`peewee.DateTimeField.formats`, which is `['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
'%Y-%m-%d']`. A timezone-AWARE datetime (e.g. `datetime.now(timezone.utc)`) serializes to
SQLite with a `+00:00` suffix that matches none of those formats, so
`playhouse.database.format_date_time()` silently falls back to returning the raw string
unparsed on read-back -- any arithmetic on that "datetime" field later raises a TypeError
(confirmed by a real bug caught in tests/test_event_processor.py during development).

The fix used throughout this codebase: always store NAIVE datetimes, always implicitly
interpreted as UTC. Use these two helpers everywhere a datetime is about to be written to
or compared against a peewee DateTimeField, instead of calling
`datetime.datetime.now(datetime.timezone.utc)` or `datetime.datetime.fromtimestamp(ts,
tz=...)` directly.
"""

from __future__ import annotations

import datetime


def utcnow() -> datetime.datetime:
    """Naive datetime representing the current UTC time -- safe to store in / compare
    against a peewee DateTimeField.
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def utc_from_timestamp(ts: float) -> datetime.datetime:
    """Naive datetime representing `ts` (a time.time()-style Unix timestamp) in UTC --
    safe to store in / compare against a peewee DateTimeField.
    """
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).replace(tzinfo=None)


def as_naive_utc(dt: datetime.datetime) -> datetime.datetime:
    """Normalizes any datetime (naive-assumed-UTC, or timezone-aware) down to a naive UTC
    datetime, for safe storage in / comparison against a peewee DateTimeField.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
