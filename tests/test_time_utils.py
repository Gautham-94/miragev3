from __future__ import annotations

import datetime

from mirage.util.time import as_naive_utc, utc_from_timestamp, utcnow


def test_utcnow_is_naive():
    result = utcnow()
    assert result.tzinfo is None


def test_utcnow_is_approximately_now():
    before = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    result = utcnow()
    after = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    assert before <= result <= after


def test_utc_from_timestamp_is_naive():
    result = utc_from_timestamp(1700000000.0)
    assert result.tzinfo is None


def test_utc_from_timestamp_matches_expected_utc_value():
    result = utc_from_timestamp(0.0)
    assert result == datetime.datetime(1970, 1, 1, 0, 0, 0)


def test_as_naive_utc_passes_through_naive_datetime_unchanged():
    naive = datetime.datetime(2026, 1, 1, 12, 0, 0)
    assert as_naive_utc(naive) == naive
    assert as_naive_utc(naive).tzinfo is None


def test_as_naive_utc_strips_tzinfo_from_aware_datetime():
    aware = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    result = as_naive_utc(aware)
    assert result.tzinfo is None
    assert result == datetime.datetime(2026, 1, 1, 12, 0, 0)


def test_as_naive_utc_converts_non_utc_timezone_correctly():
    plus_five = datetime.timezone(datetime.timedelta(hours=5))
    aware = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=plus_five)  # 07:00 UTC
    result = as_naive_utc(aware)
    assert result == datetime.datetime(2026, 1, 1, 7, 0, 0)
    assert result.tzinfo is None
