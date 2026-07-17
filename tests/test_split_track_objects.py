"""Tests for mirage.config.schema.split_track_objects -- the pure function deciding
which of a camera's objects.track words are real closed-vocab (COCO) labels vs. words
the routed detector doesn't know at all, and therefore need to be routed to the open-
vocabulary path instead of being a silently dead config value (TODO_FIX_LIST.md item 4's
original problem statement, now bridged via mirage/api/routers/config.py's
_sync_track_object_queries).
"""

from __future__ import annotations

from mirage.config.schema import split_track_objects

COCO_LABELS = {"person", "car", "dog", "cat", "bird"}


def test_all_coco_labels_are_closed_vocab():
    closed, open_vocab = split_track_objects(["person", "car"], COCO_LABELS)
    assert closed == ["person", "car"]
    assert open_vocab == []


def test_unknown_word_is_open_vocab():
    closed, open_vocab = split_track_objects(["animals"], COCO_LABELS)
    assert closed == []
    assert open_vocab == ["animals"]


def test_mixed_list_splits_correctly_and_preserves_order():
    closed, open_vocab = split_track_objects(["person", "animals", "car", "raccoon"], COCO_LABELS)
    assert closed == ["person", "car"]
    assert open_vocab == ["animals", "raccoon"]


def test_matching_is_case_insensitive():
    closed, open_vocab = split_track_objects(["Person", "DOG"], COCO_LABELS)
    assert closed == ["Person", "DOG"]
    assert open_vocab == []


def test_empty_track_list_returns_empty_lists():
    closed, open_vocab = split_track_objects([], COCO_LABELS)
    assert closed == []
    assert open_vocab == []
