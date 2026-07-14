"""Tests for mirage/openvocab/gating.py -- the perceptual-hash dedup gate (Gate 2) that
decides whether a tracked object's crop should actually be sent to the slow OWLv2
process again, or skipped because it hasn't changed enough since the last check.
"""

from __future__ import annotations

import numpy as np

from mirage.openvocab.gating import OpenVocabGate, average_hash, hamming_distance


def _solid_crop(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _half_split_crop() -> np.ndarray:
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop[:, 32:] = 255
    return crop


def test_identical_crops_have_zero_hamming_distance():
    crop = _solid_crop(128)
    assert hamming_distance(average_hash(crop), average_hash(crop)) == 0


def test_visually_distinct_crops_have_nonzero_hamming_distance():
    a = _solid_crop(10)
    b = _half_split_crop()
    assert hamming_distance(average_hash(a), average_hash(b)) > 0


def test_first_check_for_an_object_always_dispatches():
    gate = OpenVocabGate()
    assert gate.should_dispatch("obj1", _solid_crop(100), now=0.0) is True


def test_unchanged_crop_does_not_redispatch_before_recheck_interval():
    gate = OpenVocabGate(min_recheck_interval=100.0)
    crop = _solid_crop(100)
    assert gate.should_dispatch("obj1", crop, now=0.0) is True
    gate.mark_result_received("obj1")
    assert gate.should_dispatch("obj1", crop, now=1.0) is False


def test_changed_crop_redispatches_immediately():
    gate = OpenVocabGate(hamming_threshold=5)
    assert gate.should_dispatch("obj1", _solid_crop(10), now=0.0) is True
    gate.mark_result_received("obj1")
    # a visually very different crop should clear the hamming threshold even though the
    # recheck interval hasn't elapsed
    assert gate.should_dispatch("obj1", _half_split_crop(), now=0.1) is True


def test_unchanged_crop_redispatches_after_recheck_interval_elapses():
    gate = OpenVocabGate(min_recheck_interval=1.0)
    crop = _solid_crop(100)
    assert gate.should_dispatch("obj1", crop, now=0.0) is True
    gate.mark_result_received("obj1")
    assert gate.should_dispatch("obj1", crop, now=0.5) is False
    assert gate.should_dispatch("obj1", crop, now=2.0) is True


def test_in_flight_request_blocks_a_second_dispatch_for_the_same_object():
    gate = OpenVocabGate()
    crop = _solid_crop(100)
    assert gate.should_dispatch("obj1", crop, now=0.0) is True
    # no mark_result_received() yet -- a request for obj1 is still outstanding
    assert gate.should_dispatch("obj1", _half_split_crop(), now=0.1) is False


def test_mark_result_received_allows_a_new_dispatch():
    gate = OpenVocabGate(hamming_threshold=5)
    assert gate.should_dispatch("obj1", _solid_crop(10), now=0.0) is True
    gate.mark_result_received("obj1")
    assert gate.should_dispatch("obj1", _half_split_crop(), now=0.1) is True


def test_different_objects_are_gated_independently():
    gate = OpenVocabGate(min_recheck_interval=100.0)
    crop = _solid_crop(100)
    assert gate.should_dispatch("obj1", crop, now=0.0) is True
    gate.mark_result_received("obj1")
    # obj2 has never been checked, so it dispatches even though obj1's identical crop
    # would not
    assert gate.should_dispatch("obj2", crop, now=0.1) is True


def test_forget_clears_all_state_for_an_object():
    gate = OpenVocabGate(min_recheck_interval=100.0)
    crop = _solid_crop(100)
    gate.should_dispatch("obj1", crop, now=0.0)
    gate.mark_result_received("obj1")
    gate.forget("obj1")
    # after forgetting, this object is treated as brand-new again
    assert gate.should_dispatch("obj1", crop, now=0.1) is True
