"""Tests for mirage.util.shm -- validates the ring buffer survives the exact
cross-process attach/detach pattern the real system uses: one process creates a segment,
a *different* process attaches to it, writes, and the creating process reads it back after
the writer process has fully exited. This is precisely the scenario the resource_tracker
bug (python/cpython#82300) would break if UntrackedSharedMemory were wrong.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from mirage.util.shm import (
    SharedMemoryFrameManager,
    calculate_shm_ring_depth,
    detector_input_shm_name,
    detector_output_shm_name,
    frame_name,
    preallocate_ring,
    teardown_ring,
    yuv_frame_shape,
    yuv_frame_size,
)


def test_yuv_frame_size_formula():
    # width * height * 1.5, exact YUV420 planar formula (spec section 1.2.3 / 1.5.3).
    assert yuv_frame_size(640, 360) == int(640 * 360 * 1.5)
    assert yuv_frame_size(640, 360) == 345_600


def test_yuv_frame_shape():
    assert yuv_frame_shape(640, 360) == (540, 640)


def test_frame_name_convention():
    # Names are now a short derived key (see _shm_key in mirage/util/shm.py -- macOS
    # caps POSIX shm names at 30 chars, and real camera names like
    # "hikvision_ds_2cd1023g0e_i" overflow that once affixes are added), not the raw
    # camera name -- but they must still be deterministic (same input -> same output
    # every time, across processes) and vary by index/camera exactly like the old
    # convention did.
    assert frame_name("front_door", 0) == frame_name("front_door", 0)
    assert frame_name("front_door", 0) != frame_name("front_door", 12)
    assert frame_name("front_door", 0) != frame_name("back_door", 0)
    assert frame_name("front_door", 12).endswith("_frame12")


def test_detector_shm_names():
    assert detector_input_shm_name("cam1") == detector_input_shm_name("cam1")
    assert detector_input_shm_name("cam1") != detector_input_shm_name("cam2")
    assert detector_output_shm_name("cam1") == f"out-{detector_input_shm_name('cam1')}"


def test_shm_names_stay_under_macos_posix_length_limit_for_long_camera_names():
    # Regression test: this exact scenario (a real, longer-than-test-fixture camera
    # name) broke for real running python -m mirage against a wizard-added Hikvision
    # camera -- `OSError: [Errno 63] File name too long` -- before _shm_key existed.
    long_name = "hikvision_ds_2cd1023g0e_i"
    # macOS's actual limit, confirmed empirically: 30 chars succeeds, 31 fails.
    assert len(frame_name(long_name, 49)) <= 30
    assert len(detector_input_shm_name(long_name)) <= 30
    assert len(detector_output_shm_name(long_name)) <= 30

    even_longer_name = "a_very_extremely_long_camera_name_that_someone_might_actually_type"
    assert len(frame_name(even_longer_name, 49)) <= 30
    assert len(detector_input_shm_name(even_longer_name)) <= 30
    assert len(detector_output_shm_name(even_longer_name)) <= 30


def test_calculate_shm_ring_depth_caps_at_max():
    depth, min_mb = calculate_shm_ring_depth([(640, 360)], available_shm_mb=100_000, max_ring_depth=50)
    assert depth == 50


def test_calculate_shm_ring_depth_limited_by_memory():
    # Force a small memory budget so the memory-based cap kicks in below max_ring_depth.
    depth, min_mb = calculate_shm_ring_depth([(1920, 1080)], available_shm_mb=10, max_ring_depth=50)
    assert 1 <= depth < 50


def test_single_process_create_write_get_roundtrip():
    # NOTE: SharedMemory rounds the underlying segment up to the OS page size, so
    # `shm.buf` may be LARGER than the requested `size`. write(name, size) must return a
    # memoryview sliced down to exactly `size` so `buf[:] = read(size)`-style assignment
    # (as used in the real capture loop) always matches lengths exactly.
    fm = SharedMemoryFrameManager()
    name = "test_roundtrip_frame0"
    size = yuv_frame_size(64, 48)
    try:
        fm.create(name, size)
        buf = fm.write(name, size)
        assert buf is not None
        assert len(buf) == size
        payload = (bytes(range(256)) * (size // 256 + 1))[:size]
        buf[:] = payload
        fm.close(name)

        arr = fm.get(name, yuv_frame_shape(64, 48))
        assert arr is not None
        assert arr.nbytes == size
        assert bytes(arr.reshape(-1)[:256]) == bytes(range(256))
        fm.close(name)
    finally:
        fm.delete(name)


def test_delete_then_get_returns_none():
    fm = SharedMemoryFrameManager()
    name = "test_delete_frame0"
    fm.create(name, 1024)
    fm.delete(name)
    assert fm.get(name, (1024,)) is None


def test_create_fails_loudly_on_a_stale_smaller_segment_still_held_open():
    # Real-world trigger: a camera's detect resolution is increased (e.g. via the
    # add-camera wizard) and the pipeline is restarted, but some OTHER process from
    # before the change (an orphaned/ungracefully-killed capture or tracker process)
    # is STILL ALIVE and still holds its handle to the old, smaller-sized segment
    # open. On Windows, a named shared-memory mapping is reference-counted by open
    # handles and destroyed the instant the LAST one closes -- so this scenario only
    # reproduces with a second handle genuinely still open (fm_stale_holder below,
    # deliberately never closed in this test): closing every handle first would let
    # Windows tear the mapping down on its own, and the next create() would just
    # allocate fresh at the right size, never exercising the bug at all (this is
    # exactly what made an earlier, wrong version of this test pass even against the
    # unfixed code -- it called close() on the stale holder before the second
    # create(), which released Windows' only handle and silently made the "stale"
    # segment disappear before it could ever be reused).
    #
    # Before this fix, create()'s `except FileExistsError` branch just reattached to
    # the stale segment and returned `shm.buf[:size]` -- Python slicing CLAMPS rather
    # than raising when `size` exceeds the buffer's actual length, so this silently
    # handed back an undersized view. The bug then surfaced far away, the next time a
    # full-size frame was written into it: `ValueError: memoryview assignment:
    # lvalue and rvalue have different structures` in mirage/capture/capture.py's
    # `frame_buffer[:] = data`.
    #
    # The fix can't silently make this case actually work -- Windows has no way to
    # force-resize/replace a named mapping while another handle is open, full stop
    # (unlink() is a POSIX-only concept, a complete no-op on Windows). What it CAN do,
    # and what this test asserts, is turn "silently corrupt the next frame written"
    # into "fail immediately and explain why," which is what actually happened live:
    # this surfaced as a repeating ValueError crash-loop that took real investigation
    # to trace back to leftover orphaned processes -- a clear RuntimeError naming the
    # segment and explaining the cause would have pointed straight at the fix
    # (stop the leftover process) instead.
    fm_stale_holder = SharedMemoryFrameManager()
    name = "test_stale_resize_frame0"
    small_size = yuv_frame_size(64, 48)   # e.g. an old 640x480-class detect resolution
    large_size = yuv_frame_size(256, 192)  # e.g. a new, larger detect resolution
    fm_stale_holder.create(name, small_size)  # handle deliberately kept open throughout
    try:
        fm2 = SharedMemoryFrameManager()
        with pytest.raises(RuntimeError, match=name):
            fm2.create(name, large_size)
    finally:
        fm_stale_holder.delete(name)


def test_cleanup_unlinks_all_created_segments():
    fm = SharedMemoryFrameManager()
    names = [f"test_cleanup_frame{i}" for i in range(5)]
    for n in names:
        fm.create(n, 128)
    fm.cleanup()

    fm2 = SharedMemoryFrameManager()
    for n in names:
        assert fm2.get(n, (128,)) is None


def test_preallocate_and_teardown_ring():
    fm = SharedMemoryFrameManager()
    camera = "ring_test_cam"
    preallocate_ring(fm, camera, ring_depth=4, width=32, height=24)
    for i in range(4):
        arr = fm.get(frame_name(camera, i), yuv_frame_shape(32, 24))
        assert arr is not None
        assert arr.shape == yuv_frame_shape(32, 24)
    teardown_ring(fm, camera, ring_depth=4)
    for i in range(4):
        assert fm.get(frame_name(camera, i), yuv_frame_shape(32, 24)) is None


# ---------------------------------------------------------------------------------------
# Cross-process test: this is the scenario the resource_tracker workaround exists for.
# ---------------------------------------------------------------------------------------


def _child_write_and_exit(name: str, size: int, value: int, ready_event, done_event) -> None:
    """Runs in a child process: attaches to an EXISTING segment (does not create it),
    writes a known value, signals ready, then exits -- while the parent still needs the
    segment. If resource_tracker unlinked the segment on this process's exit, the parent's
    subsequent read would fail with FileNotFoundError.
    """
    fm = SharedMemoryFrameManager()
    buf = fm.write(name, size)
    buf[:] = bytes([value]) * size
    fm.close(name)
    ready_event.set()
    done_event.wait(timeout=5)


def test_segment_survives_child_process_exit():
    fm = SharedMemoryFrameManager()
    name = "test_cross_process_frame0"
    size = 256
    fm.create(name, size)
    try:
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        done = ctx.Event()
        p = ctx.Process(target=_child_write_and_exit, args=(name, size, 0xAB, ready, done))
        p.start()
        assert ready.wait(timeout=10), "child process never signaled ready"
        done.set()
        p.join(timeout=10)
        assert p.exitcode == 0

        # Parent reads AFTER the child that wrote (and never created) the segment has
        # fully exited. This is exactly the failure mode the resource_tracker bug causes
        # if UntrackedSharedMemory's monkeypatch weren't applied.
        arr = fm.get(name, (size,))
        assert arr is not None
        assert bytes(arr[:10]) == bytes([0xAB]) * 10
    finally:
        fm.delete(name)


def test_multiple_independent_frame_managers_share_segment():
    """Two separate SharedMemoryFrameManager instances (simulating two different
    processes' local caches) must be able to see the same underlying segment by name."""
    fm_a = SharedMemoryFrameManager()
    fm_b = SharedMemoryFrameManager()
    name = "test_shared_between_managers"
    size = 64
    fm_a.create(name, size)
    try:
        buf = fm_a.write(name, size)
        buf[:4] = b"\x01\x02\x03\x04"
        fm_a.close(name)

        arr = fm_b.get(name, (size,))
        assert arr is not None
        assert bytes(arr[:4]) == b"\x01\x02\x03\x04"
        fm_b.close(name)
    finally:
        fm_a.delete(name)
