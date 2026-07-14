"""Shared-memory frame ring buffer, safe for use across independent processes.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 1.5 and Appendix B.1.

Works around a real CPython bug (https://github.com/python/cpython/issues/82300): the
multiprocessing.resource_tracker unlinks a SharedMemory segment as soon as ANY process that
opened it exits -- even if other processes still hold it open. Since capture, tracker, and
detector processes all attach to the *same* named segments independently, the default
behavior would destroy frames out from under still-running consumers. We disable tracking
for these segments and manage the unlink lifecycle ourselves instead.
"""

from __future__ import annotations

import atexit
import hashlib
import re
import threading
from abc import ABC, abstractmethod
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from typing import Optional

import numpy as np


class UntrackedSharedMemory(_mpshm.SharedMemory):
    """A SharedMemory that does not register itself with multiprocessing.resource_tracker.

    Pass track=True to opt back into normal tracked behavior (not used in this codebase,
    kept for parity with the upstream pattern this is based on).
    """

    __lock = threading.Lock()

    def __init__(
        self,
        name: Optional[str] = None,
        create: bool = False,
        size: int = 0,
        *,
        track: bool = False,
    ) -> None:
        self._track = track
        if track:
            super().__init__(name=name, create=create, size=size)
            return

        with self.__lock:
            orig_register = _mprt.register
            _mprt.register = self.__noop_register
            try:
                super().__init__(name=name, create=create, size=size)
            finally:
                _mprt.register = orig_register

    @staticmethod
    def __noop_register(*args, **kwargs) -> None:
        return

    def unlink(self) -> None:
        if _mpshm._USE_POSIX and self._name:
            _mpshm._posixshmem.shm_unlink(self._name)
            if self._track:
                _mprt.unregister(self._name, "shared_memory")


class FrameManager(ABC):
    @abstractmethod
    def create(self, name: str, size: int) -> memoryview: ...

    @abstractmethod
    def write(self, name: str, size: int) -> Optional[memoryview]: ...

    @abstractmethod
    def get(self, name: str, shape: tuple, dtype=np.uint8) -> Optional[np.ndarray]: ...

    @abstractmethod
    def close(self, name: str) -> None: ...

    @abstractmethod
    def delete(self, name: str) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...


class SharedMemoryFrameManager(FrameManager):
    """Each process instantiates its own; the dict below is a process-local cache of
    already-opened handles, keyed by segment name, so repeated get()/write() calls for the
    same frame name don't reopen the POSIX object every time.

    Two CPython/`SharedMemory` behaviors this class has to work around to be safe:

    1. The OS rounds an allocation up to the page size, so `shm.buf` (and `shm.size`) can
       be LARGER than the size originally requested by `create()`. If callers
       sliced-assign into the raw `shm.buf` directly (e.g. a capture loop doing
       `buf[:] = ffmpeg.stdout.read(frame_size)`), that assignment would only succeed when
       `frame_size` happens to be an exact multiple of the page size -- essentially never
       for real YUV420 frame dimensions. `write()`/`get()` therefore always return a view
       sliced down to exactly the caller-specified size, never the raw, possibly larger,
       page-rounded buffer.
    2. `mmap.close()` (invoked by `SharedMemory.close()`) raises `BufferError: cannot
       close exported pointers exist` if any memoryview/ndarray obtained from `shm.buf`
       is still alive when close() is called -- which a naive caller could easily trigger
       by holding on to the return value of a previous `write()`/`get()` call past a
       `close()`/`delete()` on that same name. Rather than push that lifetime discipline
       onto every caller, this class remembers the last view it handed out per segment
       name and explicitly releases it before closing/unlinking, so callers only need to
       stop using an old view after requesting a new one (i.e. normal variable
       reassignment), not manually call `.release()`.

       The same BufferError can also surface at *interpreter shutdown* if a process exits
       (crash, uncaught exception, SIGTERM) without ever calling close()/delete(): CPython
       does not guarantee GC finalizes the outstanding memoryview before the SharedMemory
       object's own __del__ runs, so relying on garbage collection alone is not reliable
       even though this class always releases views on its own close()/delete()/cleanup()
       calls. To make normal process-exit paths not spam "Exception ignored in..." noise,
       this class registers its own cleanup() with atexit at construction time, which runs
       deterministically before that unordered finalization pass.
    """

    def __init__(self) -> None:
        self.shm_store: dict[str, UntrackedSharedMemory] = {}
        self._last_view: dict[str, object] = {}
        atexit.register(self._atexit_release_views_only)

    def _atexit_release_views_only(self) -> None:
        """Release outstanding views WITHOUT closing/unlinking the segments themselves --
        this process may not be the one responsible for the segment's lifecycle (e.g. a
        capture/tracker process should never unlink a ring buffer slot on exit; only the
        maintainer that created it should). This only prevents the BufferError from
        surfacing during Python's own shutdown finalization; it does not detach handles.
        """
        for name in list(self._last_view.keys()):
            self._release_last_view(name)

    def _release_last_view(self, name: str) -> None:
        view = self._last_view.pop(name, None)
        if view is not None:
            try:
                view.release()
            except AttributeError:
                pass  # a bare np.ndarray view has no .release(); its buffer does, via base

    def create(self, name: str, size: int) -> memoryview:
        try:
            shm = UntrackedSharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            shm = UntrackedSharedMemory(name=name)
        self.shm_store[name] = shm
        view = shm.buf[:size]
        self._last_view[name] = view
        return view

    def write(self, name: str, size: int) -> Optional[memoryview]:
        try:
            shm = self.shm_store.get(name) or UntrackedSharedMemory(name=name)
            self.shm_store[name] = shm
            self._release_last_view(name)
            view = shm.buf[:size]
            self._last_view[name] = view
            return view
        except FileNotFoundError:
            return None

    def get(self, name: str, shape: tuple, dtype=np.uint8) -> Optional[np.ndarray]:
        try:
            shm = self.shm_store.get(name) or UntrackedSharedMemory(name=name)
            self.shm_store[name] = shm
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self._release_last_view(name)
            view = shm.buf[:nbytes]
            self._last_view[name] = view
            return np.ndarray(shape, dtype=dtype, buffer=view)
        except FileNotFoundError:
            return None

    def close(self, name: str) -> None:
        """Detach only. Does NOT unlink -- the segment persists for the next ring lap."""
        self._release_last_view(name)
        shm = self.shm_store.pop(name, None)
        if shm is not None:
            shm.close()

    def delete(self, name: str) -> None:
        """Detach AND unlink -- destroys the underlying shared memory object. Only call
        this from the process responsible for that segment's full lifecycle (usually the
        component that originally created it), or when a camera is being torn down.
        """
        self._release_last_view(name)
        shm = self.shm_store.pop(name, None)
        if shm is None:
            try:
                shm = UntrackedSharedMemory(name=name)
            except FileNotFoundError:
                return
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass

    def cleanup(self) -> None:
        for name in list(self.shm_store.keys()):
            self._release_last_view(name)
        for shm in list(self.shm_store.values()):
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self.shm_store.clear()
        self._last_view.clear()


def yuv_frame_size(width: int, height: int) -> int:
    """Exact YUV420 planar frame byte size: Y plane (w*h) + U+V planes combined (w*h/2)."""
    return width * height * 3 // 2


def yuv_frame_shape(width: int, height: int) -> tuple[int, int]:
    """Numpy shape for a YUV420 planar buffer, laid out as one 2D uint8 array."""
    return height * 3 // 2, width


def _shm_key(camera_name: str) -> str:
    """Derives a short, stable, filesystem-safe key for a camera name, for use in POSIX
    shared-memory segment names.

    Real camera names (as opposed to the short test-fixture ones like "test_cam" used
    everywhere in this codebase's own tests) can be much longer -- e.g.
    "hikvision_ds_2cd1023g0e_i" -- and macOS caps a POSIX shm name at exactly 30
    characters total (confirmed empirically: 30-char names succeed, 31-char names raise
    `OSError: [Errno 63] File name too long`; Linux's limit is far more generous, ~255,
    but this codebase targets macOS for development per its own path-handling
    conventions elsewhere, so the tighter limit governs). Once you add the "_frame12" or
    "out-" affixes this module's own naming scheme appends, even a moderately-long real
    camera name overflows that budget -- this was invisible in every test in this
    codebase (all use short synthetic names) and only surfaced adding a real camera
    through the add-camera wizard.

    Truncating the raw name alone isn't enough on its own (two different long names
    could truncate to the same prefix and collide), so this combines a short prefix of
    the sanitized name (for human-debuggability, e.g. in `ps`/`lsof` output) with a short
    hash suffix (for collision safety) -- 16 characters total, comfortably under the
    30-char ceiling even with the longest affix ("out-" + key, or key + "_frame" + 2
    digits) added on top.
    """
    safe = re.sub(r"[^a-zA-Z0-9]", "", camera_name)[:8]
    digest = hashlib.sha1(camera_name.encode()).hexdigest()[:8]
    return f"{safe}{digest}"


def frame_name(camera_name: str, index: int) -> str:
    """Naming convention for ring buffer slots (spec section 1.5.2): '<camera>_frame<i>'."""
    return f"{_shm_key(camera_name)}_frame{index}"


def detector_input_shm_name(camera_name: str) -> str:
    """Section 3.2: input tensor SHM segment name for a camera's detection request."""
    return _shm_key(camera_name)


def detector_output_shm_name(camera_name: str) -> str:
    """Section 3.2: output detections SHM segment name, 'out-<camera>'."""
    return f"out-{_shm_key(camera_name)}"


def preallocate_ring(frame_manager: SharedMemoryFrameManager, camera_name: str, ring_depth: int, width: int, height: int) -> None:
    """Pre-create all N named SHM segments for a camera before spawning its capture/tracker
    processes (spec section 1.5.3): the maintainer must do this so child processes can
    simply attach by name.
    """
    size = yuv_frame_size(width, height)
    for i in range(ring_depth):
        frame_manager.create(frame_name(camera_name, i), size)


def teardown_ring(frame_manager: SharedMemoryFrameManager, camera_name: str, ring_depth: int) -> None:
    for i in range(ring_depth):
        frame_manager.delete(frame_name(camera_name, i))


def calculate_shm_ring_depth(
    camera_frame_dims: list[tuple[int, int]],
    available_shm_mb: float,
    max_ring_depth: int = 50,
    per_camera_overhead_bytes: int = 270_480,
) -> tuple[int, float]:
    """Spec section 1.5.3. Returns (ring_depth, recommended_min_shm_mb).

    camera_frame_dims: list of (detect_width, detect_height) for every enabled camera.
    """
    if not camera_frame_dims:
        return max_ring_depth, 0.0

    per_camera_frame_mb = 0.0
    for width, height in camera_frame_dims:
        per_camera_frame_mb += (width * height * 1.5 + per_camera_overhead_bytes) / (1024**2)

    ring_depth = min(max_ring_depth, max(1, int(available_shm_mb / per_camera_frame_mb)))
    recommended_min_shm_mb = per_camera_frame_mb * 20
    return ring_depth, recommended_min_shm_mb
