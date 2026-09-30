"""SpeciesNet backend -- Google/CalTech's camera-trap species classifier
(https://github.com/google/cameratrapai), via the real `speciesnet` PyPI package.

Runs in a SEPARATE Python venv/subprocess (mirage/species/speciesnet_worker.py), not
imported directly into mirage's own venv -- the `speciesnet` package's transitive deps
(mainly yolov5/ultralytics/sahi/roboflow, since SpeciesNet's own detector is
YOLOv5-based) require numpy>=2 and the GUI `opencv-python` package. Both conflict
irreconcilably with this repo's pinned norfair (numpy<2.0 required) and
opencv-python-headless: installing `speciesnet` into the main venv was tried and
verified to produce real, unresolvable pip dependency conflicts (not just warnings --
`pip install speciesnet` demands opencv-python, which collides at the module level with
opencv-python-headless). Isolating it in its own venv (see SpeciesModelConfig.
venv_python_path's docstring for the expected setup:
`python3 -m venv .venv-speciesnet && .venv-speciesnet/bin/pip install speciesnet`) keeps
mirage's own detection/tracking pipeline's pinned versions completely untouched.

IPC with the worker subprocess is a simple length-implicit (newline-delimited)
JSON-over-stdin/stdout protocol, NOT mp.Queue/SHM -- those require the same interpreter/
site-packages on both ends to unpickle correctly, which doesn't hold across a venv
boundary. This is a lower-frequency call site than detection (species classification
fires once per Event's lifetime, see mirage.species.dispatcher's own docstring), so the
per-call JSON+base64 serialization overhead is a non-issue.

Label format: SpeciesNet's classifier returns raw label strings shaped
`uuid;class;order;family;genus;species;common_name` (7 semicolon-separated fields --
see speciesnet.taxonomy_utils.get_ancestor_at_level's own docstring). The worker script
parses this and applies mirage's own confidence_threshold (SpeciesNet has no built-in
cutoff) before reporting a real species name -- see speciesnet_worker.py's own
_build_response for the exact rules (below-threshold, or one of SpeciesNet's own
"blank"/"animal"/"human"/"vehicle"/"unknown" pseudo-classes with no real species/genus,
both map to a "no confident result" SpeciesClassification).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

from mirage.config.schema import SpeciesModelConfig
from mirage.species.api import SpeciesClassification, SpeciesClassifierApi
from mirage.util.proc import windows_no_console_flags

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = str(Path(__file__).resolve().parent.parent / "speciesnet_worker.py")

# Model load (first run may also download weights via kagglehub) can genuinely take
# minutes on a slow connection -- this is a one-time startup cost paid once per
# SpeciesProcess lifetime, not per-classification, so a generous timeout is safe here.
_READY_TIMEOUT_SECONDS = 600.0
# Steady-state per-crop classification is fast (~seconds, see this plugin's own
# benchmarking) -- a stuck/hung worker on a single request should surface as a failed
# classification for that one event, not silently block the whole species pipeline.
_REQUEST_TIMEOUT_SECONDS = 30.0


class SpeciesNetClassifier(SpeciesClassifierApi):
    type_key = "speciesnet"

    def __init__(self, model_config: SpeciesModelConfig) -> None:
        self.model_config = model_config
        self._lock = threading.Lock()
        self._request_counter = 0

        # In a packaged desktop build, SpeciesModelConfig.venv_python_path's own default
        # (".venv-speciesnet/bin/python", a dev-tree-only path -- see that field's own
        # docstring) is never meaningful: no such venv exists on a client machine, only
        # the bundled packaging/speciesnet_worker.spec exe does. Nothing else in this
        # codebase ever overrides that default for a frozen build (confirmed live: a
        # fresh packaged install with species classification enabled tried to launch
        # ".venv-speciesnet/bin/python" and failed immediately) -- so the bundled worker
        # takes priority here whenever this is a frozen build that actually has one,
        # regardless of what's configured. A dev-tree run (paths.is_frozen() False) is
        # completely unaffected; an explicit non-default venv_python_path is only ever
        # meaningful in dev anyway, since the packaged installer has no UI to set it.
        from mirage.desktop import paths as desktop_paths

        frozen_worker = desktop_paths.speciesnet_worker_path() if desktop_paths.is_frozen() else None

        # subprocess.Popen's argv[0] must be resolved to an absolute path here -- on
        # Windows, _winapi.CreateProcess does NOT search a relative path against the
        # current working directory the way POSIX execve does (confirmed live:
        # Popen([".venv-speciesnet/Scripts/python.exe", ...]) raises
        # FileNotFoundError: [WinError 2] even when os.path.exists() on that exact
        # relative path returns True from the same cwd; the identical call with
        # os.path.abspath() applied succeeds immediately). abspath() is a no-op for an
        # already-absolute path and resolves correctly on POSIX too, so this is safe
        # for every platform/config combination, not just the Windows-relative-path one
        # that broke -- callers (including the documented
        # ".venv-speciesnet/bin/python"-style default) need zero config changes.
        venv_python_path = (
            str(frozen_worker) if frozen_worker is not None else os.path.abspath(model_config.venv_python_path)
        )

        logger.info(
            "species: launching SpeciesNet worker (venv=%s, model=%s)",
            venv_python_path, model_config.model_name,
        )
        # A PyInstaller-frozen standalone worker exe (packaging/speciesnet_worker.spec)
        # already has speciesnet_worker.py baked in -- passing _WORKER_SCRIPT as an
        # argument to it would be received as a bogus model_name instead. Detected by
        # filename convention (the frozen build is always named exactly this) rather
        # than a config flag, so existing `.venv-speciesnet/bin/python`-style configs
        # need zero changes.
        is_frozen_worker = Path(venv_python_path).stem == "mirage_speciesnet_worker"
        argv = (
            [venv_python_path, model_config.model_name, str(model_config.confidence_threshold)]
            if is_frozen_worker
            else [venv_python_path, _WORKER_SCRIPT, model_config.model_name, str(model_config.confidence_threshold)]
        )
        # stderr is deliberately a real temp file, NOT subprocess.PIPE -- see
        # mirage/util/proc.py's own module docstring for the identical class of bug
        # this already bit production once (a PIPE-based subprocess call wedging the
        # recording-maintainer thread for 20+ minutes). The mechanism here is slightly
        # different (that one was about Popen.wait()/communicate() hanging; this one is
        # a classic stdout/stderr PIPE deadlock: this class only ever actively drains
        # stdout via _readline_with_timeout, so if the worker -- e.g. kagglehub/tqdm
        # progress output during model load -- writes enough to stderr to fill the OS
        # pipe buffer, the child blocks on its own stderr write and stdout never
        # arrives either, even though the process is very much alive, just stuck at
        # 0% CPU forever) -- but the fix is the same one already established in this
        # codebase: don't use PIPE for a stream nothing is reading in real time.
        # Unlike stdout (needed live, per-request, for the whole classifier lifetime),
        # nothing here needs stderr AS IT ARRIVES -- only its final contents, and only
        # when something has already gone wrong -- so a file that can be seeked back to
        # 0 and read at that point is strictly better than a pipe for this stream.
        self._stderr_file = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr_file, text=True, bufsize=1,
            creationflags=windows_no_console_flags(),
        )

        ready_line = self._readline_with_timeout(_READY_TIMEOUT_SECONDS)
        if ready_line is None or not json.loads(ready_line).get("ready"):
            self.process.kill()
            self.process.wait()  # bounded: no pipe involved, just the process handle
            self._stderr_file.seek(0)
            stderr_tail = self._stderr_file.read().decode(errors="replace")
            self._stderr_file.close()
            raise RuntimeError(
                # The actually-launched path (frozen worker or resolved venv, per the
                # frozen_worker logic above) -- NOT the raw model_config.venv_python_path
                # field, which stays whatever's configured (e.g. the dev-tree default)
                # even when a frozen build overrode it; that mismatch made an earlier
                # version of this message actively misleading while debugging this exact
                # class of failure.
                f"SpeciesNet worker failed to start (venv_python_path={venv_python_path!r}); "
                f"is the isolated venv set up? (see this module's docstring). "
                f"returncode={self.process.returncode!r} ready_line={ready_line!r} stderr: {stderr_tail[-2000:]}"
            )
        logger.info("species: SpeciesNet worker ready")

    def _readline_with_timeout(self, timeout: float) -> str | None:
        """subprocess.stdout has no built-in readline timeout -- runs the blocking read
        in a helper thread and joins with a deadline, since a hung/crashed worker must
        never wedge the caller (SpeciesProcess's own request loop) forever.
        """
        result: list[str | None] = [None]

        def _read():
            result[0] = self.process.stdout.readline() if self.process.stdout else None

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive() or not result[0]:
            return None
        return result[0].strip()

    def classify(self, crop_rgb: np.ndarray) -> SpeciesClassification:
        if self.process.poll() is not None:
            # Worker died -- report as "no confident result" rather than raising, so one
            # dead worker doesn't repeatedly crash SpeciesProcess's request loop (which
            # already converts an uncaught exception here into species_status="failed"
            # per-event, but a permanently-dead worker would fail EVERY subsequent event
            # rather than just this one; a clear log line makes the real cause visible).
            # seek(0) first -- this branch can run on every subsequent classify() call
            # once the worker has died (poll() keeps returning the same exit code), and
            # a prior read here would otherwise leave the file position at EOF.
            self._stderr_file.seek(0)
            stderr_tail = self._stderr_file.read().decode(errors="replace")
            logger.error(
                "species: SpeciesNet worker process exited (code=%s), cannot classify. stderr: %s",
                self.process.returncode, stderr_tail[-2000:],
            )
            return SpeciesClassification(species_common=None, species_scientific=None, confidence=None, taxonomy=None)

        with self._lock:
            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                return SpeciesClassification(species_common=None, species_scientific=None, confidence=None, taxonomy=None)
            crop_b64 = base64.b64encode(encoded.tobytes()).decode()

            request = json.dumps({"request_id": request_id, "crop_b64": crop_b64})
            self.process.stdin.write(request + "\n")
            self.process.stdin.flush()

            response_line = self._readline_with_timeout(_REQUEST_TIMEOUT_SECONDS)
            if response_line is None:
                logger.error("species: SpeciesNet worker timed out classifying request %s", request_id)
                return SpeciesClassification(species_common=None, species_scientific=None, confidence=None, taxonomy=None)

            response = json.loads(response_line)

        if response.get("error"):
            raise RuntimeError(f"SpeciesNet worker classification error: {response['error']}")

        return SpeciesClassification(
            species_common=response.get("species_common"),
            species_scientific=response.get("species_scientific"),
            confidence=response.get("confidence"),
            taxonomy=response.get("taxonomy"),
        )

    def __del__(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
        stderr_file = getattr(self, "_stderr_file", None)
        if stderr_file is not None:
            stderr_file.close()
