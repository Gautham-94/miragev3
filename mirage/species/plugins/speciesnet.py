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
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from mirage.config.schema import SpeciesModelConfig
from mirage.species.api import SpeciesClassification, SpeciesClassifierApi

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

        logger.info(
            "species: launching SpeciesNet worker (venv=%s, model=%s)",
            model_config.venv_python_path, model_config.model_name,
        )
        self.process = subprocess.Popen(
            [model_config.venv_python_path, _WORKER_SCRIPT, model_config.model_name, str(model_config.confidence_threshold)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

        ready_line = self._readline_with_timeout(_READY_TIMEOUT_SECONDS)
        if ready_line is None or not json.loads(ready_line).get("ready"):
            stderr_tail = self.process.stderr.read() if self.process.stderr else ""
            self.process.kill()
            raise RuntimeError(
                f"SpeciesNet worker failed to start (venv_python_path={model_config.venv_python_path!r}); "
                f"is the isolated venv set up? (see this module's docstring). stderr: {stderr_tail[-2000:]}"
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
            logger.error("species: SpeciesNet worker process exited (code=%s), cannot classify", self.process.returncode)
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
