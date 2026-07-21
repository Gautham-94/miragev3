"""Standalone SpeciesNet worker -- runs under a SEPARATE Python venv from the rest of
mirage (see mirage/species/plugins/speciesnet.py's module docstring for why: the real
`speciesnet` PyPI package's transitive deps, mainly yolov5/ultralytics/sahi/roboflow,
require numpy>=2 and the GUI `opencv-python` package, which conflict irreconcilably
with this repo's pinned norfair (numpy<2) and opencv-python-headless).

Deliberately a STANDALONE script, not a module inside the `mirage` package -- it is
launched via `subprocess.Popen([venv_python_path, THIS_FILE], ...)` from
OnnxMegadetectorDetector's sibling, SpeciesNetClassifier (mirage/species/plugins/
speciesnet.py), running under an interpreter that does NOT have mirage installed. It
must import nothing from mirage and depend on nothing beyond the stdlib + whatever
`speciesnet` itself pulls in.

Protocol: length-prefixed JSON lines over stdin/stdout (NOT mp.Queue -- that requires
pickling compatible with the SAME interpreter/site-packages on both ends, which doesn't
hold across a venv boundary). Each request/response is a single line of JSON, so a
simple readline()/write() loop suffices -- no explicit length prefix needed since JSON
crop payloads are base64-encoded inline and neither contain raw newlines.

Request:  {"request_id": str, "crop_b64": str (base64 JPEG bytes)}
Response: {"request_id": str, "species_common": str|null, "species_scientific": str|null,
           "confidence": float|null, "taxonomy": dict|null, "error": str|null}

A `None`/null species_common+species_scientific (with no "error") means "ran fine, just
no confident result" (below confidence_threshold, or SpeciesNet's own "blank"/"unknown"
pseudo-classes) -- the caller (SpeciesNetClassifier.classify) maps this to
SpeciesClassification's own "no confident result" convention. An "error" key present
means classification itself raised; the caller surfaces this as species_status="failed".
"""

from __future__ import annotations

import base64
import json
import sys


def _parse_label(label: str) -> dict | None:
    """SpeciesNet's label convention: `uuid;class;order;family;genus;species;common_name`
    (7 semicolon-separated fields -- see speciesnet.taxonomy_utils.get_ancestor_at_level's
    own docstring, which this mirrors). Returns None for a label that doesn't parse (this
    would indicate a SpeciesNet package version mismatch, not a normal runtime outcome).
    """
    parts = label.split(";")
    if len(parts) != 7:
        return None
    _uuid, klass, order, family, genus, species, common_name = parts
    return {
        "class": klass or None,
        "order": order or None,
        "family": family or None,
        "genus": genus or None,
        "species": species or None,
        "common_name": common_name or None,
    }


def _run_server(model_name: str, confidence_threshold: float) -> None:
    import numpy as np
    import PIL.Image
    from speciesnet.classifier import SpeciesNetClassifier

    classifier = SpeciesNetClassifier(model_name)

    # Ready signal -- lets the parent (SpeciesNetClassifier.__init__, blocking on model
    # load which can take real wall-clock time on first run/model download) know
    # construction actually succeeded, distinct from "still loading."
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        request_id = request["request_id"]
        try:
            crop_bytes = base64.b64decode(request["crop_b64"])
            img = PIL.Image.open(__import__("io").BytesIO(crop_bytes)).convert("RGB")
            preprocessed = classifier.preprocess(img, bboxes=None, resize=True)
            result = classifier.predict(request_id, preprocessed)

            response = _build_response(request_id, result, confidence_threshold)
        except Exception as e:  # noqa: BLE001 -- must never crash the worker loop; report per-request
            response = {
                "request_id": request_id, "species_common": None, "species_scientific": None,
                "confidence": None, "taxonomy": None, "error": str(e),
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


def _build_response(request_id: str, result: dict, confidence_threshold: float) -> dict:
    classifications = result.get("classifications")
    if not classifications or not classifications.get("classes"):
        return {
            "request_id": request_id, "species_common": None, "species_scientific": None,
            "confidence": None, "taxonomy": None, "error": None,
        }

    top_label = classifications["classes"][0]
    top_score = classifications["scores"][0]

    if top_score < confidence_threshold:
        return {
            "request_id": request_id, "species_common": None, "species_scientific": None,
            "confidence": None, "taxonomy": None, "error": None,
        }

    taxonomy = _parse_label(top_label)
    # SpeciesNet's "blank"/"animal"/"human"/"vehicle"/"unknown" pseudo-classes (see
    # speciesnet.constants.Classification) all parse to an all-empty taxonomy (no
    # class/order/family/genus/species) -- these carry a common_name ("blank", "human",
    # ...) but no real species identification, so treat them as "no confident result"
    # too rather than reporting a fake species name.
    if taxonomy is None or taxonomy.get("species") is None:
        return {
            "request_id": request_id, "species_common": None, "species_scientific": None,
            "confidence": None, "taxonomy": None, "error": None,
        }

    scientific = f"{taxonomy['genus']} {taxonomy['species']}" if taxonomy.get("genus") else taxonomy["species"]
    return {
        "request_id": request_id,
        "species_common": taxonomy.get("common_name"),
        "species_scientific": scientific,
        "confidence": float(top_score),
        "taxonomy": taxonomy,
        "error": None,
    }


if __name__ == "__main__":
    _model_name = sys.argv[1]
    _confidence_threshold = float(sys.argv[2])
    _run_server(_model_name, _confidence_threshold)
