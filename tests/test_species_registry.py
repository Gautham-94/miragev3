"""Tests for the species classifier plugin registry (mirage/species/registry.py) --
mirrors mirage/detection/registry.py's own plugin-scan pattern.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mirage.config.schema import SpeciesClassifierConfig, SpeciesModelConfig
from mirage.species.registry import available_backends, create_classifier

# The real speciesnet package deliberately lives in a SEPARATE venv from mirage's own
# (see mirage/species/plugins/speciesnet.py's module docstring for the numpy/opencv
# conflict this avoids) -- tests that actually launch the worker subprocess and run
# real inference only run if that sibling venv has been set up locally.
_VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv-speciesnet" / "bin" / "python"
requires_speciesnet_venv = pytest.mark.skipif(
    not _VENV_PYTHON.exists(),
    reason="`.venv-speciesnet` not set up -- run: python3 -m venv .venv-speciesnet && "
    ".venv-speciesnet/bin/pip install speciesnet",
)


def test_noop_and_speciesnet_backends_are_registered():
    backends = available_backends()
    assert "noop" in backends
    assert "speciesnet" in backends


def test_create_classifier_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown species classifier backend"):
        create_classifier(SpeciesClassifierConfig(device="not-a-real-backend"))


def test_noop_classifier_always_returns_no_result():
    classifier = create_classifier(SpeciesClassifierConfig(device="noop"))
    result = classifier.classify(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result.species_common is None
    assert result.species_scientific is None
    assert result.confidence is None
    assert result.taxonomy is None


def test_speciesnet_classifier_missing_venv_raises_clear_error():
    """Selecting device="speciesnet" without the isolated venv set up must fail loudly
    and immediately at construction time (subprocess.Popen's own FileNotFoundError is
    already clear enough -- no special-casing needed), not silently no-op or hang.
    """
    config = SpeciesClassifierConfig(
        device="speciesnet", model=SpeciesModelConfig(venv_python_path="/nonexistent/venv/bin/python"),
    )
    with pytest.raises(OSError):
        create_classifier(config)


@requires_speciesnet_venv
def test_speciesnet_classifier_real_inference():
    """End-to-end: launches the real worker subprocess in the isolated venv, classifies
    a real photo. media/zidane.jpg contains a person -- SpeciesNet should confidently
    classify it as human (see speciesnet.constants.Classification.HUMAN), a stronger
    assertion than just "doesn't crash."
    """
    import cv2

    config = SpeciesClassifierConfig(device="speciesnet")
    classifier = create_classifier(config)

    bgr = cv2.imread("media/zidane.jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = classifier.classify(rgb)

    assert result.species_common == "human"
    assert result.species_scientific == "homo sapiens"
    assert result.confidence is not None and result.confidence > 0.5
    assert result.taxonomy is not None and result.taxonomy["class"] == "mammalia"


@requires_speciesnet_venv
def test_speciesnet_classifier_low_confidence_crop_returns_no_result():
    """A crop with no recognizable content should come back as a "no confident result"
    outcome (every field None), not an exception -- see SpeciesClassification's own
    docstring on this being the normal, expected shape for a low-confidence classification.
    """
    config = SpeciesClassifierConfig(device="speciesnet")
    classifier = create_classifier(config)

    noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = classifier.classify(noise)

    assert result.species_common is None
    assert result.species_scientific is None
    assert result.confidence is None
    assert result.taxonomy is None
