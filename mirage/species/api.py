"""Species classifier abstraction -- one interface, multiple pluggable backends
(SpeciesNet today; BioCLIP/BirdNET/custom classifiers addable later with zero core
changes). Mirrors mirage/detection/api.py's DetectionApi shape, but crop-in/result-out
rather than fixed-tensor-shape-out -- classification has no NMS-slot output convention
to standardize on the way closed-vocab object detection does.

Mirage V3: only ever invoked for already-detected/tracked Animal/Bird events (see
mirage/events/processor.py's SPECIES_ENRICHABLE_LABELS) -- this module answers "which
species," never "is there an animal here at all," which stays the broad detector's job.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np

from mirage.config.schema import SpeciesModelConfig


@dataclasses.dataclass
class SpeciesClassification:
    """Result of classifying one crop. species_common/species_scientific/taxonomy are
    all None on a deliberate "no confident result" outcome (e.g. crop too small/blurry,
    or the model's own confidence floor wasn't met) -- that's a normal, expected result,
    not a failure; callers should treat it as species_status="skipped" for this event,
    distinct from an exception during classify() itself (species_status="failed").
    """

    species_common: str | None
    species_scientific: str | None
    confidence: float | None
    taxonomy: dict | None


class SpeciesClassifierApi(ABC):
    """One subclass per species-classification backend. Every implementation must load
    its model in __init__ and, in classify(), accept a single RGB crop (already cut to
    the detected animal/bird's box -- see mirage.util.thumbnail.crop_jpeg_to_box, the
    caller-side helper that produces this crop) and return a SpeciesClassification.
    """

    type_key: str = ""

    @abstractmethod
    def __init__(self, model_config: SpeciesModelConfig) -> None: ...

    @abstractmethod
    def classify(self, crop_rgb: np.ndarray) -> SpeciesClassification: ...
