"""A species classifier backend that never actually classifies anything -- every
crop comes back as SpeciesClassification(None, None, None, None), i.e. "no confident
result." Exists to let the full species pipeline (config -> SpeciesProcess ->
SpeciesDispatcher -> EventProcessor dispatch -> Event.data read-modify-write) be wired
up, started, and tested end-to-end BEFORE a real classifier (SpeciesNet or otherwise)
is integrated -- see the Mirage V3 species classification plan's staging recommendation
("a fake/no-op classifier plugin first, prove the plumbing end-to-end without needing a
real model yet").

Not the SpeciesClassifierConfig.device default (that stays "speciesnet") -- select it
explicitly (device="noop") for staged rollout/manual testing.
"""

from __future__ import annotations

import numpy as np

from mirage.config.schema import SpeciesModelConfig
from mirage.species.api import SpeciesClassification, SpeciesClassifierApi


class NoopSpeciesClassifier(SpeciesClassifierApi):
    type_key = "noop"

    def __init__(self, model_config: SpeciesModelConfig) -> None:
        self.model_config = model_config

    def classify(self, crop_rgb: np.ndarray) -> SpeciesClassification:
        return SpeciesClassification(
            species_common=None, species_scientific=None, confidence=None, taxonomy=None,
        )
