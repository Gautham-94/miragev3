"""Species classifier plugin registry -- mirrors mirage/detection/registry.py exactly.

Scans mirage.species.plugins at import time, registering each SpeciesClassifierApi
subclass by its type_key, wrapped in try/except ImportError so a missing runtime
library on a given platform doesn't crash the whole app (species classification is an
optional, opt-in feature -- see mirage.config.schema.SpeciesClassifierConfig).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from mirage.config.schema import SpeciesClassifierConfig
from mirage.species import plugins as _plugins_package
from mirage.species.api import SpeciesClassifierApi

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[SpeciesClassifierApi]] = {}
_SCANNED = False


def _scan_plugins() -> None:
    global _SCANNED
    if _SCANNED:
        return
    for _, module_name, _ in pkgutil.iter_modules(_plugins_package.__path__, prefix=f"{_plugins_package.__name__}."):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.info("species classifier plugin %s unavailable: %s", module_name, e)

    for subclass in SpeciesClassifierApi.__subclasses__():
        if subclass.type_key:
            _REGISTRY[subclass.type_key] = subclass

    _SCANNED = True


def create_classifier(classifier_config: SpeciesClassifierConfig) -> SpeciesClassifierApi:
    _scan_plugins()
    api_class = _REGISTRY.get(classifier_config.device)
    if api_class is None:
        raise ValueError(
            f"unknown species classifier backend {classifier_config.device!r}; available: {sorted(_REGISTRY)}"
        )
    return api_class(classifier_config.model)


def available_backends() -> list[str]:
    """Every registered `device` key a SpeciesClassifierConfig can validly use -- e.g.
    for the config-write API to validate a new classifier selection against, without
    reaching into this module's private _REGISTRY directly.
    """
    _scan_plugins()
    return sorted(_REGISTRY)
