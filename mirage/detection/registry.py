"""Detector plugin registry.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.1.

Scans mirage.detection.plugins at import time, registering each DetectionApi subclass by
its type_key, wrapped in try/except ImportError so a missing runtime library on a given
platform doesn't crash the whole app.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from mirage.config.schema import DetectorInstanceConfig
from mirage.detection.api import DetectionApi
from mirage.detection import plugins as _plugins_package

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[DetectionApi]] = {}
_SCANNED = False


def _scan_plugins() -> None:
    global _SCANNED
    if _SCANNED:
        return
    for _, module_name, _ in pkgutil.iter_modules(_plugins_package.__path__, prefix=f"{_plugins_package.__name__}."):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.info("detector plugin %s unavailable: %s", module_name, e)

    for subclass in DetectionApi.__subclasses__():
        if subclass.type_key:
            _REGISTRY[subclass.type_key] = subclass

    _SCANNED = True


def create_detector(detector_config: DetectorInstanceConfig) -> DetectionApi:
    _scan_plugins()
    api_class = _REGISTRY.get(detector_config.device)
    if api_class is None:
        raise ValueError(
            f"unknown detector backend {detector_config.device!r}; available: {sorted(_REGISTRY)}"
        )
    return api_class(detector_config.model)


def available_backends() -> list[str]:
    """Every registered `device` key a DetectorInstanceConfig can validly use -- e.g. for
    the config-write API to validate a new detector registration against, without reaching
    into this module's private _REGISTRY directly.
    """
    _scan_plugins()
    return sorted(_REGISTRY)
