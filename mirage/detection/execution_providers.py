"""Resolves a ModelConfig.execution_provider choice into the actual onnxruntime
`providers=[...]` list to pass into InferenceSession, and reports which accelerators are
really present on this machine (mirage/config/schema.py's ExecutionProvider docstring has
the full rationale/benchmark).

onnxruntime's own provider list is priority-ordered with automatic per-op fallback -- e.g.
["CoreMLExecutionProvider", "CPUExecutionProvider"] means "prefer CoreML, silently fall
back to CPU for anything CoreML can't run" within the SAME session, not an either/or
choice. Everything here preserves that semantic rather than forcing a single provider.
"""

from __future__ import annotations

import onnxruntime as ort

from mirage.config.schema import ExecutionProvider

_PROVIDER_NAMES: dict[ExecutionProvider, str] = {
    ExecutionProvider.coreml: "CoreMLExecutionProvider",
    ExecutionProvider.cuda: "CUDAExecutionProvider",
}


def available_execution_providers() -> list[str]:
    """Every ExecutionProvider value onnxruntime can actually use on this machine right
    now (cpu is always available) -- e.g. for the Manage Detectors form to only ever
    offer an accelerator choice that will really work on the machine mirage is running
    on, rather than one that would silently no-op back to CPU.
    """
    installed = set(ort.get_available_providers())
    # cpu and auto are always valid choices -- auto degrades gracefully to cpu on a
    # machine with no accelerator, it never fails to resolve.
    result = [ExecutionProvider.cpu.value, ExecutionProvider.auto.value]
    for provider, ort_name in _PROVIDER_NAMES.items():
        if ort_name in installed:
            result.append(provider.value)
    return result


def resolve_providers(execution_provider: ExecutionProvider) -> list[str]:
    """Builds the priority-ordered onnxruntime `providers` list for a given config choice.
    CPU is always appended last as the guaranteed-available fallback (except when cpu
    itself is the explicit choice, where it's the only entry).
    """
    if execution_provider == ExecutionProvider.cpu:
        return ["CPUExecutionProvider"]

    if execution_provider == ExecutionProvider.auto:
        installed = set(ort.get_available_providers())
        # Preference order: CUDA (discrete GPU, fastest when present) before CoreML.
        preferred = [
            ort_name
            for provider, ort_name in [(ExecutionProvider.cuda, "CUDAExecutionProvider"), (ExecutionProvider.coreml, "CoreMLExecutionProvider")]
            if ort_name in installed
        ]
        return preferred + ["CPUExecutionProvider"]

    ort_name = _PROVIDER_NAMES.get(execution_provider)
    if ort_name is None:
        return ["CPUExecutionProvider"]
    return [ort_name, "CPUExecutionProvider"]
