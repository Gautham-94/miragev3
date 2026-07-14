"""Resolves which torch device OpenVocabProcess should load OWLv2 onto, and reports
which devices are really usable on this machine. Deliberately separate from
mirage/detection/execution_providers.py: that module resolves onnxruntime's
providers=[...] list for the YOLOv8n plugin; OWLv2 runs on PyTorch/transformers, which
has its own, differently-shaped device API (torch.device("cpu"/"mps"/"cuda")), not a
priority-ordered fallback list.

Real numbers measured on this repo's own dev machine (TODO_FIX_LIST.md item 6): MPS
gave a modest ~1.21x speedup over CPU (1720.7ms vs 1423.8ms avg, google/owlv2-base-
patch16-ensemble against media/bus.jpg) -- MPS only accelerates GPU-core matrix ops, it
does NOT reach the Apple Neural Engine the way onnxruntime's CoreMLExecutionProvider
does, so don't expect CoreML-YOLOv8n-scale (~5.6x) gains here.
"""

from __future__ import annotations


def available_devices() -> list[str]:
    """Every torch device string usable on this machine right now. cpu is always
    available.
    """
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def resolve_device(requested: str) -> str:
    """Validates a requested device string is actually usable, falling back to cpu (with
    a log warning, left to the caller) if not. `auto` prefers cuda, then mps, then cpu.
    """
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested
