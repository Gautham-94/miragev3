"""Labelmap loading.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 3.3.2.
"""

from __future__ import annotations


def load_labels(path: str, prefill_range: int = 91) -> dict[int, str]:
    """Supports both index-prefixed ("0 person") and plain newline-delimited (implicit
    line-number index) label files. Pre-fills indices [0, prefill_range) with "unknown"
    before overlaying real entries, so an out-of-range/unexpected class_id from a
    detector never raises an index error downstream.
    """
    labels: dict[int, str] = {i: "unknown" for i in range(prefill_range)}

    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    for i, line in enumerate(lines):
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1]
        else:
            labels[i] = line

    return labels
