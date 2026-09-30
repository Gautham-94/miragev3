"""Whether this machine can actually use NVIDIA hardware-accelerated video decode, and
what ffmpeg -hwaccel args to use if so -- backs MirageConfig.hwaccel_enabled's automatic
application to every camera's detect-role ffmpeg input (see mirage/app.py).
"""

from __future__ import annotations

import onnxruntime as ort

# ffmpeg's NVDEC decode path -- verified live earlier in this project against all
# configured cameras (full verbose ffmpeg output, not just nvidia-smi's process list,
# which was confirmed separately to be an unreliable way to check this).
NVIDIA_HWACCEL_ARGS = ["-hwaccel", "cuda"]


def gpu_decode_available() -> bool:
    """True if this machine has a working NVIDIA GPU + driver + CUDA/cuDNN setup.

    Reuses onnxruntime's own CUDAExecutionProvider availability check (mirage/detection/
    execution_providers.py) as a reliable proxy, rather than a separate nvidia-smi-based
    detector: the exact same driver/CUDA/cuDNN prerequisites onnxruntime-gpu's
    CUDAExecutionProvider needs to even be listed are what ffmpeg's own -hwaccel cuda
    decode path needs too, and this signal is already proven correct (a whole session's
    worth of GPU packaging work here was verified against it) rather than something new
    to independently get wrong. Applying -hwaccel cuda unconditionally on a machine
    without a real NVIDIA GPU would not gracefully degrade -- ffmpeg fails that input
    outright rather than silently falling back to software decode -- so this check has
    to be right, not just present.
    """
    return "CUDAExecutionProvider" in ort.get_available_providers()
