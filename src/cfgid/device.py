"""Device negotiation: CUDA > MPS > CPU, with the MPS CPU-fallback enabled.

Never hardcode ``cuda``/``mps`` elsewhere — always go through ``get_device()`` so the same
code runs on Apple-silicon MPS and on Linux CUDA without changes.
"""

from __future__ import annotations

import os

import torch

# Read by torch at op-dispatch time; set as early as possible so any MPS op that lacks a
# kernel transparently falls back to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> torch.device:
    """Best available device, preferring CUDA, then MPS, then CPU."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def is_mps(device: torch.device) -> bool:
    return device.type == "mps"


def recommended_dtype(device: torch.device) -> torch.dtype:
    """Soft default compute dtype for a device.

    The authoritative per-model dtype lives in ``configs/models.yaml``; this is only a
    fallback for ad-hoc tensors. fp16 on accelerators, fp32 on CPU.
    """
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32
