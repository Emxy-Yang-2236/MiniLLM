from __future__ import annotations

import torch


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not _mps_available():
        raise RuntimeError("MPS was requested, but this PyTorch install cannot use MPS.")
    return torch.device(name)
