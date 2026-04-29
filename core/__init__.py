from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Auto-detect device: CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
