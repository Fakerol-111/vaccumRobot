from __future__ import annotations

import logging
import sys

import torch


def get_device() -> torch.device:
    """Auto-detect device: CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with stdout handler."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
