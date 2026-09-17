"""Device selection for torch-based encoders.

CTranslate2 (used by faster-whisper, see ssa/asr.py) has NO Apple MPS
support -- that constraint does NOT apply here. torch encoders run fine on
MPS, which matters for the project's stated dev machine (a MacBook Air M3).
"""

from __future__ import annotations

import torch


def pick_device() -> str:
    """Best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
