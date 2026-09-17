"""WavLM-base frozen encoder: the permissive-backend acoustic feature extractor.

MIT-licensed end to end (WavLM-base + our own trained probe on top), unlike
the research backend (ssa/encoders/audeering.py), which is CC-BY-NC-SA-4.0.

Embeddings are mean+std pooled from the last hidden layer, per-clip
(unbatched). Batching with padding was measured to give no speedup on CPU
for this model at these clip lengths (padding waste roughly cancels the
matmul-efficiency gain) and unbatched avoids attention-mask-aware pooling
entirely, so we keep it simple.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from ssa.encoders.device import pick_device
from ssa.types import AudioClip

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/wavlm-base"
HIDDEN_SIZE = 768
EMBEDDING_DIM = HIDDEN_SIZE * 2  # mean + std pooling


class WavLMEncoder:
    """Loads microsoft/wavlm-base once; call .embed() per clip.

    Frozen: no gradients, eval mode, weights never updated. This is what
    makes the acoustic solution "lightweight" in the sense CLAUDE.md means --
    training only ever fits a small probe on top of fixed embeddings.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or pick_device()
        logger.info("loading %s on %s", MODEL_ID, self.device)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self._model = WavLMModel.from_pretrained(MODEL_ID)
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def embed(self, clip: AudioClip) -> np.ndarray:
        """Mean+std pooled embedding for one clip. Shape (EMBEDDING_DIM,)."""
        inputs = self._feature_extractor(clip.samples, sampling_rate=clip.sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden = self._model(**inputs).last_hidden_state  # (1, T, H)
        hidden = hidden.squeeze(0)  # (T, H)
        # unbiased=False avoids a NaN from the n-1 denominator on a
        # pathologically short clip that yields only one hidden-state frame.
        pooled = torch.cat([hidden.mean(dim=0), hidden.std(dim=0, unbiased=False)])
        return pooled.cpu().numpy().astype(np.float32)
