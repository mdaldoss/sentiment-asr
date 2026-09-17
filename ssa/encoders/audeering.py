"""audeering wav2vec2 VAD encoder: the research-backend acoustic model.

**CC-BY-NC-SA-4.0 -- research use only.** Trained on MSP-Podcast (naturalistic
speech, not acted), unlike WavLM-base which we only fine-tune a probe on top
of. Callers must surface the license notice; see
ssa/solutions/acoustic.py's `backend="research"` path.

The model card (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) ships
a custom architecture, not a stock transformers class, so `EmotionModel` here
is copied from the model card's own usage example -- with one fix: the
original calls the legacy `self.init_weights()`, which on current
transformers versions never populates `all_tied_weights_keys` and makes
`from_pretrained` crash inside `_finalize_model_loading`. Calling the modern
`self.post_init()` instead (a superset that also ties weights) fixes it;
verified the resulting model's output matches the model card's own documented
reference values (arousal=0.5460754, dominance=0.6062266, valence=0.4043166
on an all-zeros 1s signal) to 6 decimal places.

Output order is [arousal, dominance, valence] per the model's own
config.id2label -- NOT valence-first. Mapping this wrong would silently
swap two of the three VAD axes.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from ssa.encoders.device import pick_device
from ssa.types import VAD, AudioClip

logger = logging.getLogger(__name__)

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
LICENSE_NOTICE = (
    f"NOTICE: {MODEL_ID} is licensed CC-BY-NC-SA-4.0 (research use only). "
    "Do not use this backend in a commercial product without a separate "
    "license from audEERING."
)


class _RegressionHead(nn.Module):
    """Verbatim from the model card: dense -> tanh -> dropout -> out_proj."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2PreTrainedModel):
    """Verbatim architecture from the model card, with post_init() in place
    of the legacy init_weights() call (see module docstring)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.post_init()

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled)
        return pooled, logits


class AudeeringVADEncoder:
    """Loads the audeering VAD model once; call .predict_vad() per clip."""

    def __init__(self, device: str | None = None, *, warn_on_load: bool = True) -> None:
        self.device = device or pick_device()
        if warn_on_load:
            logger.warning(LICENSE_NOTICE)
        self._processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self._model = _EmotionModel.from_pretrained(MODEL_ID)
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def predict_vad(self, clip: AudioClip) -> VAD:
        inputs = self._processor(clip.samples, sampling_rate=clip.sr)
        input_values = inputs["input_values"][0].reshape(1, -1)
        input_values = torch.from_numpy(input_values).to(self.device)

        _pooled, logits = self._model(input_values)
        arousal, dominance, valence = (float(v) for v in logits.squeeze(0).cpu().numpy())

        # The model's outputs are not guaranteed inside [0, 1] for
        # out-of-distribution audio; VAD validates its inputs, so clip
        # defensively rather than let a rare outlier crash a whole eval run.
        return VAD(
            valence=_clip01(valence),
            arousal=_clip01(arousal),
            dominance=_clip01(dominance),
        )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
