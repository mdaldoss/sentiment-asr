"""Core data types shared across the pipeline.

These types are the contract between every module. They are deliberately rich:
the evaluation harness needs `vad` for quadrant analysis, `transcript` to tell
whether a lexical-solution failure was really an ASR failure, and `latency_ms`
for the CPU benchmark. Do not slim them down.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

# All audio in this project is mono float32 at this rate. Resampling happens
# once, at load time, in ssa.audio.load_clip.
SAMPLE_RATE = 16_000


class Sentiment(StrEnum):
    """The three-way target. Ordered negative -> neutral -> positive by valence."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"

    @classmethod
    def ordered(cls) -> tuple[Sentiment, ...]:
        """Canonical order for confusion matrices and probability vectors."""
        return (cls.NEGATIVE, cls.NEUTRAL, cls.POSITIVE)


@dataclass(frozen=True, slots=True)
class AudioClip:
    """A loaded mono 16 kHz clip."""

    clip_id: str
    samples: np.ndarray  # float32, shape (n,)
    sr: int = SAMPLE_RATE
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"{self.clip_id}: expected mono, got shape {self.samples.shape}")
        if self.sr != SAMPLE_RATE:
            raise ValueError(f"{self.clip_id}: expected {SAMPLE_RATE} Hz, got {self.sr}")

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sr


@dataclass(frozen=True, slots=True)
class VAD:
    """Valence / arousal / dominance, each in [0, 1].

    Valence is the sentiment axis. Arousal separates agitated distress
    (low valence, high arousal) from withdrawn distress (low valence, low
    arousal) -- the quadrant read-out described in DESIGN.md.
    """

    valence: float
    arousal: float
    dominance: float

    def __post_init__(self) -> None:
        for name in ("valence", "arousal", "dominance"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name}={v} outside [0, 1]")


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model output for one clip.

    Invariants (enforced in __post_init__):
      - `probs` has exactly the three Sentiment keys and sums to 1.
      - `confidence == max(probs.values())`.

    `abstained=True` does NOT change `sentiment`; it flags the prediction as
    low-trust and lets the caller decide. For a companion device, an abstention
    should mean "ask, don't assume".
    """

    sentiment: Sentiment
    probs: dict[Sentiment, float]
    confidence: float
    latency_ms: float
    solution: str
    abstained: bool = False
    vad: VAD | None = None
    transcript: str | None = None

    def __post_init__(self) -> None:
        if set(self.probs) != set(Sentiment):
            raise ValueError(f"probs must cover all sentiments, got {set(self.probs)}")
        total = sum(self.probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probs sum to {total}, expected 1.0")

    def prob_vector(self) -> np.ndarray:
        """Probabilities in canonical Sentiment.ordered() order."""
        return np.array([self.probs[s] for s in Sentiment.ordered()], dtype=np.float64)


class Solution(Protocol):
    """The interface every solution implements.

    Solutions A (lexical), B (acoustic) and C (fusion) are interchangeable from
    the evaluation harness's point of view. That is the whole point: the same
    harness produces directly comparable numbers for all three.
    """

    name: str

    def predict(self, clip: AudioClip) -> Prediction: ...

    def predict_batch(self, clips: Sequence[AudioClip]) -> list[Prediction]: ...
