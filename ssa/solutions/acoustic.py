"""Solution B: acoustic-only. Frozen encoder + trained probe -> Prediction.

Two backends behind one interface:
  - "permissive": WavLM-base + our own trained probe. MIT/Apache end to end.
  - "research": the audeering VAD model + valence thresholds fit on val.
    **CC-BY-NC-SA-4.0, research use only** -- constructing this backend
    prints a license notice (see ssa/encoders/audeering.py).

This solution never sees the transcript. On congruent data it should score
lower than the lexical solution (prosody alone is a harder signal than
words alone for many utterances); on incongruent data (E2/E3) it is the one
solution that *can* score well on PSI, because tone is all it has.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import joblib

from ssa.encoders.audeering import AudeeringVADEncoder
from ssa.encoders.wavlm import WavLMEncoder
from ssa.types import AudioClip, Prediction, Sentiment
from ssa.vad import ValenceThresholds, valence_to_sentiment

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROBE_PATH = REPO_ROOT / "data" / "cache" / "probe_permissive.joblib"
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "data" / "cache" / "thresholds_research.json"


class AcousticSolution:
    """Frozen encoder + a trained/fitted head. See module docstring for the
    two backends."""

    def __init__(
        self,
        backend: str = "permissive",
        *,
        probe_path: Path = DEFAULT_PROBE_PATH,
        thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
        encoder: WavLMEncoder | AudeeringVADEncoder | None = None,
    ) -> None:
        self.backend = backend
        if backend == "permissive":
            if not probe_path.exists():
                raise FileNotFoundError(
                    f"no trained probe at {probe_path} -- run `make train` first"
                )
            self._encoder = encoder if encoder is not None else WavLMEncoder()
            bundle = joblib.load(probe_path)
            self._scaler = bundle["scaler"]
            self._clf = bundle["clf"]
            self.name = f"B:acoustic(permissive/{bundle['model_type']})"
        elif backend == "research":
            if not thresholds_path.exists():
                raise FileNotFoundError(
                    f"no fitted thresholds at {thresholds_path} -- run "
                    "`make train BACKEND=research` first"
                )
            self._encoder = encoder if encoder is not None else AudeeringVADEncoder()
            data = json.loads(thresholds_path.read_text())
            self._thresholds = ValenceThresholds(low=data["low"], high=data["high"])
            self.name = "B:acoustic(research/audeering-vad)"
        else:
            raise ValueError(f"unknown backend {backend!r}, expected 'permissive' or 'research'")

    def predict(self, clip: AudioClip) -> Prediction:
        if self.backend == "permissive":
            return self._predict_permissive(clip)
        return self._predict_research(clip)

    def _predict_permissive(self, clip: AudioClip) -> Prediction:
        t0 = time.perf_counter()
        embedding = self._encoder.embed(clip)
        embedding_scaled = self._scaler.transform(embedding.reshape(1, -1))
        proba = self._clf.predict_proba(embedding_scaled)[0]

        # Zip against clf.classes_ explicitly rather than assuming an order --
        # sklearn sorts string labels alphabetically, which happens to match
        # Sentiment.ordered() here, but relying on that coincidence would be
        # exactly the kind of silent-label-mixup bug CLAUDE.md warns about.
        probs = {Sentiment(cls): float(p) for cls, p in zip(self._clf.classes_, proba, strict=True)}
        sentiment = max(probs, key=probs.get)  # type: ignore[arg-type]
        latency_ms = (time.perf_counter() - t0) * 1000

        return Prediction(
            sentiment=sentiment,
            probs=probs,
            confidence=probs[sentiment],
            latency_ms=latency_ms,
            solution=self.name,
        )

    def _predict_research(self, clip: AudioClip) -> Prediction:
        t0 = time.perf_counter()
        vad = self._encoder.predict_vad(clip)
        sentiment = valence_to_sentiment(vad.valence, self._thresholds)

        # This backend has no learned probability distribution -- only a
        # thresholded point estimate. Confidence is proxied by distance from
        # the nearer threshold, scaled to (0.34, 1.0) so it's never below
        # chance for a 3-way problem; probs are a soft one-hot around that.
        confidence = _threshold_confidence(vad.valence, self._thresholds)
        probs = {s: (1.0 - confidence) / 2 for s in Sentiment}
        probs[sentiment] = confidence

        latency_ms = (time.perf_counter() - t0) * 1000
        return Prediction(
            sentiment=sentiment,
            probs=probs,
            confidence=confidence,
            latency_ms=latency_ms,
            solution=self.name,
            vad=vad,
        )

    def predict_batch(self, clips: Sequence[AudioClip]) -> list[Prediction]:
        return [self.predict(c) for c in clips]


def _threshold_confidence(valence: float, thresholds: ValenceThresholds) -> float:
    """Distance from the nearer threshold, normalised into (1/3, 1.0].

    A point estimate has no real probability distribution behind it; this is
    a deliberately crude proxy, not a calibrated confidence -- Solution C's
    fusion calibrates properly and should be preferred wherever calibration
    matters (see ssa/solutions/fusion.py)."""
    if valence <= thresholds.low:
        dist = thresholds.low - valence
        scale = max(thresholds.low, 1e-6)
    elif valence >= thresholds.high:
        dist = valence - thresholds.high
        scale = max(1.0 - thresholds.high, 1e-6)
    else:
        span = thresholds.high - thresholds.low
        dist = min(valence - thresholds.low, thresholds.high - valence)
        scale = max(span / 2, 1e-6)
    normalised = min(dist / scale, 1.0)
    return 1 / 3 + normalised * (2 / 3)
