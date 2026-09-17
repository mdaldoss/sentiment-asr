"""Solution C: calibrated late fusion of A (lexical) and B (acoustic).

Late, not early: fusion combines each branch's already-computed probability
distribution, never their internal representations. This is a deliberate
accuracy sacrifice -- joint/early fusion would likely score marginally
higher -- because late fusion keeps each branch's output independently
measurable. Per-branch PSI (does the lexical half read words, does the
acoustic half hear tone?) is the analysis this whole project exists to
produce, and early fusion would destroy the ability to compute it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ssa.types import AudioClip, Prediction, Sentiment, Solution

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class FusionParams:
    """Fitted on the validation split only -- see fit_fusion_params."""

    weight_lexical: float  # weight_acoustic is implicitly (1 - weight_lexical)
    temperature: float
    abstain_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight_lexical <= 1.0:
            raise ValueError(f"weight_lexical must be in [0, 1], got {self.weight_lexical}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> FusionParams:
        return cls(**json.loads(path.read_text()))


DEFAULT_PARAMS = FusionParams(weight_lexical=0.5, temperature=1.0, abstain_threshold=0.0)


class FusionSolution:
    """Combines a lexical and an acoustic Solution's probability vectors:
    softmax((w * log p_lexical + (1-w) * log p_acoustic) / temperature).
    """

    def __init__(
        self,
        lexical: Solution,
        acoustic: Solution,
        params: FusionParams = DEFAULT_PARAMS,
    ) -> None:
        self._lexical = lexical
        self._acoustic = acoustic
        self._params = params
        self.name = f"C:fusion({lexical.name}+{acoustic.name})"

    def predict(self, clip: AudioClip) -> Prediction:
        t0 = time.perf_counter()
        pred_a = self._lexical.predict(clip)
        pred_b = self._acoustic.predict(clip)
        probs, sentiment, confidence = fuse_probs(pred_a.probs, pred_b.probs, self._params)
        latency_ms = (time.perf_counter() - t0) * 1000

        return Prediction(
            sentiment=sentiment,
            probs=probs,
            confidence=confidence,
            latency_ms=latency_ms,
            solution=self.name,
            abstained=confidence < self._params.abstain_threshold,
            vad=pred_b.vad,
            transcript=pred_a.transcript,
        )

    def predict_batch(self, clips: Sequence[AudioClip]) -> list[Prediction]:
        return [self.predict(c) for c in clips]


def fuse_probs(
    probs_a: dict[Sentiment, float],
    probs_b: dict[Sentiment, float],
    params: FusionParams = DEFAULT_PARAMS,
) -> tuple[dict[Sentiment, float], Sentiment, float]:
    """The pure fusion math, factored out so it can be unit-tested and
    grid-searched over without needing real Solution objects."""
    ordered = Sentiment.ordered()
    log_a = np.log(np.clip([probs_a[s] for s in ordered], _EPS, 1.0))
    log_b = np.log(np.clip([probs_b[s] for s in ordered], _EPS, 1.0))

    combined = params.weight_lexical * log_a + (1 - params.weight_lexical) * log_b
    combined = combined / params.temperature
    combined = combined - combined.max()  # numerically stable softmax
    exp = np.exp(combined)
    probs_arr = exp / exp.sum()

    probs = {s: float(p) for s, p in zip(ordered, probs_arr, strict=True)}
    sentiment = max(probs, key=probs.get)  # type: ignore[arg-type]
    return probs, sentiment, probs[sentiment]


def fit_fusion_params(
    probs_a_list: Sequence[dict[Sentiment, float]],
    probs_b_list: Sequence[dict[Sentiment, float]],
    y_true: Sequence[Sentiment],
    *,
    target_abstention_rate: float = 0.10,
    weight_candidates: int = 21,
    temperature_candidates: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
) -> FusionParams:
    """Fit weight_lexical and temperature by grid search on (probs_a_list,
    probs_b_list, y_true) -- intended for the VALIDATION split only, never
    train or test (see docs/ARCHITECTURE.md section 4). abstain_threshold is
    then set to the target_abstention_rate quantile of the resulting
    (calibrated) confidences, so roughly that fraction of val predictions
    would abstain.
    """
    from ssa.eval.metrics import uar  # local import: avoids a module-load cycle with eval/

    if not (len(probs_a_list) == len(probs_b_list) == len(y_true)):
        raise ValueError("probs_a_list, probs_b_list, and y_true must be the same length")
    if len(y_true) == 0:
        raise ValueError("cannot fit fusion params on an empty validation set")

    weights = np.linspace(0.0, 1.0, weight_candidates)
    best_weight, best_uar = 0.5, -1.0
    for w in weights:
        params = FusionParams(weight_lexical=float(w), temperature=1.0, abstain_threshold=0.0)
        preds = [
            fuse_probs(pa, pb, params)[1] for pa, pb in zip(probs_a_list, probs_b_list, strict=True)
        ]
        score = uar(y_true, preds)
        if score > best_uar:
            best_uar = score
            best_weight = float(w)

    # Temperature doesn't change argmax (hence not UAR) -- it only reshapes
    # confidence, so pick it by held-out negative log-likelihood instead.
    best_temp, best_nll = 1.0, float("inf")
    for t in temperature_candidates:
        params = FusionParams(weight_lexical=best_weight, temperature=t, abstain_threshold=0.0)
        nll = 0.0
        for pa, pb, y in zip(probs_a_list, probs_b_list, y_true, strict=True):
            probs, _sent, _conf = fuse_probs(pa, pb, params)
            nll -= np.log(max(probs[y], _EPS))
        if nll < best_nll:
            best_nll = nll
            best_temp = t

    final_params = FusionParams(
        weight_lexical=best_weight, temperature=best_temp, abstain_threshold=0.0
    )
    confidences = np.array(
        [
            fuse_probs(pa, pb, final_params)[2]
            for pa, pb in zip(probs_a_list, probs_b_list, strict=True)
        ]
    )
    abstain_threshold = float(np.quantile(confidences, target_abstention_rate))

    return FusionParams(
        weight_lexical=best_weight, temperature=best_temp, abstain_threshold=abstain_threshold
    )
