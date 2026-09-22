"""Valence-arousal-dominance read-outs.

Sentiment is a threshold read-out of valence -- this is the required
deliverable, and it's how Solution B's research backend produces a
pos/neu/neg label from a continuous VAD prediction. The same VAD triple
also gives distress quadrants "for free" (a roadmap extension, not part of
the required deliverable) -- see DESIGN.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ssa.types import VAD, Sentiment


@dataclass(frozen=True, slots=True)
class ValenceThresholds:
    """valence <= low -> NEGATIVE, valence >= high -> POSITIVE, else NEUTRAL."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.low < self.high <= 1.0):
            raise ValueError(f"expected 0 <= low < high <= 1, got low={self.low}, high={self.high}")


# A reasonable prior before any threshold has been fit -- splits [0,1] into
# equal thirds. fit_valence_thresholds() should always be preferred in
# practice; this exists mainly as a fallback for e.g. quick manual testing.
DEFAULT_THRESHOLDS = ValenceThresholds(low=1 / 3, high=2 / 3)


def valence_to_sentiment(
    valence: float, thresholds: ValenceThresholds = DEFAULT_THRESHOLDS
) -> Sentiment:
    if valence <= thresholds.low:
        return Sentiment.NEGATIVE
    if valence >= thresholds.high:
        return Sentiment.POSITIVE
    return Sentiment.NEUTRAL


def fit_valence_thresholds(
    valences: Sequence[float],
    y_true: Sequence[Sentiment],
    *,
    n_candidates: int = 41,
) -> ValenceThresholds:
    """Grid-search the (low, high) pair that maximises UAR against y_true.

    Per docs/ARCHITECTURE.md section 4: call this on the VALIDATION split
    only. Fitting on train or test would leak label information into
    Solution B's "research" backend the same way an unfit threshold with
    peeked-at test labels would -- the whole point of the speaker-disjoint
    split is defeated if thresholds are tuned on data used to report the
    final number.
    """
    from ssa.eval.metrics import uar  # local import: avoids a module-load cycle with eval/

    if len(valences) != len(y_true):
        raise ValueError(f"valences ({len(valences)}) and y_true ({len(y_true)}) length mismatch")

    candidates = np.linspace(0.0, 1.0, n_candidates)
    best = DEFAULT_THRESHOLDS
    best_uar = -1.0
    for i, low in enumerate(candidates):
        for high in candidates[i + 1 :]:
            preds = [
                valence_to_sentiment(v, ValenceThresholds(float(low), float(high)))
                for v in valences
            ]
            score = uar(y_true, preds)
            if score > best_uar:  # NaN comparisons are always False -- degenerate splits just lose
                best_uar = score
                best = ValenceThresholds(low=float(low), high=float(high))
    return best


class DistressQuadrant(StrEnum):
    """Roadmap read-out, NOT part of the required sentiment deliverable.

    Framed as tone-adaptation input (see DESIGN.md's Ami-specific roadmap),
    never as screening or diagnosis. Untested against any real distress
    data -- this is a design proposal, not a validated classifier.
    """

    FINE = "fine"  # valence high
    AGITATED = "agitated"  # low valence, high arousal
    WITHDRAWN = "withdrawn"  # low valence, low arousal


def vad_to_quadrant(
    vad: VAD, *, valence_threshold: float = 0.5, arousal_threshold: float = 0.5
) -> DistressQuadrant:
    if vad.valence >= valence_threshold:
        return DistressQuadrant.FINE
    return (
        DistressQuadrant.AGITATED
        if vad.arousal >= arousal_threshold
        else DistressQuadrant.WITHDRAWN
    )
