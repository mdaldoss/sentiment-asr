"""Evaluation metrics.

Two families live here. The accuracy-family metrics (uar, macro_f1,
confusion_matrix, expected_calibration_error) are standard, but note rule 7
in CLAUDE.md: UAR, not plain accuracy, is the headline, because CREMA-D and
friends are class-imbalanced (negative is ~68% of CREMA-D) and a majority-
class predictor scores well on plain accuracy without learning anything.

The Prosody Sensitivity Index (psi_contested / psi_strict) is specific to
this project: it measures whether a solution's predictions track the tone
of a clip or the words, on clips where the two disagree. This is the
headline metric because it directly answers the question the whole
evaluation design exists to ask.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix

from ssa.types import Prediction, Sentiment

_LABELS: tuple[str, ...] = tuple(s.value for s in Sentiment.ordered())


def uar(y_true: Sequence[Sentiment], y_pred: Sequence[Sentiment]) -> float:
    """Unweighted average recall = macro-averaged per-class recall.

    Equivalent to sklearn's balanced_accuracy_score for multiclass. Returns
    nan (not 0.0) on empty input -- an undefined metric should never look
    like a bad score.
    """
    if len(y_true) == 0:
        return float("nan")
    return float(balanced_accuracy_score(_labels(y_true), _labels(y_pred)))


def macro_f1(y_true: Sequence[Sentiment], y_pred: Sequence[Sentiment]) -> float:
    """Macro-averaged F1, reported alongside UAR."""
    if len(y_true) == 0:
        return float("nan")
    return float(
        f1_score(
            _labels(y_true), _labels(y_pred), labels=list(_LABELS), average="macro", zero_division=0
        )
    )


def confusion_matrix(y_true: Sequence[Sentiment], y_pred: Sequence[Sentiment]) -> np.ndarray:
    """Confusion matrix with rows/columns in Sentiment.ordered() order
    (negative, neutral, positive), regardless of which labels are present."""
    return _sk_confusion_matrix(_labels(y_true), _labels(y_pred), labels=list(_LABELS))


def expected_calibration_error(
    probs: np.ndarray, y_true: Sequence[Sentiment], n_bins: int = 10
) -> float:
    """Top-label ECE: bins samples by max-probability confidence, compares
    each bin's accuracy to its mean confidence, and weight-averages the gap.

    probs: shape (n, 3), columns in Sentiment.ordered() order (e.g. from
    Prediction.prob_vector()). A well-calibrated model has ECE near 0; a
    model that is confidently wrong -- the failure mode that matters most
    for a companion device that should ask rather than assume -- has ECE
    that stays high even when accuracy looks fine.
    """
    n = len(y_true)
    if n == 0:
        return float("nan")

    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (n, len(_LABELS)):
        raise ValueError(f"expected probs shape ({n}, {len(_LABELS)}), got {probs.shape}")

    confidences = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    ordered = Sentiment.ordered()
    correct = np.array([ordered[i] == t for i, t in zip(pred_idx, y_true, strict=True)])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i > 0:
            mask = (confidences > lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def psi_contested(preds: Sequence[Prediction], df: pd.DataFrame) -> float:
    """Prosody Sensitivity Index, contested variant -- THE HEADLINE METRIC.

    Over incongruent clips (text_sentiment != prosody_sentiment) where the
    prediction equals either label -- i.e. the model "picked a side" -- the
    fraction that picked the prosody label. This isolates the lexical-vs-
    acoustic decision from predictions that landed on neither label.

    1.0 = always follows tone.  0.0 = always follows words.  0.5 = chance.
    nan when there are no contested clips (never 0.0 -- undefined is not
    the same as "reads the transcript").

    `preds` and `df` must be the same length and in the same row order; the
    evaluation harness guarantees this by construction (it generates one
    Prediction per manifest row, in order).
    """
    _check_aligned(preds, df)
    contested = 0
    picked_prosody = 0
    for pred, row in zip(preds, df.itertuples(index=False), strict=True):
        if row.is_congruent:
            continue
        pred_s = pred.sentiment.value
        if pred_s == row.prosody_sentiment:
            contested += 1
            picked_prosody += 1
        elif pred_s == row.text_sentiment:
            contested += 1
    if contested == 0:
        return float("nan")
    return picked_prosody / contested


def psi_strict(preds: Sequence[Prediction], df: pd.DataFrame) -> float:
    """Prosody Sensitivity Index, strict variant.

    Over ALL incongruent clips (not just contested ones), the fraction
    matching the prosody label -- equivalent to prosody-accuracy on
    incongruent data. Reported beside psi_contested because the contested
    variant ignores third-label predictions and could flatter a model that
    mostly predicts the neutral class regardless of tone.

    Chance = 1/3 (three possible labels). nan when there are no incongruent
    clips at all.
    """
    _check_aligned(preds, df)
    incongruent_n = 0
    matched_prosody = 0
    for pred, row in zip(preds, df.itertuples(index=False), strict=True):
        if row.is_congruent:
            continue
        incongruent_n += 1
        if pred.sentiment.value == row.prosody_sentiment:
            matched_prosody += 1
    if incongruent_n == 0:
        return float("nan")
    return matched_prosody / incongruent_n


def _labels(sentiments: Sequence[Sentiment]) -> list[str]:
    return [s.value for s in sentiments]


def _check_aligned(preds: Sequence[Prediction], df: pd.DataFrame) -> None:
    if len(preds) != len(df):
        raise ValueError(f"preds ({len(preds)}) and df ({len(df)}) length mismatch")
