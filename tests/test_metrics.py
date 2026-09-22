"""Metric invariants, especially the Prosody Sensitivity Index.

test_psi_* fixtures are hand-computed: an all-prosody predictor must score
PSI 1.0, an all-text predictor 0.0, and the empty case must give nan (never
0.0 -- an undefined metric must never look like a confident wrong answer).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ssa.eval.metrics import (
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    psi_contested,
    psi_strict,
    uar,
)
from ssa.types import Prediction, Sentiment


def _pred(sentiment: Sentiment, confidence: float = 0.8) -> Prediction:
    remainder = (1.0 - confidence) / 2
    probs = {s: remainder for s in Sentiment}
    probs[sentiment] = confidence
    return Prediction(
        sentiment=sentiment,
        probs=probs,
        confidence=confidence,
        latency_ms=1.0,
        solution="test",
    )


@pytest.fixture
def incongruent_manifest() -> pd.DataFrame:
    """4 incongruent clips: text says one thing, prosody says another."""
    df = pd.DataFrame(
        {
            "clip_id": ["a", "b", "c", "d"],
            "text_sentiment": [
                Sentiment.NEUTRAL.value,
                Sentiment.POSITIVE.value,
                Sentiment.NEGATIVE.value,
                Sentiment.NEUTRAL.value,
            ],
            "prosody_sentiment": [
                Sentiment.NEGATIVE.value,
                Sentiment.NEGATIVE.value,
                Sentiment.POSITIVE.value,
                Sentiment.POSITIVE.value,
            ],
        }
    )
    df["is_congruent"] = df["text_sentiment"] == df["prosody_sentiment"]
    return df


class TestPSI:
    def test_all_prosody_predictions_give_psi_one(self, incongruent_manifest: pd.DataFrame) -> None:
        preds = [_pred(Sentiment(s)) for s in incongruent_manifest["prosody_sentiment"]]
        assert psi_contested(preds, incongruent_manifest) == pytest.approx(1.0)
        assert psi_strict(preds, incongruent_manifest) == pytest.approx(1.0)

    def test_all_text_predictions_give_psi_zero(self, incongruent_manifest: pd.DataFrame) -> None:
        preds = [_pred(Sentiment(s)) for s in incongruent_manifest["text_sentiment"]]
        assert psi_contested(preds, incongruent_manifest) == pytest.approx(0.0)
        assert psi_strict(preds, incongruent_manifest) == pytest.approx(0.0)

    def test_empty_incongruent_set_gives_nan(self) -> None:
        df = pd.DataFrame(
            {
                "clip_id": ["a"],
                "text_sentiment": [Sentiment.NEUTRAL.value],
                "prosody_sentiment": [Sentiment.NEUTRAL.value],
            }
        )
        df["is_congruent"] = True
        preds = [_pred(Sentiment.NEUTRAL)]
        assert math.isnan(psi_contested(preds, df))
        assert math.isnan(psi_strict(preds, df))

    def test_contested_ignores_third_label_strict_does_not(
        self, incongruent_manifest: pd.DataFrame
    ) -> None:
        """A predictor that always says NEUTRAL never matches text or prosody
        on rows where neither label is neutral -- contested should exclude
        those rows entirely, while strict counts them as misses."""
        # row 'a': text=neutral, prosody=negative -> predicting neutral matches text
        # row 'b': text=positive, prosody=negative -> predicting neutral matches neither
        # row 'c': text=negative, prosody=positive -> predicting neutral matches neither
        # row 'd': text=neutral, prosody=positive -> predicting neutral matches text
        preds = [_pred(Sentiment.NEUTRAL) for _ in range(4)]
        # contested: rows a and d are contested (matched text), 0/2 matched prosody
        assert psi_contested(preds, incongruent_manifest) == pytest.approx(0.0)
        # strict: 0/4 matched prosody
        assert psi_strict(preds, incongruent_manifest) == pytest.approx(0.0)

    def test_mismatched_lengths_raise(self, incongruent_manifest: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            psi_contested([_pred(Sentiment.NEUTRAL)], incongruent_manifest)


class TestUarMacroF1:
    def test_perfect_predictions_give_uar_one(self) -> None:
        y = [Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE, Sentiment.NEGATIVE]
        assert uar(y, y) == pytest.approx(1.0)
        assert macro_f1(y, y) == pytest.approx(1.0)

    def test_uar_averages_per_class_not_per_sample(self) -> None:
        """9 negatives (all correct) + 1 positive (wrong) should NOT score
        90% under UAR the way it would under plain accuracy -- UAR averages
        per-class recall, so the one wrong class drags it down much more."""
        y_true = [Sentiment.NEGATIVE] * 9 + [Sentiment.POSITIVE]
        y_pred = [Sentiment.NEGATIVE] * 9 + [Sentiment.NEGATIVE]  # positive misclassified
        plain_accuracy = 9 / 10
        assert uar(y_true, y_pred) < plain_accuracy

    def test_empty_input_gives_nan(self) -> None:
        assert math.isnan(uar([], []))
        assert math.isnan(macro_f1([], []))


class TestConfusionMatrix:
    def test_shape_and_order(self) -> None:
        y_true = [Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE]
        y_pred = [Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(cm), [1, 1, 1])


class TestECE:
    def test_perfectly_calibrated_gives_low_ece(self) -> None:
        # confidence matches accuracy: 10 samples at 0.9 confidence, 9 correct
        y_true = [Sentiment.NEGATIVE] * 9 + [Sentiment.NEUTRAL]
        probs = np.tile([0.9, 0.05, 0.05], (10, 1))  # ordered: neg, neu, pos
        ece = expected_calibration_error(probs, y_true, n_bins=10)
        assert ece < 0.15

    def test_empty_gives_nan(self) -> None:
        assert math.isnan(expected_calibration_error(np.zeros((0, 3)), []))

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="expected probs shape"):
            expected_calibration_error(np.zeros((3, 2)), [Sentiment.NEUTRAL] * 3)
