"""VAD threshold-fitting and quadrant read-out tests."""

from __future__ import annotations

import pytest

from ssa.types import VAD, Sentiment
from ssa.vad import (
    DistressQuadrant,
    ValenceThresholds,
    fit_valence_thresholds,
    vad_to_quadrant,
    valence_to_sentiment,
)


class TestValenceThresholds:
    def test_valid_thresholds_accepted(self) -> None:
        ValenceThresholds(low=0.3, high=0.7)

    def test_low_must_be_less_than_high(self) -> None:
        with pytest.raises(ValueError, match="0 <= low < high <= 1"):
            ValenceThresholds(low=0.7, high=0.3)

    def test_equal_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            ValenceThresholds(low=0.5, high=0.5)


class TestValenceToSentiment:
    def test_below_low_is_negative(self) -> None:
        t = ValenceThresholds(low=0.3, high=0.7)
        assert valence_to_sentiment(0.1, t) == Sentiment.NEGATIVE
        assert valence_to_sentiment(0.3, t) == Sentiment.NEGATIVE  # boundary is inclusive

    def test_above_high_is_positive(self) -> None:
        t = ValenceThresholds(low=0.3, high=0.7)
        assert valence_to_sentiment(0.9, t) == Sentiment.POSITIVE
        assert valence_to_sentiment(0.7, t) == Sentiment.POSITIVE  # boundary is inclusive

    def test_between_is_neutral(self) -> None:
        t = ValenceThresholds(low=0.3, high=0.7)
        assert valence_to_sentiment(0.5, t) == Sentiment.NEUTRAL


class TestFitValenceThresholds:
    def test_perfectly_separable_data_recovers_correct_ordering(self) -> None:
        """3 tight clusters at valence 0.1 / 0.5 / 0.9 -- any reasonable grid
        search should find thresholds that separate them perfectly."""
        valences = [*[0.05, 0.1, 0.15], *[0.45, 0.5, 0.55], *[0.85, 0.9, 0.95]]
        y_true = [
            *[Sentiment.NEGATIVE] * 3,
            *[Sentiment.NEUTRAL] * 3,
            *[Sentiment.POSITIVE] * 3,
        ]
        thresholds = fit_valence_thresholds(valences, y_true)
        preds = [valence_to_sentiment(v, thresholds) for v in valences]
        assert preds == y_true

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            fit_valence_thresholds([0.1, 0.2], [Sentiment.NEGATIVE])


class TestDistressQuadrant:
    def test_high_valence_is_fine_regardless_of_arousal(self) -> None:
        assert (
            vad_to_quadrant(VAD(valence=0.9, arousal=0.9, dominance=0.5)) == DistressQuadrant.FINE
        )
        assert (
            vad_to_quadrant(VAD(valence=0.9, arousal=0.1, dominance=0.5)) == DistressQuadrant.FINE
        )

    def test_low_valence_high_arousal_is_agitated(self) -> None:
        vad = VAD(valence=0.1, arousal=0.9, dominance=0.5)
        assert vad_to_quadrant(vad) == DistressQuadrant.AGITATED

    def test_low_valence_low_arousal_is_withdrawn(self) -> None:
        vad = VAD(valence=0.1, arousal=0.1, dominance=0.5)
        assert vad_to_quadrant(vad) == DistressQuadrant.WITHDRAWN
