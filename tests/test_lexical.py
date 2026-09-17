"""Lexical solution tests.

_probs_from_raw and _abstain are pure functions, tested without loading any
model. Full end-to-end tests (real ASR + real text classifier) are marked
@pytest.mark.network and skipped by default -- see pyproject.toml.
"""

from __future__ import annotations

import math

import pytest

from ssa.solutions.lexical import LexicalSolution, _abstain, _probs_from_raw
from ssa.types import Sentiment


class TestProbsFromRaw:
    def test_maps_known_labels(self) -> None:
        raw = [
            {"label": "positive", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "negative", "score": 0.1},
        ]
        probs = _probs_from_raw(raw)
        assert probs[Sentiment.POSITIVE] == pytest.approx(0.7)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_normalises_non_unit_sum(self) -> None:
        """Defensive: scores that don't sum to 1 (e.g. a different
        pipeline config) are renormalised rather than trusted verbatim."""
        raw = [
            {"label": "positive", "score": 0.9},
            {"label": "neutral", "score": 0.9},
            {"label": "negative", "score": 0.9},
        ]
        probs = _probs_from_raw(raw)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert probs[Sentiment.POSITIVE] == pytest.approx(1 / 3)

    def test_unknown_label_raises(self) -> None:
        raw = [
            {"label": "LABEL_0", "score": 0.5},
            {"label": "LABEL_1", "score": 0.3},
            {"label": "LABEL_2", "score": 0.2},
        ]
        with pytest.raises(ValueError, match="don't map onto Sentiment"):
            _probs_from_raw(raw)

    def test_missing_a_sentiment_raises(self) -> None:
        raw = [{"label": "positive", "score": 0.6}, {"label": "negative", "score": 0.4}]
        with pytest.raises(ValueError, match="expected all 3 sentiments"):
            _probs_from_raw(raw)


class TestAbstain:
    def test_uniform_probs_and_neutral_fallback(self) -> None:
        pred = _abstain("test-solution", 5.0, transcript="")
        assert pred.abstained is True
        assert pred.sentiment == Sentiment.NEUTRAL
        assert all(math.isclose(p, 1 / 3) for p in pred.probs.values())
        assert pred.confidence == pytest.approx(1 / 3)
        assert pred.transcript == ""

    def test_probs_sum_to_one(self) -> None:
        pred = _abstain("s", 1.0, transcript="")
        assert sum(pred.probs.values()) == pytest.approx(1.0)


@pytest.mark.network
@pytest.mark.slow
class TestLexicalSolutionEndToEnd:
    """Requires downloading faster-whisper + the text classifier."""

    @classmethod
    @pytest.fixture(scope="class")
    def solution(cls) -> LexicalSolution:
        return LexicalSolution()

    def test_transcribes_and_classifies(self, solution: LexicalSolution) -> None:
        import numpy as np

        from ssa.types import AudioClip

        # silence should abstain
        silent = AudioClip(clip_id="silence", samples=np.zeros(16000, dtype="float32"))
        pred = solution.predict(silent)
        assert pred.abstained is True
        assert pred.transcript == ""
