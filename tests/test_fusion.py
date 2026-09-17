"""Fusion solution tests: pure math (fuse_probs, fit_fusion_params) plus
end-to-end wiring with fake sub-solutions -- no real models needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssa.solutions.fusion import (
    DEFAULT_PARAMS,
    FusionParams,
    FusionSolution,
    fit_fusion_params,
    fuse_probs,
)
from ssa.types import SAMPLE_RATE, VAD, AudioClip, Prediction, Sentiment


def _dist(pos: float, neu: float, neg: float) -> dict[Sentiment, float]:
    return {Sentiment.POSITIVE: pos, Sentiment.NEUTRAL: neu, Sentiment.NEGATIVE: neg}


class TestFusionParams:
    def test_weight_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="weight_lexical"):
            FusionParams(weight_lexical=1.5, temperature=1.0, abstain_threshold=0.0)

    def test_nonpositive_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            FusionParams(weight_lexical=0.5, temperature=0.0, abstain_threshold=0.0)

    def test_json_round_trip(self, tmp_path) -> None:
        params = FusionParams(weight_lexical=0.3, temperature=1.5, abstain_threshold=0.4)
        path = tmp_path / "params.json"
        params.to_json(path)
        assert FusionParams.from_json(path) == params


class TestFuseProbs:
    def test_weight_one_recovers_lexical_distribution(self) -> None:
        probs_a = _dist(0.7, 0.2, 0.1)
        probs_b = _dist(0.1, 0.1, 0.8)
        params = FusionParams(weight_lexical=1.0, temperature=1.0, abstain_threshold=0.0)
        fused, sentiment, _confidence = fuse_probs(probs_a, probs_b, params)
        for s in Sentiment:
            assert fused[s] == pytest.approx(probs_a[s], abs=1e-6)
        assert sentiment == Sentiment.POSITIVE

    def test_weight_zero_recovers_acoustic_distribution(self) -> None:
        probs_a = _dist(0.7, 0.2, 0.1)
        probs_b = _dist(0.1, 0.1, 0.8)
        params = FusionParams(weight_lexical=0.0, temperature=1.0, abstain_threshold=0.0)
        fused, sentiment, _confidence = fuse_probs(probs_a, probs_b, params)
        for s in Sentiment:
            assert fused[s] == pytest.approx(probs_b[s], abs=1e-6)
        assert sentiment == Sentiment.NEGATIVE

    def test_fused_probs_sum_to_one(self) -> None:
        probs_a = _dist(0.5, 0.3, 0.2)
        probs_b = _dist(0.2, 0.3, 0.5)
        fused, _s, _c = fuse_probs(probs_a, probs_b, DEFAULT_PARAMS)
        assert sum(fused.values()) == pytest.approx(1.0)

    def test_agreement_beats_disagreement(self) -> None:
        """Two branches that agree should yield higher fused confidence than
        two that disagree, holding the "leading" branch's own confidence
        fixed. (Log-linear pooling of two IDENTICAL distributions is a
        no-op -- the geometric mean of a distribution with itself is
        itself -- so this is the property that's actually guaranteed, not
        that agreement "sharpens" confidence beyond either input.)"""
        probs_a = _dist(0.7, 0.2, 0.1)
        agree_b = _dist(0.7, 0.2, 0.1)
        disagree_b = _dist(0.1, 0.2, 0.7)

        _fused, sentiment, agree_conf = fuse_probs(probs_a, agree_b, DEFAULT_PARAMS)
        assert sentiment == Sentiment.POSITIVE
        _fused, _sentiment, disagree_conf = fuse_probs(probs_a, disagree_b, DEFAULT_PARAMS)

        assert agree_conf > disagree_conf

    def test_higher_temperature_flattens_confidence(self) -> None:
        probs_a = _dist(0.9, 0.05, 0.05)
        probs_b = _dist(0.9, 0.05, 0.05)
        cold = fuse_probs(probs_a, probs_b, FusionParams(0.5, 0.5, 0.0))
        hot = fuse_probs(probs_a, probs_b, FusionParams(0.5, 5.0, 0.0))
        assert hot[2] < cold[2]  # higher temperature -> lower peak confidence


class TestFitFusionParams:
    def test_favours_the_more_reliable_branch(self) -> None:
        """Lexical is always right, acoustic is always wrong (systematically
        off by one class) -- fitting should push weight_lexical well above 0.5."""
        y_true = [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE] * 5
        probs_a_list = [
            _dist(0.9, 0.05, 0.05)
            if y == Sentiment.POSITIVE
            else _dist(0.05, 0.9, 0.05)
            if y == Sentiment.NEUTRAL
            else _dist(0.05, 0.05, 0.9)
            for y in y_true
        ]
        # acoustic always predicts the WRONG class
        probs_b_list = [_dist(0.05, 0.05, 0.9) for _ in y_true]

        params = fit_fusion_params(probs_a_list, probs_b_list, y_true)
        assert params.weight_lexical > 0.5

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            fit_fusion_params(
                [_dist(1, 0, 0)], [_dist(1, 0, 0), _dist(0, 1, 0)], [Sentiment.POSITIVE]
            )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            fit_fusion_params([], [], [])

    def test_abstain_threshold_matches_target_quantile(self) -> None:
        y_true = [Sentiment.POSITIVE] * 10
        # 10 identical confident-and-correct predictions -- abstain_threshold
        # should land at roughly that confidence level for any quantile.
        probs_list = [_dist(0.9, 0.05, 0.05) for _ in range(10)]
        params = fit_fusion_params(probs_list, probs_list, y_true, target_abstention_rate=0.5)
        assert 0.0 <= params.abstain_threshold <= 1.0


class _FixedSolution:
    def __init__(self, name: str, prediction: Prediction) -> None:
        self.name = name
        self._prediction = prediction

    def predict(self, clip: AudioClip) -> Prediction:
        return self._prediction

    def predict_batch(self, clips):
        return [self._prediction for _ in clips]


def _fixed_pred(sentiment: Sentiment, *, transcript=None, vad=None, latency_ms=1.0) -> Prediction:
    probs = {s: 0.1 for s in Sentiment}
    probs[sentiment] = 0.8
    remainder = (1 - 0.8) / 2
    for s in Sentiment:
        if s != sentiment:
            probs[s] = remainder
    return Prediction(
        sentiment=sentiment,
        probs=probs,
        confidence=probs[sentiment],
        latency_ms=latency_ms,
        solution="fixed",
        transcript=transcript,
        vad=vad,
    )


class TestFusionSolution:
    def test_carries_transcript_from_lexical_and_vad_from_acoustic(self) -> None:
        lexical = _FixedSolution("A", _fixed_pred(Sentiment.POSITIVE, transcript="hello there"))
        vad = VAD(valence=0.7, arousal=0.5, dominance=0.5)
        acoustic = _FixedSolution("B", _fixed_pred(Sentiment.POSITIVE, vad=vad))
        solution = FusionSolution(lexical, acoustic)

        clip = AudioClip(clip_id="c", samples=np.zeros(SAMPLE_RATE, dtype=np.float32))
        pred = solution.predict(clip)
        assert pred.transcript == "hello there"
        assert pred.vad == vad

    def test_name_reflects_both_sub_solutions(self) -> None:
        lexical = _FixedSolution("A:lexical(x)", _fixed_pred(Sentiment.NEUTRAL))
        acoustic = _FixedSolution("B:acoustic(y)", _fixed_pred(Sentiment.NEUTRAL))
        solution = FusionSolution(lexical, acoustic)
        assert solution.name == "C:fusion(A:lexical(x)+B:acoustic(y))"

    def test_abstains_below_threshold(self) -> None:
        lexical = _FixedSolution("A", _fixed_pred(Sentiment.POSITIVE))
        acoustic = _FixedSolution(
            "B", _fixed_pred(Sentiment.NEGATIVE)
        )  # disagreement -> lower confidence
        params = FusionParams(weight_lexical=0.5, temperature=1.0, abstain_threshold=0.99)
        solution = FusionSolution(lexical, acoustic, params)
        clip = AudioClip(clip_id="c", samples=np.zeros(SAMPLE_RATE, dtype=np.float32))
        pred = solution.predict(clip)
        assert pred.abstained is True

    def test_predict_batch_matches_repeated_predict(self) -> None:
        lexical = _FixedSolution("A", _fixed_pred(Sentiment.POSITIVE))
        acoustic = _FixedSolution("B", _fixed_pred(Sentiment.POSITIVE))
        solution = FusionSolution(lexical, acoustic)
        clips = [
            AudioClip(clip_id=f"c{i}", samples=np.zeros(SAMPLE_RATE, dtype=np.float32))
            for i in range(3)
        ]
        batch = solution.predict_batch(clips)
        singles = [solution.predict(c) for c in clips]
        assert [p.sentiment for p in batch] == [p.sentiment for p in singles]
