"""Tests for the Hume probe's pure grid/summary logic -- no API calls."""

from __future__ import annotations

from scripts.probe_hume import CARRIER_TEXT, DESCRIPTIONS, Measurement, build_grid, summarize
from ssa.carriers import EMOTIONS


def _measurement(clip_id: str, **overrides) -> Measurement:
    base = dict(
        clip_id=clip_id,
        f0_mean=150.0,
        f0_std=10.0,
        research_valence=0.5,
        research_arousal=0.5,
        permissive_predicted_sentiment="neutral",
        permissive_valence_proxy=0.0,
    )
    base.update(overrides)
    return Measurement(**base)


class TestBuildGrid:
    def test_ten_clips_five_emotions_times_two(self) -> None:
        grid = build_grid()
        assert len(grid) == 10
        assert {s.emotion for s in grid} == set(EMOTIONS)

    def test_each_emotion_has_description_and_nodescription_variant(self) -> None:
        grid = build_grid()
        for emotion in EMOTIONS:
            variants = {s.has_description for s in grid if s.emotion == emotion}
            assert variants == {True, False}

    def test_clip_ids_unique(self) -> None:
        grid = build_grid()
        ids = [s.clip_id for s in grid]
        assert len(ids) == len(set(ids))


class TestDescriptionsAndCarrierText:
    def test_every_emotion_has_a_description(self) -> None:
        assert set(DESCRIPTIONS) == set(EMOTIONS)

    def test_descriptions_are_short(self) -> None:
        for desc in DESCRIPTIONS.values():
            assert len(desc) <= 100

    def test_carrier_text_matches_d1_neutral_short(self) -> None:
        from ssa.carriers import D1_NEUTRAL_TEXT

        assert D1_NEUTRAL_TEXT["short"] == CARRIER_TEXT


class TestSummarize:
    def test_valence_ordering_true_when_correctly_ordered(self) -> None:
        grid = build_grid()
        emotion_valence = {"happy": 0.7, "calm": 0.5, "sad": 0.3, "angry": 0.2, "frustrated": 0.25}
        measurements = {
            s.clip_id: _measurement(s.clip_id, research_valence=emotion_valence[s.emotion])
            for s in grid
        }
        summary = summarize(grid, measurements)
        assert summary["with_description"]["valence_ordering_matches_intended_sentiment"] is True
        assert summary["without_description"]["valence_ordering_matches_intended_sentiment"] is True

    def test_f0_span_computed_per_group(self) -> None:
        grid = build_grid()
        emotion_f0 = {
            "happy": 220.0,
            "sad": 90.0,
            "angry": 180.0,
            "calm": 110.0,
            "frustrated": 160.0,
        }
        measurements = {
            s.clip_id: _measurement(s.clip_id, f0_mean=emotion_f0[s.emotion]) for s in grid
        }
        summary = summarize(grid, measurements)
        assert summary["with_description"]["f0_span_hz"] == 130.0

    def test_missing_valence_gives_none_ordering(self) -> None:
        grid = build_grid()
        measurements = {s.clip_id: _measurement(s.clip_id, research_valence=None) for s in grid}
        summary = summarize(grid, measurements)
        assert summary["with_description"]["valence_ordering_matches_intended_sentiment"] is None
