"""Tests for the Hume probe's pure grid/summary/recoverability logic --
no API calls, no audio. Mirrors tests/test_gen_emotion_probe_d1.py's
approach for the same method applied to Hume instead of Cartesia."""

from __future__ import annotations

import pytest

from scripts.probe_hume import (
    DESCRIPTIONS,
    LENGTHS,
    TEXT_CONDITIONS,
    Measurement,
    build_grid,
    recoverability_cv,
    summarize,
)
from ssa.carriers import D1_CONGRUENT_TEXT, D1_NEUTRAL_TEXT, EMOTIONS


def _measurement(clip_id: str, **overrides) -> Measurement:
    base = dict(
        clip_id=clip_id,
        f0_mean=150.0,
        f0_std=10.0,
        f0_range=30.0,
        egemaps_loudness=0.5,
        egemaps_hnr=15.0,
        egemaps_jitter=0.01,
        egemaps_shimmer_db=0.5,
        rms=0.1,
        speech_rate_cps=15.0,
        research_valence=0.5,
        research_arousal=0.5,
        permissive_predicted_sentiment="neutral",
        permissive_valence_proxy=0.0,
    )
    base.update(overrides)
    return Measurement(**base)


class TestBuildGrid:
    def test_forty_clips(self) -> None:
        grid = build_grid()
        assert len(grid) == 5 * len(TEXT_CONDITIONS) * len(LENGTHS) * 2

    def test_every_factor_combination_present_once(self) -> None:
        grid = build_grid()
        combos = {(s.emotion, s.text_condition, s.length, s.has_description) for s in grid}
        assert len(combos) == len(grid)

    def test_clip_ids_unique(self) -> None:
        grid = build_grid()
        ids = [s.clip_id for s in grid]
        assert len(ids) == len(set(ids))

    def test_neutral_text_identical_across_emotions_at_fixed_length(self) -> None:
        grid = build_grid()
        for length in LENGTHS:
            texts = {s.text for s in grid if s.text_condition == "neutral" and s.length == length}
            assert texts == {D1_NEUTRAL_TEXT[length]}

    def test_congruent_text_matches_per_emotion_carrier(self) -> None:
        grid = build_grid()
        for spec in grid:
            if spec.text_condition == "congruent":
                assert spec.text == D1_CONGRUENT_TEXT[spec.emotion][spec.length]

    def test_carrier_groups_two_neutral_plus_ten_congruent(self) -> None:
        grid = build_grid()
        neutral_groups = {s.carrier_group for s in grid if s.text_condition == "neutral"}
        congruent_groups = {s.carrier_group for s in grid if s.text_condition == "congruent"}
        assert len(neutral_groups) == 2
        assert len(congruent_groups) == 10


class TestDescriptions:
    def test_every_emotion_has_a_description(self) -> None:
        assert set(DESCRIPTIONS) == set(EMOTIONS)

    def test_descriptions_are_short(self) -> None:
        for desc in DESCRIPTIONS.values():
            assert len(desc) <= 100


class TestRecoverabilityCV:
    def test_perfectly_separable_scores_above_chance(self) -> None:
        grid = build_grid()
        emotion_f0 = {
            "happy": 250.0,
            "sad": 90.0,
            "angry": 220.0,
            "calm": 110.0,
            "frustrated": 180.0,
        }
        measurements = {
            s.clip_id: _measurement(s.clip_id, f0_mean=emotion_f0[s.emotion]) for s in grid
        }
        result = recoverability_cv(grid, measurements)
        assert result["accuracy"] is not None
        assert result["accuracy"] > result["chance"]
        assert result["chance"] == pytest.approx(0.2)
        assert result["n_carrier_groups"] == 12

    def test_missing_features_dropped_not_crashed(self) -> None:
        grid = build_grid()
        measurements = {s.clip_id: _measurement(s.clip_id) for s in grid}
        first_id = grid[0].clip_id
        measurements[first_id] = _measurement(first_id, f0_mean=None)
        result = recoverability_cv(grid, measurements)
        assert result["n_dropped_missing_features"] == 1


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

    def test_f0_span_computed_per_description_group(self) -> None:
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
        assert summary["with_description"]["f0_span_hz"] == pytest.approx(130.0)

    def test_missing_valence_gives_none_ordering(self) -> None:
        grid = build_grid()
        measurements = {s.clip_id: _measurement(s.clip_id, research_valence=None) for s in grid}
        summary = summarize(grid, measurements)
        assert summary["with_description"]["valence_ordering_matches_intended_sentiment"] is None

    def test_neutral_text_cut_has_twenty_clips_split_by_description(self) -> None:
        grid = build_grid()
        measurements = {s.clip_id: _measurement(s.clip_id) for s in grid}
        summary = summarize(grid, measurements)
        # 5 emotions x 2 lengths = 10 clips per description condition, neutral text only
        assert summary["neutral_text_with_description"]["n_clips"] == 10
        assert summary["congruent_text_with_description"]["n_clips"] == 10

    def test_with_and_without_description_each_cover_half_the_grid(self) -> None:
        grid = build_grid()
        measurements = {s.clip_id: _measurement(s.clip_id) for s in grid}
        summary = summarize(grid, measurements)
        assert summary["with_description"]["n_clips"] == 20
        assert summary["without_description"]["n_clips"] == 20
