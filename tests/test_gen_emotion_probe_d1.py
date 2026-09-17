"""Tests for D1's pure grid/analysis logic -- no TTS/API calls, no audio."""

from __future__ import annotations

import pytest

from scripts.gen_emotion_probe_d1 import (
    LENGTHS,
    SPEED_CONDITIONS,
    TEXT_CONDITIONS,
    D1ClipSpec,
    Measurement,
    _sentiment_ordering_holds,
    build_grid,
    build_model_comparison,
    pick_best_cell,
    recoverability_cv,
)
from ssa.carriers import D1_CONGRUENT_TEXT, D1_NEUTRAL_TEXT, D1_SPEED_ADJUSTED, EMOTIONS


def _measurement(clip_id: str, **overrides: float | str | None) -> Measurement:
    base = dict(
        clip_id=clip_id,
        duration_s=2.0,
        rms=0.1,
        speech_rate_cps=15.0,
        f0_mean=150.0,
        f0_std=10.0,
        f0_range=30.0,
        egemaps_loudness=0.5,
        egemaps_hnr=15.0,
        egemaps_jitter=0.01,
        egemaps_shimmer_db=0.5,
        permissive_valence_proxy=0.0,
        permissive_predicted_sentiment="neutral",
        research_valence=0.5,
        research_arousal=0.5,
    )
    base.update(overrides)
    return Measurement(**base)  # type: ignore[arg-type]


class TestBuildGrid:
    def test_exact_size(self) -> None:
        grid = build_grid()
        assert len(grid) == 5 * len(TEXT_CONDITIONS) * len(LENGTHS) * len(SPEED_CONDITIONS)
        assert len(grid) == 40

    def test_clip_ids_unique(self) -> None:
        grid = build_grid()
        ids = [s.clip_id for s in grid]
        assert len(ids) == len(set(ids))

    def test_every_factor_combination_present_exactly_once(self) -> None:
        grid = build_grid()
        combos = {(s.emotion, s.text_condition, s.length, s.speed_condition) for s in grid}
        assert len(combos) == len(grid)

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

    def test_default_speed_condition_omits_speed(self) -> None:
        grid = build_grid()
        assert all(s.speed is None for s in grid if s.speed_condition == "default")

    def test_adjusted_speed_condition_uses_emotion_table(self) -> None:
        grid = build_grid()
        for spec in grid:
            if spec.speed_condition == "adjusted":
                assert spec.speed == D1_SPEED_ADJUSTED[spec.emotion]

    def test_carrier_groups_are_two_neutral_plus_ten_congruent(self) -> None:
        grid = build_grid()
        neutral_groups = {s.carrier_group for s in grid if s.text_condition == "neutral"}
        congruent_groups = {s.carrier_group for s in grid if s.text_condition == "congruent"}
        assert len(neutral_groups) == 2  # one per length, shared across all 5 emotions
        assert len(congruent_groups) == 5 * 2  # one per (emotion, length)
        assert neutral_groups.isdisjoint(congruent_groups)

    def test_model_id_defaults_to_sonic3(self) -> None:
        grid = build_grid()
        assert all(s.model_id == "sonic-3" for s in grid)


class TestPickBestCell:
    def test_picks_cell_with_widest_f0_spread(self) -> None:
        grid = build_grid()
        f0_means = {}
        for spec in grid:
            # every cell gets a near-identical value except one, which gets
            # a deliberately wide spread across its 5 emotions
            target_cell = ("neutral", "short", "default")
            if spec.cell == target_cell:
                f0_means[spec.clip_id] = {"happy": 250.0, "sad": 90.0}.get(spec.emotion, 150.0)
            else:
                f0_means[spec.clip_id] = 150.0 + hash(spec.clip_id) % 3  # tiny noise, narrow spread

        best = pick_best_cell(grid, f0_means)
        assert best == ("neutral", "short", "default")

    def test_raises_when_no_cell_has_two_measurements(self) -> None:
        grid = build_grid()
        f0_means = {grid[0].clip_id: 150.0}  # only one clip measured at all
        with pytest.raises(ValueError, match="no cell had at least 2"):
            pick_best_cell(grid, f0_means)

    def test_ignores_none_measurements(self) -> None:
        grid = build_grid()
        f0_means = {s.clip_id: None for s in grid}
        target_cell = ("congruent", "long", "adjusted")
        cell_specs = [s for s in grid if s.cell == target_cell]
        f0_means[cell_specs[0].clip_id] = 100.0
        f0_means[cell_specs[1].clip_id] = 200.0
        assert pick_best_cell(grid, f0_means) == target_cell


class TestBuildModelComparison:
    def test_five_clips_one_per_emotion(self) -> None:
        comparison = build_model_comparison(("neutral", "short", "default"))
        assert len(comparison) == 5
        assert {s.emotion for s in comparison} == set(EMOTIONS)

    def test_uses_alt_model_id(self) -> None:
        comparison = build_model_comparison(("neutral", "short", "default"))
        assert all(s.model_id == "sonic-3.5" for s in comparison)

    def test_clip_ids_distinct_from_grid(self) -> None:
        grid = build_grid()
        comparison = build_model_comparison(("congruent", "long", "adjusted"))
        assert set(s.clip_id for s in comparison).isdisjoint(s.clip_id for s in grid)

    def test_text_and_speed_match_the_source_cell(self) -> None:
        best_cell = ("congruent", "long", "adjusted")
        comparison = build_model_comparison(best_cell)
        for spec in comparison:
            assert spec.text == D1_CONGRUENT_TEXT[spec.emotion]["long"]
            assert spec.speed == D1_SPEED_ADJUSTED[spec.emotion]


class TestRecoverabilityCV:
    def test_perfectly_separable_features_score_well_above_chance(self) -> None:
        grid = build_grid()
        # f0_mean alone perfectly separates the 5 emotions; everything else
        # is held constant so this really is testing the CV plumbing.
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

    def test_rows_with_missing_features_are_dropped(self) -> None:
        grid = build_grid()
        measurements = {s.clip_id: _measurement(s.clip_id) for s in grid}
        # knock out one clip's f0_mean -- it must be excluded, not crash the fit
        first_id = grid[0].clip_id
        measurements[first_id] = _measurement(first_id, f0_mean=None)
        result = recoverability_cv(grid, measurements)
        assert result["n_dropped_missing_features"] == 1

    def test_fewer_than_two_groups_skips_cv(self) -> None:
        # A hand-built tiny grid sharing a single carrier_group.
        specs = [
            D1ClipSpec(
                clip_id=f"x{i}",
                emotion="happy",
                text_condition="neutral",
                length="short",
                speed_condition="default",
                text="hi",
                speed=None,
                model_id="sonic-3",
                carrier_group="only_group",
            )
            for i in range(3)
        ]
        measurements = {s.clip_id: _measurement(s.clip_id) for s in specs}
        result = recoverability_cv(specs, measurements)
        assert result["accuracy"] is None
        assert "note" in result


class TestSentimentOrderingHolds:
    def test_true_when_ordered_correctly(self) -> None:
        summary = {
            "happy": {"field": 0.5},
            "calm": {"field": 0.0},
            "sad": {"field": -0.5},
            "angry": {"field": -0.6},
            "frustrated": {"field": -0.4},
        }
        assert _sentiment_ordering_holds(summary, "field") is True

    def test_false_when_scrambled(self) -> None:
        """Mirrors D0's actual finding: the negative bucket (dragged up by
        frustrated) reads higher, on average, than the positive bucket."""
        summary = {
            "happy": {"field": 0.1},
            "calm": {"field": 0.0},
            "sad": {"field": -0.1},
            "angry": {"field": -0.2},
            "frustrated": {"field": 1.0},  # scrambles the negative bucket's mean above positive's
        }
        assert _sentiment_ordering_holds(summary, "field") is False

    def test_none_when_a_value_missing(self) -> None:
        summary = {
            "happy": {"field": 0.5},
            "calm": {"field": None},
            "sad": {"field": -0.5},
            "angry": {"field": -0.6},
            "frustrated": {"field": -0.4},
        }
        assert _sentiment_ordering_holds(summary, "field") is None
