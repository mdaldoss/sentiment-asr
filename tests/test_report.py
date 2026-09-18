"""Report generator tests: pure rendering functions, fed synthetic result
dicts -- no real evaluation runs needed. The point is that report.py never
recomputes a metric (CLAUDE.md), so these tests check it renders exactly
what it's given, and degrades gracefully when data is missing."""

from __future__ import annotations

from ssa.report import (
    _fmt,
    _pct,
    _solution_short,
    render_backend_combo_section,
    render_d1_section,
    render_e3_control_section,
    render_hume_section,
    render_leakage_table,
    render_main_table,
    render_models_section,
    render_prosody_samples_section,
    render_status,
    render_what_didnt_work,
)


def _fake_eval_result(**overrides) -> dict:
    base = {
        "solution": "B:acoustic(permissive/logreg)",
        "dataset": "cremad",
        "split_type": "speaker_disjoint_test",
        "n_clips": 1470,
        "n_incongruent": 1255,
        "uar": 0.744,
        "macro_f1": 0.744,
        "accuracy": 0.816,
        "ece": 0.05,
        "psi_contested": 0.945,
        "psi_strict": 0.9,
        "confusion_matrix": [[886, 40, 78], [51, 149, 15], [65, 21, 165]],
        "confusion_labels": ["negative", "neutral", "positive"],
        "abstention_rate": 0.0,
        "latency_p50_ms": 50.0,
        "latency_p95_ms": 80.0,
        "git_sha": "abc123",
        "timestamp": "2026-09-17T00:00:00Z",
        "predictions": [],
    }
    base.update(overrides)
    return base


class TestFormatHelpers:
    def test_pct_formats_fraction(self) -> None:
        assert _pct(0.123) == "12.3%"

    def test_pct_handles_none(self) -> None:
        assert _pct(None) == "&mdash;"

    def test_pct_handles_nan(self) -> None:
        assert _pct(float("nan")) == "&mdash;"

    def test_fmt_default_digits(self) -> None:
        assert _fmt(0.74444) == "0.744"

    def test_fmt_custom_digits(self) -> None:
        assert _fmt(0.74444, digits=1) == "0.7"

    def test_fmt_none_is_dash(self) -> None:
        assert _fmt(None) == "&mdash;"


class TestSolutionShort:
    def test_lexical_label(self) -> None:
        assert "Lexical" in _solution_short("A:lexical(small+twitter-roberta)")

    def test_acoustic_permissive_label(self) -> None:
        label = _solution_short("B:acoustic(permissive/logreg)")
        assert "Acoustic" in label and "permissive" in label

    def test_acoustic_research_label(self) -> None:
        label = _solution_short("B:acoustic(research/audeering-vad)")
        assert "Acoustic" in label and "research" in label

    def test_fusion_label(self) -> None:
        assert "Fusion" in _solution_short("C:fusion(A:x+B:y)")


class TestRenderMainTable:
    def test_empty_results_shows_pending(self) -> None:
        html = render_main_table([])
        assert "pending" in html

    def test_renders_a_row_per_result(self) -> None:
        results = [_fake_eval_result(), _fake_eval_result(solution="A:lexical(x)")]
        html = render_main_table(results)
        # count body rows only -- the header row also contains a <tr>
        body = html.split("<tbody>", 1)[1]
        assert body.count("<tr>") == 2

    def test_missing_optional_field_does_not_crash(self) -> None:
        """A result dict missing e.g. psi_contested (an older format, or a
        solution that doesn't compute it) must render '&mdash;', not crash."""
        incomplete = _fake_eval_result()
        del incomplete["psi_contested"]
        html = render_main_table([incomplete])
        assert "&mdash;" in html


class TestRenderLeakageTable:
    def test_both_none_shows_pending(self) -> None:
        html = render_leakage_table(None, None)
        assert "pending" in html

    def test_renders_gap_when_both_present(self) -> None:
        disjoint = _fake_eval_result(uar=0.744)
        leaky = {
            "n_clips": 1488,
            "n_speakers_overlapping_with_train": 91,
            "uar": 0.761,
            "macro_f1": 0.743,
        }
        html = render_leakage_table(disjoint, leaky)
        assert "0.744" in html and "0.761" in html
        assert "+0.017" in html


class TestRenderStatus:
    def test_e1_always_marked_done(self) -> None:
        html = render_status([])
        assert "done" in html
        assert "CREMA-D" in html


def _fake_d1() -> dict:
    return {
        "grid_size": 40,
        "comparison_size": 5,
        "recoverability_cv": {"accuracy": 0.175, "chance": 0.2},
        "f0_span_across_emotions_hz": 7.98,
        "f0_rank_consistency": {
            "chance_mean_rank": 3.0,
            "mean_rank_by_emotion": {
                "happy": 2.75,
                "sad": 3.125,
                "angry": 2.875,
                "calm": 3.0,
                "frustrated": 3.25,
            },
        },
        "valence_ordering_matches_intended_sentiment": {"permissive": False, "research": True},
        "interpretation": "coarse synthetic probe, see CLAUDE.md rule 6",
    }


def _fake_e3_control() -> dict:
    return {
        "cartesia_d1": {
            "recoverability_accuracy": 0.175,
            "recoverability_chance": 0.2,
            "f0_span_hz": 7.98,
        },
        "human_e3": {
            "e3a": {
                "recoverability_cv": {"accuracy": 0.667, "chance": 0.333},
                "f0_span_hz": 32.9,
                "f0_rank_consistency": {"mean_rank_by_sentiment": {"positive": 1.0}},
            },
            "e3b": {
                "recoverability_cv": {"accuracy": 0.741, "chance": 0.333},
                "f0_span_hz": 33.9,
                "f0_rank_consistency": {"mean_rank_by_sentiment": {"positive": 1.0}},
            },
        },
        "test_retest_correlation_take0_vs_take1": {
            "f0_mean": 0.785,
            "rms": 0.410,
            "research_valence": 0.921,
            "permissive_valence_proxy": -0.191,
        },
        "interpretation": "same instruments as D1, run on human speech",
    }


class TestRenderD1Section:
    def test_none_shows_pending(self) -> None:
        assert "pending" in render_d1_section(None)

    def test_renders_recoverability_and_chance(self) -> None:
        html = render_d1_section(_fake_d1())
        assert "0.175" in html
        assert "0.200" in html

    def test_renders_all_five_emotion_ranks(self) -> None:
        html = render_d1_section(_fake_d1())
        for emotion in ("happy", "sad", "angry", "calm", "frustrated"):
            assert emotion in html

    def test_renders_interpretation(self) -> None:
        html = render_d1_section(_fake_d1())
        assert "coarse synthetic probe" in html


class TestRenderE3ControlSection:
    def test_none_shows_pending(self) -> None:
        assert "pending" in render_e3_control_section(None)

    def test_renders_cartesia_and_both_takes(self) -> None:
        html = render_e3_control_section(_fake_e3_control())
        assert "0.175" in html
        assert "0.667" in html
        assert "0.741" in html

    def test_renders_test_retest_correlations(self) -> None:
        html = render_e3_control_section(_fake_e3_control())
        assert "0.92" in html  # research_valence r, rounded to 2dp by _fmt
        assert "-0.19" in html  # permissive_valence_proxy r


class TestRenderWhatDidntWork:
    def test_mentions_every_known_failure(self) -> None:
        html = render_what_didnt_work()
        for marker in ("D0", "D1", "E2", "Solution B", "recalibration", "Live browser demo"):
            assert marker in html


def _fake_prosody_samples() -> dict:
    return {
        "n_samples": 3,
        "n_cremad": 2,
        "n_e3": 1,
        "model": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (Wagner et al. 2023)",
        "samples": [
            {
                "clip_id": "cremad_x",
                "source": "CREMA-D (public)",
                "label": "HAP",
                "valence": 0.5,
                "arousal": 0.7,
                "f0_mean": 214.0,
            },
            {
                "clip_id": "cremad_y",
                "source": "CREMA-D (public)",
                "label": "SAD",
                "valence": 0.26,
                "arousal": 0.0,
                "f0_mean": 91.0,
            },
            {
                "clip_id": "e3a_z",
                "source": "E3 (mine, e3a)",
                "label": "positive",
                "valence": 0.44,
                "arousal": 0.45,
                "f0_mean": 151.0,
            },
        ],
    }


class TestRenderProsodySamplesSection:
    def test_none_shows_pending(self) -> None:
        assert "pending" in render_prosody_samples_section(None)

    def test_renders_all_clip_ids(self) -> None:
        html = render_prosody_samples_section(_fake_prosody_samples())
        assert "cremad_x" in html and "cremad_y" in html and "e3a_z" in html

    def test_renders_model_citation(self) -> None:
        html = render_prosody_samples_section(_fake_prosody_samples())
        assert "Wagner et al. 2023" in html

    def test_distinguishes_circle_and_square_markers(self) -> None:
        html = render_prosody_samples_section(_fake_prosody_samples())
        assert "<circle" in html  # CREMA-D
        assert "<rect" in html  # E3 (mine)


class TestRenderModelsSection:
    def test_names_both_papers(self) -> None:
        html = render_models_section()
        assert "2110.13900" in html
        assert "2203.07378" in html

    def test_clarifies_wavlm_is_not_the_emotion_model(self) -> None:
        html = render_models_section()
        assert "never on an emotion label" in html


def _fake_hume() -> dict:
    return {
        "n_clips": 40,
        "voice": "Ava Song",
        "summary": {
            "with_description": {
                "n_clips": 20,
                "f0_mean_by_emotion": {"happy": 234.3, "sad": 150.7, "angry": 254.1},
                "f0_span_hz": 126.7,
                "valence_mean_by_emotion": {"happy": 0.64, "sad": 0.55, "angry": 0.46},
                "valence_ordering_matches_intended_sentiment": True,
                "recoverability_cv": {"accuracy": 0.55, "chance": 0.2, "n_carrier_groups": 12},
            },
            "without_description": {
                "n_clips": 20,
                "f0_mean_by_emotion": {"happy": 163.5, "sad": 157.8, "angry": 183.4},
                "f0_span_hz": 32.3,
                "valence_mean_by_emotion": {"happy": 0.57, "sad": 0.49, "angry": 0.49},
                "valence_ordering_matches_intended_sentiment": False,
                "recoverability_cv": {"accuracy": 0.15, "chance": 0.2, "n_carrier_groups": 12},
            },
            "neutral_text_with_description": {
                "n_clips": 10,
                "f0_span_hz": 90.0,
                "valence_range": 0.3,
            },
            "congruent_text_with_description": {
                "n_clips": 10,
                "f0_span_hz": 140.0,
                "valence_range": 0.4,
            },
        },
        "interpretation": "falsification test of D1's Cartesia finding",
    }


def _fake_backend_combos() -> dict:
    def eval_pair(uar_cremad: float, uar_e3: float | None) -> dict:
        cremad = {
            "dataset": "cremad_test",
            "split_type": "combo_x",
            "n_clips": 300,
            "uar": uar_cremad,
            "macro_f1": uar_cremad - 0.01,
        }
        e3 = (
            {
                "dataset": "e3_both_takes",
                "split_type": "combo_x",
                "n_clips": 54,
                "uar": uar_e3,
                "macro_f1": uar_e3 - 0.01,
            }
            if uar_e3 is not None
            else {"excluded_reason": "combo trains on E3's speaker"}
        )
        return {"cremad_test": cremad, "e3_both_takes": e3}

    research_eval = eval_pair(0.5, 0.44)
    return {
        "eval_subset_n": 300,
        "eval_subset_seed": 0,
        "combos": {
            "cremad_only": {
                "description": "CREMA-D train split only.",
                "train_n_clips": 5235,
                "permissive": {"eval": eval_pair(0.60, 0.44)},
                "research": {"eval": research_eval},
            },
            "cremad_e3": {
                "description": "CREMA-D train + both E3 takes.",
                "train_n_clips": 5289,
                "permissive": {"eval": eval_pair(0.61, None)},
                "research": {"eval": research_eval},
            },
            "cremad_e3_hume": {
                "description": "CREMA-D train + both E3 takes + Hume.",
                "train_n_clips": 5309,
                "permissive": {"eval": eval_pair(0.615, None)},
                "research": {"eval": research_eval},
            },
            "cremad_hume": {
                "description": "CREMA-D train + Hume, no E3.",
                "train_n_clips": 5255,
                "permissive": {"eval": eval_pair(0.605, 0.46)},
                "research": {"eval": research_eval},
            },
        },
        "research_backend_note": "identical across combos by construction",
        "interpretation": "direct comparison of the two backends across four combos",
    }


class TestRenderBackendComboSection:
    def test_none_shows_pending(self) -> None:
        assert "pending" in render_backend_combo_section(None)

    def test_renders_all_four_combo_names(self) -> None:
        html = render_backend_combo_section(_fake_backend_combos())
        for name in ("cremad_only", "cremad_e3", "cremad_e3_hume", "cremad_hume"):
            assert name in html

    def test_excluded_e3_shown_as_excl_not_a_number(self) -> None:
        html = render_backend_combo_section(_fake_backend_combos())
        assert "excl.*" in html

    def test_renders_research_note_and_interpretation(self) -> None:
        html = render_backend_combo_section(_fake_backend_combos())
        assert "identical across combos by construction" in html
        assert "direct comparison of the two backends" in html

    def test_renders_included_e3_numbers(self) -> None:
        html = render_backend_combo_section(_fake_backend_combos())
        assert "0.440" in html or "0.44" in html


class TestRenderHumeSection:
    def test_none_shows_pending(self) -> None:
        assert "pending" in render_hume_section(None)

    def test_renders_both_f0_spans(self) -> None:
        html = render_hume_section(_fake_hume())
        assert "126.7" in html
        assert "32.3" in html

    def test_renders_ordering_results(self) -> None:
        html = render_hume_section(_fake_hume())
        assert "True" in html
        assert "False" in html

    def test_renders_interpretation(self) -> None:
        html = render_hume_section(_fake_hume())
        assert "falsification test" in html
