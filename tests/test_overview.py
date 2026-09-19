"""Overview-report tests. The invariant that matters here is the same one
report.py has: the page RENDERS numbers, it never invents them. So these
check that (a) real values reach the HTML, (b) a missing result file
degrades to an em-dash instead of crashing or fabricating, and (c) the
literature section stays clearly fenced off from our own measurements
(CLAUDE.md rule 6)."""

from __future__ import annotations

from typing import Any

import pytest

from ssa.overview import (
    _fmt,
    _get,
    build_artifact_page,
    render_improvements,
    render_results,
    render_state_of_the_art,
    render_what_is_possible,
    wrap_tables,
)


def _eval_result(**overrides: Any) -> dict[str, Any]:
    base = {
        "uar": 0.742,
        "macro_f1": 0.731,
        "accuracy": 0.810,
        "psi_contested": 0.912,
        "n_clips": 300,
    }
    base.update(overrides)
    return base


def _numbers(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "lex": _eval_result(uar=0.360, psi_contested=0.090),
        "perm_sub": _eval_result(uar=0.797, psi_contested=0.920),
        "fusion_sub": _eval_result(uar=0.797, psi_contested=0.891),
        "perm_full": _eval_result(uar=0.744, n_clips=1470),
        "leaky": {"uar": 0.761, "n_clips": 1488, "n_speakers_overlapping_with_train": 91},
        "research_cremad": _eval_result(uar=0.450),
        "control": {
            "test_retest_correlation_take0_vs_take1": {
                "research_valence": 0.921,
                "permissive_valence_proxy": -0.191,
            }
        },
        "d1": {"recoverability_cv": {"accuracy": 0.175}},
        "hume": {
            "summary": {
                "with_description": {"recoverability_cv": {"accuracy": 0.5}},
                "without_description": {"recoverability_cv": {"accuracy": 0.2}},
            }
        },
        "e3": {
            "lex_a": _eval_result(uar=0.333),
            "lex_b": _eval_result(uar=0.370),
            "perm_a": _eval_result(uar=0.444),
            "perm_b": _eval_result(uar=0.519),
            "res_a": _eval_result(uar=0.444),
            "res_b": _eval_result(uar=0.593),
        },
    }
    base.update(overrides)
    return base


class TestGetHelper:
    def test_walks_nested_keys(self) -> None:
        assert _get({"a": {"b": {"c": 1}}}, "a", "b", "c") == 1

    def test_none_anywhere_short_circuits(self) -> None:
        assert _get(None, "a", "b") is None

    def test_missing_key_returns_none_not_keyerror(self) -> None:
        assert _get({"a": {}}, "a", "missing") is None


class TestFmt:
    def test_rounds_to_requested_digits(self) -> None:
        assert _fmt(0.74444) == "0.744"
        assert _fmt(0.74444, 2) == "0.74"

    def test_none_is_dash(self) -> None:
        assert _fmt(None) == "&mdash;"


class TestRenderResults:
    def test_renders_every_solution_uar(self) -> None:
        html = render_results(_numbers())
        for value in ("0.360", "0.797", "0.744", "0.761"):
            assert value in html

    def test_renders_test_retest_correlations(self) -> None:
        """The reliability finding is the most important number in the
        project -- it must survive rendering."""
        html = render_results(_numbers())
        assert "0.92" in html
        assert "-0.19" in html

    def test_missing_result_file_degrades_to_dash(self) -> None:
        html = render_results(_numbers(leaky=None, perm_full=None))
        assert "&mdash;" in html

    def test_no_crash_when_everything_missing(self) -> None:
        empty = _numbers(
            lex=None,
            perm_sub=None,
            fusion_sub=None,
            perm_full=None,
            leaky=None,
            research_cremad=None,
            control=None,
            d1=None,
            hume=None,
            e3=dict.fromkeys(("lex_a", "lex_b", "perm_a", "perm_b", "res_a", "res_b")),
        )
        html = render_results(empty)
        assert "&mdash;" in html


class TestRenderWhatIsPossible:
    def test_contrasts_benchmark_against_real_voice(self) -> None:
        html = render_what_is_possible(_numbers())
        assert "0.80" in html  # benchmark UAR, 2dp
        assert "0.44" in html  # real-voice floor
        assert "0.33" in html  # chance


class TestStateOfTheArtIsFencedOff:
    """CLAUDE.md rule 6: never let other people's results read as ours."""

    def test_says_explicitly_these_are_not_our_numbers(self) -> None:
        html = render_state_of_the_art()
        assert "other researchers" in html.lower()

    def test_every_literature_claim_carries_a_link(self) -> None:
        html = render_state_of_the_art()
        for arxiv_id in ("2508.02448", "2510.25054", "2604.25776"):
            assert f"arxiv.org/abs/{arxiv_id}" in html


class TestWrapTables:
    """Tables are the only thing on the page wider than a phone; each needs
    its own scroll container or the whole page scrolls sideways."""

    def test_wraps_each_table_in_a_scroll_container(self) -> None:
        out = wrap_tables("<p>x</p><table><tr><td>a</td></tr></table>")
        assert out.count('<div class="tw">') == 1
        assert out.endswith("</table></div>")

    def test_open_and_close_stay_balanced(self) -> None:
        out = wrap_tables("<table>1</table><table>2</table>")
        assert out.count('<div class="tw">') == out.count("</table></div>") == 2

    def test_leaves_table_free_html_untouched(self) -> None:
        assert wrap_tables("<p>no tables</p>") == "<p>no tables</p>"


class TestBuildArtifactPage:
    """The published version must not carry its own document skeleton --
    the Artifact host supplies one -- and must not link to sibling files
    that exist only inside the repo."""

    def test_omits_document_skeleton(self) -> None:
        html = build_artifact_page()
        for tag in ("<!DOCTYPE", "<html", "<head>", "<body>"):
            assert tag not in html

    def test_carries_its_own_title_and_style(self) -> None:
        html = build_artifact_page()
        assert html.startswith("<title>")
        assert "<style>" in html

    def test_no_repo_relative_links_when_no_url_given(self) -> None:
        html = build_artifact_page()
        assert "../index.html" not in html
        assert 'href="index.html"' not in html

    def test_uses_absolute_url_when_given(self) -> None:
        html = build_artifact_page("https://example.com/repo")
        assert "https://example.com/repo" in html


class TestRenderImprovements:
    def test_names_the_broken_pipeline_entry_point(self) -> None:
        html = render_improvements()
        assert "ssa/eval/__main__.py" in html

    def test_names_the_missing_confidence_intervals(self) -> None:
        html = render_improvements()
        assert "onfidence interval" in html

    def test_covers_all_four_tiers(self) -> None:
        html = render_improvements()
        for tier in ("Tier 1", "Tier 2", "Tier 3", "Tier 4"):
            assert tier in html

    @pytest.mark.parametrize(
        "gap",
        ["never evaluated", "one speaker", "cross-corpus", "elderly"],
    )
    def test_names_each_known_evidence_gap(self, gap: str) -> None:
        assert gap in render_improvements().lower()
