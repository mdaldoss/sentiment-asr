"""Tests for the root-level navigation pages (index.html, architecture.html).
Pure rendering, no results data needed -- these pages carry no metrics, only
navigation and explanation. Same style as tests/test_report.py."""

from __future__ import annotations

import re

from ssa.site import (
    build_architecture_page,
    build_index_page,
    render_architecture_body,
    render_index_body,
    render_pipeline_diagram,
)


def _local_hrefs(html: str) -> list[str]:
    return [h for h in re.findall(r'href="([^"]+)"', html) if not h.startswith("http")]


class TestRenderIndexBody:
    def test_links_to_every_expected_destination(self) -> None:
        html = render_index_body()
        for target in (
            "architecture.html",
            "report/index.html",
            "report/d1_listening.html",
            "report/listening_sorted.html",
            "DESIGN.md",
            "docs/ARCHITECTURE.md",
            "CLAUDE.md",
            "README.md",
        ):
            assert target in html

    def test_no_dangling_markup(self) -> None:
        html = render_index_body()
        assert html.count("<a ") == html.count("</a>")
        assert html.count('<div class="cards">') == 1


class TestRenderPipelineDiagram:
    def test_mentions_all_three_solutions(self) -> None:
        svg = render_pipeline_diagram()
        assert "A: Lexical" in svg
        assert "B: Acoustic" in svg
        assert "C: Fusion" in svg

    def test_svg_tag_balanced(self) -> None:
        svg = render_pipeline_diagram()
        assert svg.count("<svg") == svg.count("</svg>")

    def test_mentions_psi(self) -> None:
        assert "Prosody Sensitivity Index" in render_pipeline_diagram()


class TestRenderArchitectureBody:
    def test_includes_pipeline_diagram(self) -> None:
        html = render_architecture_body()
        assert "<svg" in html

    def test_includes_models_section(self) -> None:
        html = render_architecture_body()
        assert "arXiv:2110.13900" in html
        assert "arXiv:2203.07378" in html

    def test_links_back_to_index(self) -> None:
        assert "index.html" in render_architecture_body()

    def test_lists_all_evaluation_datasets(self) -> None:
        html = render_architecture_body()
        for dataset in ("CREMA-D", "D0", "D1", "E2", "E3", "Hume"):
            assert dataset in html


class TestFullPages:
    def test_index_page_is_well_formed_and_titled(self) -> None:
        html = build_index_page()
        assert "<!DOCTYPE html>" in html
        assert "<title>Speech Sentiment Analyzer</title>" in html
        assert html.count("<html") == html.count("</html>")

    def test_architecture_page_is_well_formed(self) -> None:
        html = build_architecture_page()
        assert "<!DOCTYPE html>" in html
        assert html.count("<body>") == html.count("</body>")

    def test_index_hrefs_do_not_include_broken_placeholders(self) -> None:
        html = build_index_page()
        hrefs = _local_hrefs(html)
        assert len(hrefs) >= 8
        assert all(h and "None" not in h for h in hrefs)
