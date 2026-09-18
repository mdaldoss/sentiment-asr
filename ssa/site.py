"""Builds the two root-level navigation pages: /index.html (links to
everything) and /architecture.html (the pipeline diagram + module roles).

Unlike ssa/report.py, these pages carry no metrics -- they're navigation and
explanation, not results -- so there is nothing here for CLAUDE.md's "never
report a metric without its condition" rule to apply to. What IS shared
with report.py is render_models_section(): the "which model actually
extracts prosody/emotion" explanation is written once and reused verbatim
across index/architecture/report/listening_sorted, rather than drifting
into four slightly different retellings.

Usage:
    uv run python -m ssa.site
"""

from __future__ import annotations

import logging
from pathlib import Path

from ssa.report import render_models_section

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_OUT = REPO_ROOT / "index.html"
ARCHITECTURE_OUT = REPO_ROOT / "architecture.html"

# Same tokens as ssa/report.py's CSS, kept independent (these pages don't
# import report.py's build_report(), just its one shared content function)
# so the two modules can evolve their layouts without coupling.
_BASE_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f5f4f1; --text: #0b0b0b; --text-2: #52514e;
  --border: #e2e0da; --good: #0ca30c; --warn: #fab219;
  --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a; --red: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242422; --text: #ffffff; --text-2: #c3c2b7;
    --border: #3a3936;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242422; --text: #ffffff; --text-2: #c3c2b7;
  --border: #3a3936;
}
body { background: var(--surface); color: var(--text); font-family: -apple-system, BlinkMacSystemFont,
       "Segoe UI", sans-serif; max-width: 980px; margin: 0 auto; padding: 24px 16px 80px;
       line-height: 1.5; }
h1 { font-size: 1.7rem; margin-bottom: 4px; }
h2 { font-size: 1.25rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
p.tagline { color: var(--text-2); margin-top: 0; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.92rem; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
thead th { color: var(--text-2); font-weight: 600; font-size: 0.82rem; text-transform: uppercase; }
.caption { color: var(--text-2); font-size: 0.86rem; }
.unmeasured { background: var(--surface-2); border-left: 3px solid var(--text-2); padding: 10px 14px;
              border-radius: 4px; font-style: italic; }
header { border-bottom: 2px solid var(--border); padding-bottom: 16px; margin-bottom: 8px; }
footer { margin-top: 3rem; border-top: 1px solid var(--border); padding-top: 12px; }
a { color: var(--blue); }
code { background: var(--surface-2); padding: 1px 5px; border-radius: 3px; font-size: 0.92em; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; margin: 16px 0; }
.card { display: block; border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px;
        text-decoration: none; color: var(--text); background: var(--surface-2); }
.card:hover { border-color: var(--blue); }
.card h3 { margin: 0 0 6px; font-size: 1.02rem; color: var(--blue); }
.card p { margin: 0; font-size: 0.86rem; color: var(--text-2); }
.pipeline { width: 100%; max-width: 900px; display: block; margin: 16px 0; }
.pbox { fill: var(--surface-2); stroke: var(--border); stroke-width: 1.5; rx: 6; }
.pbox-a { stroke: var(--blue); } .pbox-b { stroke: var(--orange); } .pbox-c { stroke: var(--aqua); }
.plabel { fill: var(--text); font-size: 13px; }
.psmall { fill: var(--text-2); font-size: 11px; }
.parrow { stroke: var(--text-2); stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
"""


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{_BASE_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def render_pipeline_diagram() -> str:
    """Audio -> {A, B, C} -> Prediction, plus the VAD plane -> sentiment
    read-out. Boxes/arrows via inline SVG, same convention as ssa.report's
    scatter charts (no external diagramming library)."""
    return """
    <svg viewBox="0 0 900 430" class="pipeline" role="img"
         aria-label="Pipeline: audio clip into solutions A, B, C, producing a sentiment prediction">
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--text-2)"/>
        </marker>
      </defs>

      <rect class="pbox" x="20" y="190" width="120" height="50" rx="6"/>
      <text class="plabel" x="80" y="211" text-anchor="middle">Audio clip</text>
      <text class="psmall" x="80" y="227" text-anchor="middle">16kHz mono</text>

      <path class="parrow" d="M140,215 h40"/>

      <!-- Solution A -->
      <rect class="pbox pbox-a" x="200" y="40" width="190" height="70" rx="6"/>
      <text class="plabel" x="295" y="65" text-anchor="middle" font-weight="600">A: Lexical</text>
      <text class="psmall" x="295" y="82" text-anchor="middle">faster-whisper (ASR)</text>
      <text class="psmall" x="295" y="97" text-anchor="middle">&rarr; text sentiment classifier</text>

      <!-- Solution B -->
      <rect class="pbox pbox-b" x="200" y="180" width="190" height="90" rx="6"/>
      <text class="plabel" x="295" y="202" text-anchor="middle" font-weight="600">B: Acoustic</text>
      <text class="psmall" x="295" y="219" text-anchor="middle">permissive: WavLM + our probe</text>
      <text class="psmall" x="295" y="234" text-anchor="middle">research: audeering wav2vec2 &rarr; VAD</text>
      <text class="psmall" x="295" y="252" text-anchor="middle" font-style="italic">frozen encoder, no fine-tuning</text>

      <!-- Solution C label near fusion box -->
      <rect class="pbox pbox-c" x="470" y="150" width="200" height="90" rx="6"/>
      <text class="plabel" x="570" y="175" text-anchor="middle" font-weight="600">C: Fusion</text>
      <text class="psmall" x="570" y="192" text-anchor="middle">calibrated late fusion (log-linear)</text>
      <text class="psmall" x="570" y="207" text-anchor="middle">+ abstention</text>
      <text class="psmall" x="570" y="225" text-anchor="middle" font-style="italic">recommended default</text>

      <path class="parrow" d="M140,220 C170,220 170,75 200,75"/>
      <path class="parrow" d="M140,220 h60"/>
      <path class="parrow" d="M390,75 C440,75 440,175 470,175"/>
      <path class="parrow" d="M390,225 C430,225 430,200 470,200"/>

      <!-- VAD plane -->
      <rect class="pbox" x="470" y="20" width="200" height="90" rx="6"/>
      <text class="plabel" x="570" y="42" text-anchor="middle" font-weight="600">Shared: VAD plane</text>
      <text class="psmall" x="570" y="59" text-anchor="middle">valence-arousal-dominance</text>
      <text class="psmall" x="570" y="74" text-anchor="middle">sentiment = threshold(valence)</text>
      <text class="psmall" x="570" y="89" text-anchor="middle" font-style="italic">distress quadrants: roadmap</text>
      <path class="parrow" d="M390,75 C420,75 420,65 470,65"/>

      <!-- Output -->
      <rect class="pbox" x="740" y="185" width="140" height="60" rx="6"/>
      <text class="plabel" x="810" y="210" text-anchor="middle" font-weight="600">Prediction</text>
      <text class="psmall" x="810" y="227" text-anchor="middle">sentiment + confidence</text>
      <path class="parrow" d="M670,195 h70"/>

      <!-- PSI annotation -->
      <text class="psmall" x="450" y="320" text-anchor="middle" font-style="italic">
        Prosody Sensitivity Index (PSI): on incongruent clips, does the prediction follow
      </text>
      <text class="psmall" x="450" y="335" text-anchor="middle" font-style="italic">
        the tone (B, C) or the words (A)? The whole evaluation design serves this question.
      </text>
    </svg>
    """


def render_index_body() -> str:
    cards = [
        (
            "Architecture",
            "architecture.html",
            "Pipeline diagram, the lexical↔acoustic axis, and which model actually extracts prosody/emotion (with papers).",
        ),
        (
            "Results dashboard",
            "report/index.html",
            "The full write-up: main comparison, D0/D1/E3 findings, the Hume falsification test, and what didn't work.",
        ),
        (
            "Listen: D1 A/B pairs",
            "report/d1_listening.html",
            "Matched Cartesia clips, same words, different emotion tag — judge for yourself.",
        ),
        (
            "Listen: sorted (D1+E3)",
            "report/listening_sorted.html",
            "All 99 Cartesia + human clips ranked by detection score.",
        ),
        (
            "Design write-up",
            "DESIGN.md",
            "Approach, trade-offs, measured results, known limitations, what's next.",
        ),
        (
            "Module contracts",
            "docs/ARCHITECTURE.md",
            "Module-by-module contracts and rationale for anyone extending the code.",
        ),
        (
            "Project rules",
            "CLAUDE.md",
            "The hard rules and pinned facts every change in this repo must respect.",
        ),
        (
            "README / quick start",
            "README.md",
            "make setup, make test, make demo — clone-to-running in a few commands.",
        ),
    ]
    card_html = "\n".join(
        f'<a class="card" href="{href}"><h3>{title}</h3><p>{desc}</p></a>'
        for title, href, desc in cards
    )
    return f"""
<header>
  <h1>Speech Sentiment Analyzer</h1>
  <p class="tagline">Sentiment (positive / neutral / negative) from raw speech audio &mdash;
  not from a transcript. Built to measure whether a model hears the <em>tone</em>, or is
  just reading the <em>words</em>.</p>
</header>

<h2>Start here</h2>
<div class="cards">
{card_html}
</div>

<footer class="caption">
  <p>Generated by <code>ssa/site.py</code>. Regenerate with <code>make site</code>. This
  page and <code>architecture.html</code> carry no metrics of their own &mdash; every
  number lives in <code>report/index.html</code>, generated from <code>results/*.json</code>
  by <code>ssa/report.py</code>, per CLAUDE.md's rule that a number is only ever rendered,
  never invented.</p>
</footer>
"""


def render_architecture_body() -> str:
    return f"""
<header>
  <h1>Architecture</h1>
  <p class="tagline"><a href="index.html">&larr; back to index</a></p>
</header>

<h2>The one idea everything serves</h2>
<p>An audio sentiment model can cheat by reading the <strong>words</strong> instead of
hearing the <strong>tone</strong> &mdash; published work confirms this is the default
failure mode for audio models (arXiv 2510.10444, arXiv 2510.25054). Ami (the target
product, a voice-first companion with no screen) has no signal but tone. Every design
choice below exists to measure whether a model actually hears it.</p>

<h2>Pipeline</h2>
{render_pipeline_diagram()}
<p class="caption">Three solutions on the lexical&harr;acoustic axis, one interface
(<code>ssa.types.Solution</code>), so the evaluation harness treats them identically.
Late fusion (not joint) deliberately sacrifices some accuracy to keep PSI computable
per branch &mdash; see DESIGN.md's Key trade-offs.</p>

<table>
  <thead><tr><th></th><th>Solution</th><th>Reads</th><th>License</th></tr></thead>
  <tbody>
    <tr><td><strong>A</strong></td><td>Lexical &mdash; ASR (faster-whisper) &rarr; text sentiment</td><td>the words</td><td>MIT</td></tr>
    <tr><td><strong>B</strong></td><td>Acoustic &mdash; frozen encoder + trained probe</td><td>the tone</td><td>permissive (WavLM) or research (audeering, CC-BY-NC-SA-4.0)</td></tr>
    <tr><td><strong>C</strong></td><td>Fusion &mdash; calibrated late fusion + abstention</td><td>both</td><td>recommended default</td></tr>
  </tbody>
</table>

<h2>Which model actually extracts prosody/emotion</h2>
{render_models_section()}

<h2>Evaluation datasets</h2>
<table>
  <thead><tr><th>Dataset</th><th>What it is</th><th>Role</th></tr></thead>
  <tbody>
    <tr><td>E1 &mdash; CREMA-D</td><td>7,442 clips, 91 actors, public (ODbL)</td>
        <td>Primary benchmark. Text always neutral &rarr; PSI here is near-vacuous by construction.</td></tr>
    <tr><td>D0 / D1</td><td>Cartesia TTS probes (unsupervised clustering; factorial grid)</td>
        <td>Is Cartesia's emotion rendering measurable at all? (Answer: no, on this content &mdash; see the report.)</td></tr>
    <tr><td>E2</td><td>76 Cartesia incongruence clips</td>
        <td>Superseded by D1's finding; kept as the historical exhibit.</td></tr>
    <tr><td>E3</td><td>Two independent human-recorded takes, same speaker</td>
        <td>The validity anchor &mdash; and the control that rules out the measuring instruments
        as D1's explanation.</td></tr>
    <tr><td>Hume probe</td><td>10 clips, same design as D1, different vendor</td>
        <td>Falsification test: is Cartesia's failure vendor-specific? (Yes.)</td></tr>
  </tbody>
</table>
<p class="caption">Full numbers, methodology, and honest caveats for every one of these are
in <a href="report/index.html">the results dashboard</a> and
<a href="DESIGN.md">DESIGN.md</a>.</p>

<footer class="caption">
  <p>See <a href="docs/ARCHITECTURE.md">docs/ARCHITECTURE.md</a> for full module-by-module
  contracts (signatures, invariants, rationale). Generated by <code>ssa/site.py</code>.</p>
</footer>
"""


def build_index_page() -> str:
    return _page("Speech Sentiment Analyzer", render_index_body())


def build_architecture_page() -> str:
    return _page("Architecture — Speech Sentiment Analyzer", render_architecture_body())


def main() -> None:
    INDEX_OUT.write_text(build_index_page())
    ARCHITECTURE_OUT.write_text(build_architecture_page())
    logger.info("wrote %s and %s", INDEX_OUT, ARCHITECTURE_OUT)


if __name__ == "__main__":
    main()
