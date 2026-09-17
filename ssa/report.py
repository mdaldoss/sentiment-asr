"""Builds report/index.html from results/*.json.

Renders only -- never recomputes a metric (CLAUDE.md: every number that
reaches the report comes from a JSON file on disk). Self-contained, no CDN
dependencies, so it opens offline. Palette from the dataviz skill's
validated default: categorical slots 1-3 (blue/orange/aqua) for the three
solutions, diverging blue<->red for sentiment polarity.

Usage:
    uv run python -m ssa.report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = REPO_ROOT / "report" / "index.html"

# Validated palette (dataviz skill, references/palette.md)
COLOR_SOLUTION = {
    "A": "#2a78d6",  # categorical slot 1, blue
    "B": "#eb6834",  # categorical slot 2, orange
    "C": "#1baf7a",  # categorical slot 3, aqua
}
COLOR_NEGATIVE = "#e34948"  # diverging red pole
COLOR_NEUTRAL = "#8a897f"  # neutral gray
COLOR_POSITIVE = "#2a78d6"  # diverging blue pole
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
UNMEASURED_NOTE = "argued from literature, not measured in this study"


def load_all_results() -> list[dict[str, Any]]:
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("skipping unreadable result %s: %s", path, exc)
    return results


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and x != x):  # NaN check without importing math
        return "&mdash;"
    return f"{x * 100:.1f}%"


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "&mdash;"
    return f"{x:.{digits}f}"


def _solution_short(name: str) -> str:
    """'A:lexical(small+twitter-roberta...)' -> 'A: Lexical'"""
    letter = name.split(":", 1)[0]
    label = {"A": "Lexical (ASR + text)", "B": "Acoustic", "C": "Fusion"}.get(letter, name)
    if "permissive" in name:
        label += " — permissive"
    elif "research" in name:
        label += " — research"
    return f"{letter}: {label}"


def _solution_letter(name: str) -> str:
    return name.split(":", 1)[0]


def render_main_table(results: list[dict[str, Any]]) -> str:
    """Solutions x datasets, UAR / macro-F1 / PSI."""
    if not results:
        return "<p class='pending'>No evaluation results yet.</p>"

    rows = []
    for r in sorted(results, key=lambda r: (r["dataset"], r["split_type"], r["solution"])):
        rows.append(
            f"<tr>"
            f"<td>{_solution_short(r['solution'])}</td>"
            f"<td>{r['dataset']} / {r['split_type']}</td>"
            f"<td>{r.get('n_clips', '?')}</td>"
            f"<td class='num'>{_fmt(r.get('uar'))}</td>"
            f"<td class='num'>{_fmt(r.get('macro_f1'))}</td>"
            f"<td class='num'>{_fmt(r.get('accuracy'))}</td>"
            f"<td class='num'>{_fmt(r.get('psi_contested'))}</td>"
            f"<td class='num'>{_fmt(r.get('psi_strict'))}</td>"
            f"<td class='num'>{_pct(r.get('abstention_rate'))}</td>"
            f"</tr>"
        )
    return f"""
    <table>
      <thead><tr>
        <th>Solution</th><th>Dataset / split</th><th>n</th>
        <th>UAR</th><th>Macro-F1</th><th>Accuracy</th>
        <th>PSI<sub>contested</sub></th><th>PSI<sub>strict</sub></th><th>Abstention</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    <p class="caption">UAR is the headline accuracy-family metric (class-imbalance robust). PSI is the
    headline prosody metric: 1.0 = follows tone, 0.0 = follows words, 0.5/0.33 = chance
    (contested/strict respectively). See DESIGN.md for the full metric definitions.</p>
    """


def render_leakage_table(disjoint: dict | None, leaky: dict | None) -> str:
    if disjoint is None or leaky is None:
        return "<p class='pending'>Leakage comparison pending.</p>"
    gap = leaky["uar"] - disjoint["uar"]
    return f"""
    <table>
      <thead><tr><th>Split</th><th>n test</th><th>Speakers overlapping train/test</th><th>UAR</th><th>Macro-F1</th></tr></thead>
      <tbody>
        <tr><td>Speaker-disjoint (honest)</td><td>{disjoint["n_clips"]}</td><td>0</td>
            <td class="num">{_fmt(disjoint["uar"])}</td><td class="num">{_fmt(disjoint["macro_f1"])}</td></tr>
        <tr><td>Random (leaky)</td><td>{leaky["n_clips"]}</td>
            <td>{leaky.get("n_speakers_overlapping_with_train", "?")}</td>
            <td class="num">{_fmt(leaky["uar"])}</td><td class="num">{_fmt(leaky["macro_f1"])}</td></tr>
      </tbody>
    </table>
    <p class="caption">Leakage inflation on this dataset+model: <strong>{gap:+.3f} UAR points</strong>.
    Smaller than the 10&ndash;40 point swings reported for cross-corpus generalisation in the
    literature (arXiv 2207.02104) &mdash; plausibly because WavLM's pretrained representation is
    already fairly speaker-invariant, and CREMA-D's 91 actors all read the same 12 sentences, so
    within-speaker variation may be less dominant here than in less controlled corpora. Reported as
    measured, not adjusted to match the expected range.</p>
    """


def render_d0_section(d0: dict | None) -> str:
    if d0 is None:
        return "<p class='pending'>D0 emotion-space probe not yet run.</p>"

    points = d0["per_clip"]
    svg_w, svg_h, pad = 480, 320, 40
    valences = [p["valence"] for p in points]
    arousals = [p["arousal"] for p in points]
    vmin, vmax = min(valences), max(valences)
    amin, amax = min(arousals), max(arousals)
    vspan = max(vmax - vmin, 1e-6)
    aspan = max(amax - amin, 1e-6)

    def x(v: float) -> float:
        return pad + (v - vmin) / vspan * (svg_w - 2 * pad)

    def y(a: float) -> float:
        return svg_h - pad - (a - amin) / aspan * (svg_h - 2 * pad)

    cluster_colors = [COLOR_SOLUTION["A"], COLOR_SOLUTION["B"], COLOR_SOLUTION["C"], COLOR_NEGATIVE]
    dots = []
    for p in points:
        color = cluster_colors[p["cluster"] % len(cluster_colors)]
        dots.append(
            f'<circle cx="{x(p["valence"]):.1f}" cy="{y(p["arousal"]):.1f}" r="4" '
            f'fill="{color}" fill-opacity="0.75"><title>{p["tag"]} ({p["carrier"]}) '
            f"v={p['valence']:.3f} a={p['arousal']:.3f}</title></circle>"
        )

    return f"""
    <svg viewBox="0 0 {svg_w} {svg_h}" class="scatter" role="img"
         aria-label="Valence-arousal scatter of {d0["n_clips"]} Cartesia clips across {d0["n_tags"]} tags">
      <line x1="{pad}" y1="{svg_h - pad}" x2="{svg_w - pad}" y2="{svg_h - pad}" class="axis"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{svg_h - pad}" class="axis"/>
      <text x="{svg_w / 2}" y="{svg_h - 8}" class="axis-label" text-anchor="middle">Valence &rarr;</text>
      <text x="12" y="{svg_h / 2}" class="axis-label" text-anchor="middle"
            transform="rotate(-90 12 {svg_h / 2})">Arousal &rarr;</text>
      {"".join(dots)}
    </svg>
    <p class="caption">
      {d0["n_clips"]} clips, {d0["n_tags"]} tags, k={d0["best_k"]} clusters
      (silhouette={_fmt(d0["silhouette_score"])}), valence span={_fmt(d0.get("valence_span"))}.
      Hover a point for its tag.
    </p>
    <p class="finding"><strong>Finding:</strong> {d0["interpretation"]}</p>
    """


def render_voicehealth_demo() -> str:
    return f"""
    <p>Extraction implemented and verified against synthetic signals (see ssa/voicehealth.py,
    ssa/paralinguistic.py): CPPS, HNR, jitter, shimmer, and an AVQI composite via
    praat-parselmouth; whisper/breathiness detection via openSMILE eGeMAPS.</p>
    <p class="unmeasured">The per-speaker longitudinal baseline gate (separating presbyphonia &mdash;
    a trait drifting over years &mdash; from acute illness &mdash; a state deviating from a personal
    baseline over days) is a <strong>design proposal, {UNMEASURED_NOTE}</strong>: no longitudinal
    single-speaker data exists to validate it against. See DESIGN.md's roadmap.</p>
    """


def render_status(results: list[dict[str, Any]]) -> str:
    datasets = {(r["dataset"], r["split_type"]) for r in results}
    have_e2 = (
        any("synthetic" in str(d) for d in datasets)
        or (REPO_ROOT / "data/synthetic/manifest.csv").exists()
    )
    have_e3 = (REPO_ROOT / "data/recorded").exists() and any(
        (REPO_ROOT / "data/recorded").glob("*.wav")
    )
    items = [
        ("E1 (CREMA-D, public benchmark)", True),
        ("D0 (Cartesia emotion-space probe)", (RESULTS_DIR / "d0_emotion_space.json").exists()),
        ("E2 (synthetic incongruence set)", have_e2),
        ("E3 (human recordings)", have_e3),
    ]
    li_parts = []
    for name, done in items:
        css_class = "done" if done else "pending"
        marker = "&check;" if done else "&hellip;"
        li_parts.append(f"<li class='{css_class}'>{marker} {name}</li>")
    lis = "".join(li_parts)
    return f"<ul class='status-list'>{lis}</ul>"


def build_report() -> str:
    results = load_all_results()
    # "confusion_matrix" only exists on full EvalResult dumps -- distinguishes them
    # from smaller hand-written result files (e.g. leakage_comparison_*.json).
    eval_results = [r for r in results if "confusion_matrix" in r]
    d0 = load_json_if_exists(RESULTS_DIR / "d0_emotion_space.json")
    leaky = load_json_if_exists(RESULTS_DIR / "leakage_comparison_permissive.json")
    disjoint_permissive = next(
        (
            r
            for r in eval_results
            if _solution_letter(r["solution"]) == "B"
            and "permissive" in r["solution"]
            and r["split_type"] == "speaker_disjoint_test"
        ),
        None,
    )

    css = """
    :root {
      color-scheme: light;
      --surface: #fcfcfb; --surface-2: #f5f4f1; --text: #0b0b0b; --text-2: #52514e;
      --border: #e2e0da; --good: #0ca30c; --warn: #fab219;
    }
    @media (prefers-color-scheme: dark) {
      :root:not([data-theme="light"]) {
        color-scheme: dark;
        --surface: #1a1a19; --surface-2: #242422; --text: #ffffff; --text-2: #c3c2b7;
        --border: #3a3936; --good: #0ca30c; --warn: #fab219;
      }
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --surface: #1a1a19; --surface-2: #242422; --text: #ffffff; --text-2: #c3c2b7;
      --border: #3a3936; --good: #0ca30c; --warn: #fab219;
    }
    body { background: var(--surface); color: var(--text); font-family: -apple-system, BlinkMacSystemFont,
           "Segoe UI", sans-serif; max-width: 980px; margin: 0 auto; padding: 24px 16px 80px;
           line-height: 1.5; }
    h1 { font-size: 1.6rem; } h2 { font-size: 1.25rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border);
         padding-bottom: 6px; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.92rem; }
    th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    thead th { color: var(--text-2); font-weight: 600; font-size: 0.82rem; text-transform: uppercase; }
    .caption { color: var(--text-2); font-size: 0.86rem; }
    .finding { background: var(--surface-2); border-left: 3px solid #eda100; padding: 10px 14px; border-radius: 4px; }
    .unmeasured { background: var(--surface-2); border-left: 3px solid var(--text-2); padding: 10px 14px;
                  border-radius: 4px; font-style: italic; }
    .pending { color: var(--text-2); font-style: italic; }
    .scatter { width: 100%; max-width: 480px; display: block; }
    .axis { stroke: var(--border); stroke-width: 1; }
    .axis-label { fill: var(--text-2); font-size: 11px; }
    .status-list { list-style: none; padding: 0; }
    .status-list li { padding: 4px 0; }
    .status-list li.done { color: var(--good); }
    .status-list li.pending { color: var(--warn); }
    header { border-bottom: 2px solid var(--border); padding-bottom: 16px; margin-bottom: 8px; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Speech Sentiment Analyzer — Results</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Speech Sentiment Analyzer — Results Dashboard</h1>
  <p class="caption">Generated from <code>results/*.json</code> by <code>ssa/report.py</code> &mdash;
  every number here traces to a committed result file. See <code>DESIGN.md</code> for the
  full write-up.</p>
</header>

<h2>Status</h2>
{render_status(eval_results)}

<h2>Main comparison: solutions &times; datasets</h2>
{render_main_table(eval_results)}

<h2>The leakage exposé (Solution B, permissive backend)</h2>
{render_leakage_table(disjoint_permissive, leaky)}

<h2>D0: Cartesia emotion-space probe</h2>
{render_d0_section(d0)}

<h2>Voice-health demo</h2>
{render_voicehealth_demo()}

<footer class="caption">
  <p>🤖 Report generator: <code>ssa/report.py</code>. Regenerate with <code>make report</code>.</p>
</footer>
</body>
</html>"""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = build_report()
    OUT_PATH.write_text(html)
    logger.info("wrote %s (%d bytes)", OUT_PATH, len(html))


if __name__ == "__main__":
    main()
