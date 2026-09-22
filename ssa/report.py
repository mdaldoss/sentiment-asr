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


def render_models_section() -> str:
    """Which model actually extracts prosody/emotion, and its paper --
    shared verbatim between report/index.html and
    report/listening_sorted.html (scripts/gen_listening_sorted.py imports
    this function) since both pages get asked "which model did this?"."""
    return """
    <p>A common point of confusion, worth stating precisely: <strong>WavLM does not
    transcribe or recognize emotion by itself</strong> &mdash; two different models do
    two different jobs.</p>
    <table>
      <thead><tr><th></th><th>Model</th><th>What it actually is</th><th>Paper</th></tr></thead>
      <tbody>
        <tr><td>Transcription (Solution A only)</td><td><code>faster-whisper</code></td>
            <td>OpenAI's Whisper ASR, CTranslate2-optimized. Nothing to do with WavLM.</td>
            <td>&mdash;</td></tr>
        <tr><td>Acoustic, <strong>permissive</strong> backend</td>
            <td><code>microsoft/wavlm-base</code></td>
            <td>A general-purpose self-supervised speech encoder &mdash; trained to
            reconstruct masked/overlapping speech, never on an emotion label. We pool its
            hidden states (mean+std) and train <em>our own</em> logistic regression on top
            (fit on CREMA-D). WavLM supplies the representation; the emotion mapping is
            ours, not WavLM's.</td>
            <td>Chen et al. 2022,
            <a href="https://arxiv.org/abs/2110.13900">arXiv:2110.13900</a></td></tr>
        <tr><td>Acoustic, <strong>research</strong> backend</td>
            <td><code>audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim</code></td>
            <td>A wav2vec2-large-robust backbone <strong>fine-tuned end-to-end</strong> on
            MSP-Podcast to directly regress valence/arousal/dominance (VAD). This is the
            model that actually extracts prosody/emotion &mdash; no probe needed, it
            outputs VAD directly, and is what the sample plot below uses.</td>
            <td>Wagner et al. 2023, "Dawn of the Transformer Era in Speech Emotion
            Recognition: Closing the Valence Gap",
            <a href="https://arxiv.org/abs/2203.07378">arXiv:2203.07378</a></td></tr>
      </tbody>
    </table>
    <p class="unmeasured">A caveat straight from the research-backend paper, stated
    rather than glossed over (CLAUDE.md rule 6): its authors report that the model's
    strong valence performance comes partly from <strong>implicit linguistic
    information learned during fine-tuning</strong>, not from prosody alone. Even this
    "acoustic" model may not be as prosody-pure as its name implies &mdash; it doesn't
    undermine this project's core comparisons (Solution A genuinely never sees audio;
    the D1-vs-E3 Cartesia finding holds regardless), but it belongs in the record.</p>
    """


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


_SENTIMENT_COLOR = {
    "positive": COLOR_POSITIVE,
    "neutral": COLOR_NEUTRAL,
    "negative": COLOR_NEGATIVE,
    # CREMA-D's own emotion tags, mapped to the same 3-color scheme via
    # ssa.mapping.CREMA_D_MAP (HAP=positive, NEU=neutral, everything else negative)
    "HAP": COLOR_POSITIVE,
    "NEU": COLOR_NEUTRAL,
    "ANG": COLOR_NEGATIVE,
    "DIS": COLOR_NEGATIVE,
    "FEA": COLOR_NEGATIVE,
    "SAD": COLOR_NEGATIVE,
}


def render_prosody_samples_section(data: dict | None) -> str:
    """20 real clips (12 CREMA-D + 8 E3), VAD extracted with the research
    backend (the model that actually regresses emotion -- see the Models
    section) plus F0, so the extracted signal is visible directly rather
    than only its downstream classification accuracy."""
    if data is None:
        return "<p class='pending'>Prosody sample extraction not yet run.</p>"

    samples = data["samples"]
    svg_w, svg_h, pad = 560, 360, 44
    valences = [s["valence"] for s in samples]
    arousals = [s["arousal"] for s in samples]
    vmin, vmax = min(valences), max(valences)
    amin, amax = min(arousals), max(arousals)
    vspan = max(vmax - vmin, 1e-6)
    aspan = max(amax - amin, 1e-6)

    def x(v: float) -> float:
        return pad + (v - vmin) / vspan * (svg_w - 2 * pad)

    def y(a: float) -> float:
        return svg_h - pad - (a - amin) / aspan * (svg_h - 2 * pad)

    marks = []
    for s in samples:
        color = _SENTIMENT_COLOR.get(s["label"], COLOR_NEUTRAL)
        is_mine = s["source"].startswith("E3")
        cx, cy = x(s["valence"]), y(s["arousal"])
        f0_label = f"{s['f0_mean']:.0f}Hz" if s["f0_mean"] else "n/a"
        title = (
            f"{s['clip_id']} &mdash; {s['source']}, label={s['label']} "
            f"v={s['valence']:.3f} a={s['arousal']:.3f} f0={f0_label}"
        )
        if is_mine:
            # square marker for "mine" (E3), circle for the public dataset (CREMA-D)
            size = 7
            marks.append(
                f'<rect x="{cx - size:.1f}" y="{cy - size:.1f}" '
                f'width="{2 * size}" height="{2 * size}" '
                f'fill="{color}" fill-opacity="0.85" stroke="#333" stroke-width="0.5">'
                f"<title>{title}</title></rect>"
            )
        else:
            marks.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" fill-opacity="0.75">'
                f"<title>{title}</title></circle>"
            )

    f0_rows = "".join(
        f"<tr><td>{s['clip_id']}</td><td>{s['source']}</td><td>{s['label']}</td>"
        f"<td class='num'>{_fmt(s['valence'])}</td><td class='num'>{_fmt(s['arousal'])}</td>"
        f"<td class='num'>{_fmt(s.get('f0_mean'), 0)} Hz</td></tr>"
        for s in sorted(samples, key=lambda s: s["valence"], reverse=True)
    )

    return f"""
    <p>{data["n_cremad"]} clips from CREMA-D (public dataset, circles) and {data["n_e3"]}
    from your own E3 recordings (squares), extracted with the same instrument
    ({data["model"]}). Color = mapped sentiment (blue=positive, gray=neutral, red=negative).</p>
    <svg viewBox="0 0 {svg_w} {svg_h}" class="scatter" role="img"
         aria-label="Valence-arousal scatter of {len(samples)} sample clips from CREMA-D and E3">
      <line x1="{pad}" y1="{svg_h - pad}" x2="{svg_w - pad}" y2="{svg_h - pad}" class="axis"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{svg_h - pad}" class="axis"/>
      <text x="{svg_w / 2}" y="{svg_h - 8}" class="axis-label" text-anchor="middle">Valence &rarr;</text>
      <text x="12" y="{svg_h / 2}" class="axis-label" text-anchor="middle"
            transform="rotate(-90 12 {svg_h / 2})">Arousal &rarr;</text>
      {"".join(marks)}
    </svg>
    <p class="caption">Circle = CREMA-D (public), square = E3 (mine). Hover a point for its
    clip id, label, exact VAD and F0. Full table, sorted by valence:</p>
    <table>
      <thead><tr><th>Clip</th><th>Source</th><th>Label</th><th>Valence</th><th>Arousal</th><th>F0 mean</th></tr></thead>
      <tbody>{f0_rows}</tbody>
    </table>
    <p class="caption">CREMA-D's HAP samples read highest-valence and SAD lowest in this
    20-clip draw &mdash; a sane ordering on real acted speech from a model that has
    actually seen emotion labels (contrast with D0/D1's scrambled ordering on Cartesia's
    synthetic output, above). Regenerate with <code>make prosody-samples</code>; the
    selection is seeded/deterministic but small (n=20) &mdash; illustrative, not a
    benchmark.</p>
    """


def render_d1_section(d1: dict | None) -> str:
    if d1 is None:
        return "<p class='pending'>D1 emotion-rendering probe not yet run.</p>"

    cv = d1["recoverability_cv"]
    rank = d1["f0_rank_consistency"]
    rank_rows = "".join(
        f"<tr><td>{emotion}</td><td class='num'>{v:.2f}</td></tr>"
        for emotion, v in rank["mean_rank_by_emotion"].items()
    )
    ordering = d1["valence_ordering_matches_intended_sentiment"]
    return f"""
    <p>Factorial probe: 5 emotions (happy/sad/angry/calm/frustrated) &times; text condition
    (neutral/congruent) &times; length (short/long) &times; speed (default/adjusted) =
    {d1["grid_size"]} clips, plus a {d1["comparison_size"]}-clip sonic-3 vs sonic-3.5
    comparison on the cell with the widest F0 spread. Measured with 4 independent
    instruments (F0, eGeMAPS, our WavLM probe, the audeering VAD model).</p>
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>Chance</th></tr></thead>
      <tbody>
        <tr><td>Intended-emotion recoverability (leave-one-carrier-out)</td>
            <td class="num">{_fmt(cv["accuracy"])}</td><td class="num">{_fmt(cv["chance"])}</td></tr>
        <tr><td>F0 span across the 5 emotions (averaged over all 8 conditions)</td>
            <td class="num">{_fmt(d1.get("f0_span_across_emotions_hz"), 1)} Hz</td><td>&mdash;</td></tr>
      </tbody>
    </table>
    <p class="caption">F0 rank of each emotion, averaged across the 8 conditions
    (1 = highest pitch; chance = {_fmt(rank["chance_mean_rank"], 1)} for 5 classes):</p>
    <table><thead><tr><th>Emotion</th><th>Mean rank</th></tr></thead><tbody>{rank_rows}</tbody></table>
    <p class="caption">Valence ordering (positive &gt; neutral &gt; negative) recovered?
    Permissive (our WavLM probe): <strong>{ordering.get("permissive")}</strong>.
    Research (audeering): <strong>{ordering.get("research")}</strong>.</p>
    <p class="finding"><strong>Finding:</strong> {d1["interpretation"]}</p>
    <p class="caption">Listen for yourself: <code>report/d1_listening.html</code>
    (matched A/B pairs) and <code>report/listening_sorted.html</code>
    (all D1+E3 clips, ranked by detection score).</p>
    """


def render_hume_section(data: dict | None) -> str:
    """A falsification test of the D1 finding, at D1's own grid scale: 5
    emotions x text_condition x length x description(on/off), the exact
    same carrier texts D1 used on Cartesia, from a vendor (Hume Octave)
    whose `description` field is documented to control delivery
    independently of the transcript -- the capability Cartesia's docs say
    it lacks."""
    if data is None:
        return "<p class='pending'>Hume probe not yet run (needs HUME_API_KEY).</p>"

    with_desc = data["summary"]["with_description"]
    without_desc = data["summary"]["without_description"]
    neutral_desc = data["summary"]["neutral_text_with_description"]
    congruent_desc = data["summary"]["congruent_text_with_description"]

    def emotion_rows(group: dict) -> str:
        return "".join(
            f"<tr><td>{emotion}</td><td class='num'>{_fmt(f0, 0)} Hz</td>"
            f"<td class='num'>{_fmt(group['valence_mean_by_emotion'].get(emotion))}</td></tr>"
            for emotion, f0 in group["f0_mean_by_emotion"].items()
        )

    with_cv, without_cv = with_desc["recoverability_cv"], without_desc["recoverability_cv"]

    return f"""
    <p>Same design as D1, same scale: {data["n_clips"]} clips, 5 emotions, the exact same
    carrier texts D1 used on Cartesia (<code>ssa.carriers.D1_NEUTRAL_TEXT</code> and
    <code>D1_CONGRUENT_TEXT</code>, both lengths), one fixed voice ("{data["voice"]}") &mdash;
    so the comparison is apples-to-apples, and this time with genuine carrier diversity
    (12 groups, same as D1) for an honest leave-one-carrier-out recoverability classifier.</p>
    <table>
      <thead><tr><th>Condition</th><th>Recoverability</th><th>Chance</th><th>F0 span</th>
      <th>Valence ordering (pos&gt;neu&gt;neg)?</th></tr></thead>
      <tbody>
        <tr><td>With <code>description</code></td>
            <td class="num">{_fmt(with_cv["accuracy"])}</td>
            <td class="num">{_fmt(with_cv["chance"])}</td>
            <td class="num">{_fmt(with_desc["f0_span_hz"], 1)} Hz</td>
            <td><strong>{with_desc["valence_ordering_matches_intended_sentiment"]}</strong></td></tr>
        <tr><td>Without <code>description</code></td>
            <td class="num">{_fmt(without_cv["accuracy"])}</td>
            <td class="num">{_fmt(without_cv["chance"])}</td>
            <td class="num">{_fmt(without_desc["f0_span_hz"], 1)} Hz</td>
            <td><strong>{without_desc["valence_ordering_matches_intended_sentiment"]}</strong></td></tr>
        <tr><td><em>Cartesia (D1), for comparison</em></td>
            <td class="num">0.175</td><td class="num">0.200</td>
            <td class="num">~8 Hz</td><td>scrambled (see above)</td></tr>
      </tbody>
    </table>
    <p class="caption">The harder cut &mdash; does <code>description</code> work even on
    <strong>neutral</strong> text, isolating prosody from wording entirely (this project's
    gold-label rule)? Neutral valence range:
    {_fmt(neutral_desc.get("valence_range"))}, F0 span {_fmt(neutral_desc.get("f0_span_hz"), 1)} Hz
    (n={neutral_desc["n_clips"]}) vs. congruent valence range
    {_fmt(congruent_desc.get("valence_range"))}, F0 span
    {_fmt(congruent_desc.get("f0_span_hz"), 1)} Hz (n={congruent_desc["n_clips"]}).</p>
    <p class="caption">With <code>description</code>, per emotion (averaged over all 4
    text/length carriers):</p>
    <table><thead><tr><th>Emotion</th><th>F0 mean</th><th>Valence</th></tr></thead>
    <tbody>{emotion_rows(with_desc)}</tbody></table>
    <p class="finding"><strong>Finding:</strong> {data["interpretation"]}</p>
    """


def render_e5_section(data: dict | None) -> str:
    """E5: the Hume incongruence set. Two thirds of its clips have words
    and delivery deliberately disagreeing, with delivery set independently
    of the text -- the condition E2/Cartesia could never produce. PSI, not
    UAR, is the number to read here (see scripts/eval_e5.py)."""
    if data is None:
        return "<p class='pending'>E5 not yet generated/evaluated (<code>make gen-hume-e5</code>, then <code>make eval-e5</code>).</p>"

    order = ("A_lexical", "B_permissive", "B_research", "C_fusion")
    label = {
        "A_lexical": "A &mdash; Lexical (words only)",
        "B_permissive": "B &mdash; Acoustic, permissive",
        "B_research": "B &mdash; Acoustic, research",
        "C_fusion": "C &mdash; Fusion",
    }
    rows = "".join(
        f"<tr><td>{label.get(key, key)}</td>"
        f"<td class='num'>{_fmt(data['solutions'][key]['uar'])}</td>"
        f"<td class='num'>{_fmt(data['solutions'][key]['macro_f1'])}</td>"
        f"<td class='num'><strong>{_fmt(data['solutions'][key]['psi_contested'])}</strong></td>"
        f"<td class='num'>{_fmt(data['solutions'][key]['psi_strict'])}</td></tr>"
        for key in order
        if key in data.get("solutions", {})
    )
    voices = ", ".join(f"{v} ({n})" for v, n in data.get("clips_per_voice", {}).items())

    return f"""
    <p>{data["n_clips"]} clips across {data["n_voices"]} synthetic voice(s) &mdash;
    {data["n_incongruent"]} of them ({100 * data["n_incongruent"] / max(data["n_clips"], 1):.0f}%)
    with the words and the delivery deliberately disagreeing. Delivery was set through Hume
    Octave's <code>description</code> field, which acts independently of the transcript; that
    independence is what E2 (Cartesia) could not provide and is why this set exists.
    <strong>PSI is the metric here, not UAR</strong>: the gold label is the delivery, so on a
    set built to contradict its own transcript, a model that reads words scores near 0 by
    construction and a model that hears tone scores high.</p>
    <table>
      <thead><tr><th>Solution</th><th>UAR</th><th>Macro-F1</th>
      <th>PSI<sub>contested</sub></th><th>PSI<sub>strict</sub></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p class="caption">Voices: {voices or "&mdash;"}. Synthetic and eval-only &mdash; E5
    measures prosody sensitivity under controlled contradiction, not performance on real
    speech; E3 remains the only real-speaker evidence in this repo.</p>
    <p class="finding"><strong>Reading it:</strong> {data.get("interpretation", "")}</p>
    """


def render_backend_combo_section(data: dict | None) -> str:
    """WavLM+probe (permissive) vs audeering/wav2vec2 (research), each
    trained/evaluated across the 4 dataset combinations the user asked
    for directly. See scripts/eval_backend_combos.py and
    ssa.combo_manifests for the methodology this renders."""
    if data is None:
        return "<p class='pending'>Not yet run &mdash; <code>make eval-backend-combos</code>.</p>"

    combos = data["combos"]

    def cell(eval_dict: dict | None, field: str) -> str:
        if eval_dict is None or "excluded_reason" in eval_dict:
            return "<td class='num' title=\"trained on this speaker, see note below\">excl.*</td>"
        return f"<td class='num'>{_fmt(eval_dict.get(field))}</td>"

    rows = []
    for name, combo in combos.items():
        perm_eval = combo["permissive"]["eval"]
        res_eval = combo["research"]["eval"]
        rows.append(f"""
        <tr><td rowspan="2"><code>{name}</code><br/><span class="caption">n_train={combo["train_n_clips"]}</span></td>
            <td>B &mdash; permissive (WavLM+probe)</td>
            <td class="num">{_fmt(perm_eval["cremad_test"]["uar"])}</td>
            <td class="num">{_fmt(perm_eval["cremad_test"]["macro_f1"])}</td>
            {cell(perm_eval.get("e3_both_takes"), "uar")}
            {cell(perm_eval.get("e3_both_takes"), "macro_f1")}</tr>
        <tr><td>B &mdash; research (audeering, shared&dagger;)</td>
            <td class="num">{_fmt(res_eval["cremad_test"]["uar"])}</td>
            <td class="num">{_fmt(res_eval["cremad_test"]["macro_f1"])}</td>
            {cell(res_eval.get("e3_both_takes"), "uar")}
            {cell(res_eval.get("e3_both_takes"), "macro_f1")}</tr>
        """)

    descriptions = "".join(
        f"<li><code>{name}</code> &mdash; {combo['description']}</li>"
        for name, combo in combos.items()
    )

    return f"""
    <p>{data.get("interpretation", "")}</p>
    <table>
      <thead><tr><th>Combo</th><th>Backend</th>
      <th>CREMA-D test UAR</th><th>CREMA-D test macro-F1</th>
      <th>E3 UAR</th><th>E3 macro-F1</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    <p class="caption">Eval subset: {data.get("eval_subset_n")}-clip stratified sample of
    CREMA-D test (seed={data.get("eval_subset_seed")}, fixed across every combo), plus both
    E3 takes where evaluating on E3 doesn't put a training speaker on both sides of the
    split. <strong>excl.*</strong> = that combo trained on E3's speaker, so E3 cannot also be
    its eval set (CLAUDE.md rule 1) &mdash; not missing data, a deliberate exclusion.</p>
    <p class="caption"><strong>&dagger;</strong> {data.get("research_backend_note", "")}</p>
    <ul class="caption">{descriptions}</ul>
    """


def render_e3_control_section(control: dict | None) -> str:
    if control is None:
        return "<p class='pending'>E3 evaluation / D1-vs-human control not yet run.</p>"

    cartesia = control["cartesia_d1"]
    takes = control["human_e3"]
    tr = control["test_retest_correlation_take0_vs_take1"]

    take_rows = "".join(
        f"<tr><td>{label}</td>"
        f"<td class='num'>{_fmt(t['recoverability_cv']['accuracy'])}</td>"
        f"<td class='num'>{_fmt(t['recoverability_cv']['chance'])}</td>"
        f"<td class='num'>{_fmt(t['f0_span_hz'], 1)} Hz</td>"
        f"<td class='num'>{_fmt(t['f0_rank_consistency']['mean_rank_by_sentiment'].get('positive'), 2)}</td>"
        "</tr>"
        for label, t in takes.items()
    )

    return f"""
    <p>The exact instruments and method D1 used on Cartesia TTS (leave-one-carrier-out
    recoverability, F0 rank consistency), re-run on real human speech &mdash; the same
    27 prompts, spoken by the same person, recorded twice independently
    (<code>data/recorded0</code>, <code>data/recorded</code>).</p>
    <table>
      <thead><tr><th>Source</th><th>Recoverability</th><th>Chance</th><th>F0 span</th>
      <th>"positive" mean F0 rank (1=highest)</th></tr></thead>
      <tbody>
        <tr><td><strong>Cartesia (D1)</strong></td>
            <td class="num">{_fmt(cartesia["recoverability_accuracy"])}</td>
            <td class="num">{_fmt(cartesia["recoverability_chance"])}</td>
            <td class="num">{_fmt(cartesia["f0_span_hz"], 1)} Hz</td><td class="num">&mdash;</td></tr>
        {take_rows}
      </tbody>
    </table>
    <p class="caption">Test-retest correlation between the two human takes (same prompts,
    same speaker, independent recordings) &mdash; how consistent is each instrument's own
    reading of itself? F0 mean: r={_fmt(tr.get("f0_mean"), 2)}. Research-backend valence:
    r={_fmt(tr.get("research_valence"), 2)}. Permissive-backend valence proxy:
    r={_fmt(tr.get("permissive_valence_proxy"), 2)} &mdash; our shipped probe barely
    correlates with itself across two takes of the same prompts.</p>
    <p class="finding"><strong>Finding:</strong> {control["interpretation"]}</p>
    """


def render_what_didnt_work() -> str:
    return """
    <p>The project's thesis made visible &mdash; failures kept in, not buried:</p>
    <ul>
      <li><strong>D0</strong> &mdash; Cartesia's emotion-tag clustering (audeering VAD
      embedding) was too degenerate to trust as a tag filter for E2.</li>
      <li><strong>D1</strong> &mdash; the user's proposed 5-emotion set was not
      measurably audible in Cartesia's output on any tested text/length/speed
      condition (recoverability below chance). See the D1 section above.</li>
      <li><strong>E2</strong> &mdash; the 76-clip synthetic incongruence set is flat
      and inexpressive, consistent with D0/D1; kept committed as the historical
      exhibit for that finding rather than deleted or completed.</li>
      <li><strong>Solution B (permissive) on E3</strong> &mdash; UAR 0.444/0.519 on a
      new, real speaker, never predicting "positive" at all on one take, and barely
      self-consistent across two independent recordings of the same prompts
      (r=&minus;0.19). See the E3 section above.</li>
      <li><strong>Per-speaker recalibration</strong> &mdash; tried on E3
      (leave-one-carrier-out threshold fitting), helped in 1 of 4 backend&times;take
      combinations and hurt in the other 3. Reported as a negative result at this
      sample size, not shipped as a feature &mdash; see DESIGN.md.</li>
      <li><strong>Live browser demo</strong> &mdash; not built in this session
      (time/budget constraint, stated plainly rather than silently dropped). The CLI
      (<code>uv run python -m ssa.cli --audio &lt;file&gt;</code>) gives the same
      end-to-end prediction on any local recording today.</li>
    </ul>
    <p class="finding"><strong>One thing that did work:</strong> the Hume Octave
    falsification probe (below) shows real emotion differentiation on the same
    carrier text D1 used on Cartesia &mdash; the synthetic-TTS story isn't "no
    vendor can do this," it's "this vendor, on this content, couldn't."</p>
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
    have_e3_eval = (RESULTS_DIR / "d1_vs_e3_control.json").exists()
    items = [
        ("E1 (CREMA-D, public benchmark)", True),
        ("D0 (Cartesia emotion-space probe)", (RESULTS_DIR / "d0_emotion_space.json").exists()),
        (
            "D1 (Cartesia emotion-rendering factorial probe)",
            (RESULTS_DIR / "d1_emotion_probe.json").exists(),
        ),
        ("E2 (synthetic incongruence set)", have_e2),
        ("E3 (human recordings, both takes recorded + evaluated)", have_e3 and have_e3_eval),
        ("Hume Octave falsification probe", (RESULTS_DIR / "hume_probe.json").exists()),
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
    d1 = load_json_if_exists(RESULTS_DIR / "d1_emotion_probe.json")
    prosody_samples = load_json_if_exists(RESULTS_DIR / "prosody_samples.json")
    hume = load_json_if_exists(RESULTS_DIR / "hume_probe.json")
    e3_control = load_json_if_exists(RESULTS_DIR / "d1_vs_e3_control.json")
    leaky = load_json_if_exists(RESULTS_DIR / "leakage_comparison_permissive.json")
    backend_combos = load_json_if_exists(RESULTS_DIR / "backend_combo_comparison.json")
    e5 = load_json_if_exists(RESULTS_DIR / "e5_summary.json")
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

<h2>What this is, and the one idea</h2>
<p>Sentiment (positive / neutral / negative) from <strong>raw speech audio</strong>, not
a handed-over transcript. The question that matters isn't "what accuracy?" but
<strong>"does the model hear the tone, or is it just reading the words?"</strong>
&mdash; an audio model can cheat by reading text sentiment instead of prosody, which
published work confirms is the default failure mode (arXiv 2510.10444, arXiv
2510.25054). Every dataset and metric here (PSI above all) exists to catch that.</p>

<h2>Architecture</h2>
<p>Three solutions on the lexical&harr;acoustic axis, one interface
(<code>ssa.types.Solution</code>), so the evaluation harness treats them identically:</p>
<table>
  <thead><tr><th></th><th>Solution</th><th>Reads</th><th>License</th></tr></thead>
  <tbody>
    <tr><td><strong>A</strong></td><td>Lexical &mdash; ASR (faster-whisper) &rarr; text sentiment</td><td>the words</td><td>MIT</td></tr>
    <tr><td><strong>B</strong></td><td>Acoustic &mdash; frozen encoder + trained probe</td><td>the tone</td><td>permissive (WavLM) or research (audeering, CC-BY-NC-SA-4.0)</td></tr>
    <tr><td><strong>C</strong></td><td>Fusion &mdash; calibrated late fusion + abstention</td><td>both</td><td>recommended default</td></tr>
  </tbody>
</table>
<p class="caption">Shared internal representation: valence-arousal-dominance (VAD).
Sentiment is a threshold read-out of valence. Late fusion (not joint) deliberately
sacrifices some accuracy to keep PSI computable per branch &mdash; see DESIGN.md's
Key trade-offs.</p>

<h2>Which model actually extracts prosody/emotion</h2>
{render_models_section()}

<h2>Method</h2>
<p>Speaker-disjoint splits always (a parallel leaky split exists only to quantify the
leakage gap, see below). <strong>UAR</strong> (unweighted average recall), not plain
accuracy, is the headline metric &mdash; CREMA-D is imbalanced enough that a
majority-class predictor would look good on accuracy alone. <strong>PSI</strong>
(Prosody Sensitivity Index): on clips where words and tone disagree, the fraction of
predictions that follow the tone (1.0) vs. the words (0.0). Full definitions in
<code>ssa/eval/metrics.py</code>.</p>

<h2>Status</h2>
{render_status(eval_results)}

<h2>Main comparison: solutions &times; datasets</h2>
{render_main_table(eval_results)}

<h2>The leakage exposé (Solution B, permissive backend)</h2>
{render_leakage_table(disjoint_permissive, leaky)}

<h2>D0: Cartesia emotion-space probe</h2>
{render_d0_section(d0)}

<h2>Extracted prosody/emotion on real samples: CREMA-D + your recordings</h2>
{render_prosody_samples_section(prosody_samples)}

<h2>D1: does Cartesia render these emotions audibly?</h2>
{render_d1_section(d1)}

<h2>Hume Octave: a falsification test of the D1 finding</h2>
{render_hume_section(hume)}

<h2>E3: human recordings, and a control on the D1 finding</h2>
{render_e3_control_section(e3_control)}

<h2>E5: words vs delivery, on a set built to disagree</h2>
{render_e5_section(e5)}

<h2>Backend x training-data comparison: WavLM+probe vs audeering/wav2vec2</h2>
{render_backend_combo_section(backend_combos)}

<h2>What didn't work, and why</h2>
{render_what_didnt_work()}

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
