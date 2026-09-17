#!/usr/bin/env python3
"""D1: factorial probe into WHETHER (and which factor makes) Cartesia's
emotion tags audible on the user's 5-emotion set.

Motivation, from results/d0_emotion_space.json and DESIGN.md: D0 measured
the user's proposed 5 emotions (happy, sad, angry, calm, frustrated) as
spanning 2.8x more valence than E2's 3 tags -- but in a SCRAMBLED order
(frustrated > happy; angry > calm). That is consistent with either a
measurement-instrument problem (D0's own domain-gap finding: the audeering
VAD model doesn't reliably decode valence from synthetic TTS prosody) or
genuinely inert rendering. This probe tells the two apart using SEVERAL
independent instruments, not just the one D0 already flagged as
untrustworthy on synthetic speech, and produces a page the user can
actually listen to -- the real arbiter, since measurement can be wrong in
ways listening isn't.

Grid (40 clips, 1 carrier per cell): 5 emotions x text_condition
(neutral/congruent) x length (short/long) x speed (default/adjusted).
`text_condition=neutral` uses IDENTICAL words across all 5 emotions at a
given length (ssa.carriers.D1_NEUTRAL_TEXT) -- the condition the project
actually wants, since it isolates prosody from lexical content entirely.
`text_condition=congruent` uses per-emotion words (Cartesia's own docs say
emotion tags "only work when the emotion is consistent with the
transcript").

Plus ~10 clips comparing sonic-3 vs sonic-3.5: the main grid is generated
on sonic-3, and the 5 emotions in whichever cell shows the widest F0-mean
spread (a cheap, pre-registered tie-breaker -- see `pick_best_cell`) are
regenerated on sonic-3.5. The sonic-3 half of that comparison is reused
from the main grid, not regenerated.

Measured with 4 independent instruments: descriptive F0 stats (parselmouth,
no fitting), eGeMAPS (openSMILE, no fitting), our own WavLM probe (trained
on real human emotional speech -- CREMA-D -- a model family independent of
audeering), and the audeering VAD model (continuity with D0). The headline
"intended-emotion recoverability" number is a logistic regression on the
no-fitting descriptive/eGeMAPS features, leave-one-carrier-out CV -- see
`recoverability_cv` for why it is reported as coarse, corroborating
evidence, not the primary evidence.

**Requires CARTESIA_API_KEY** (`pip install -e .[generate]` for the
`cartesia` package). Evaluation itself never depends on this script having
been run -- results/d1_emotion_probe.json, report/d1_listening.html, and
the clips under data/probes/d1/ are committed once generated, same
precedent as D0/E2.

Usage:
    uv run python scripts/gen_emotion_probe_d1.py
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from ssa.audio import load_clip  # noqa: E402
from ssa.carriers import (  # noqa: E402
    D1_CONGRUENT_TEXT,
    D1_EMOTION_SENTIMENT,
    D1_NEUTRAL_TEXT,
    D1_SPEED_ADJUSTED,
    EMOTIONS,
)
from ssa.paralinguistic import extract_features  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.tts import VOICE_IDS, generate_clip, get_client  # noqa: E402
from ssa.types import AudioClip, Sentiment  # noqa: E402
from ssa.voicehealth import pitch_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "data" / "probes" / "d1"
RESULT_PATH = REPO_ROOT / "results" / "d1_emotion_probe.json"
LISTENING_PAGE_PATH = REPO_ROOT / "report" / "d1_listening.html"

# One voice -- D1 isolates emotion rendering, not voice diversity.
VOICE_ID = VOICE_IDS["skylar"]
ALT_MODEL_ID = "sonic-3.5"

TEXT_CONDITIONS: tuple[str, ...] = ("neutral", "congruent")
LENGTHS: tuple[str, ...] = ("short", "long")
SPEED_CONDITIONS: tuple[str, ...] = ("default", "adjusted")

# The 9 no-fitting descriptive features fed to the recoverability
# classifier -- deliberately compact (not the full 88-dim eGeMAPS vector)
# given ~40 samples over 5 classes; see recoverability_cv's docstring.
FEATURE_NAMES: tuple[str, ...] = (
    "f0_mean",
    "f0_std",
    "f0_range",
    "egemaps_loudness",
    "egemaps_hnr",
    "egemaps_jitter",
    "egemaps_shimmer_db",
    "rms",
    "speech_rate_cps",
)


@dataclass(frozen=True, slots=True)
class D1ClipSpec:
    clip_id: str
    emotion: str
    text_condition: str  # "neutral" | "congruent"
    length: str  # "short" | "long"
    speed_condition: str  # "default" | "adjusted"
    text: str
    speed: float | None
    model_id: str
    carrier_group: str  # grouping key for leave-one-carrier-out CV

    @property
    def cell(self) -> tuple[str, str, str]:
        return (self.text_condition, self.length, self.speed_condition)


def _text_for(emotion: str, text_condition: str, length: str) -> str:
    if text_condition == "neutral":
        return D1_NEUTRAL_TEXT[length]
    return D1_CONGRUENT_TEXT[emotion][length]


def _speed_for(emotion: str, speed_condition: str) -> float | None:
    return D1_SPEED_ADJUSTED[emotion] if speed_condition == "adjusted" else None


def _carrier_group(emotion: str, text_condition: str, length: str) -> str:
    """The distinct-sentence identity, used as the leave-one-carrier-out CV
    group. Neutral text is IDENTICAL across all 5 emotions at a given
    length (that is the point -- isolate prosody), so its group key
    ignores emotion; congruent text is unique per emotion. This means a
    neutral-condition fold holds out ~10 clips at once (only 2 neutral
    groups exist) while a congruent-condition fold holds out just 2 (one
    emotion x length, both speed conditions) -- a real asymmetry, not a
    bug, and part of why this CV is reported as coarse."""
    if text_condition == "neutral":
        return f"neutral_{length}"
    return f"congruent_{emotion}_{length}"


def build_grid() -> list[D1ClipSpec]:
    """The 40-clip main factorial grid, one carrier per cell."""
    specs: list[D1ClipSpec] = []
    for emotion, text_condition, length, speed_condition in itertools.product(
        EMOTIONS, TEXT_CONDITIONS, LENGTHS, SPEED_CONDITIONS
    ):
        specs.append(
            D1ClipSpec(
                clip_id=f"d1_{emotion}_{text_condition}_{length}_{speed_condition}",
                emotion=emotion,
                text_condition=text_condition,
                length=length,
                speed_condition=speed_condition,
                text=_text_for(emotion, text_condition, length),
                speed=_speed_for(emotion, speed_condition),
                model_id="sonic-3",
                carrier_group=_carrier_group(emotion, text_condition, length),
            )
        )
    return specs


def pick_best_cell(
    grid: list[D1ClipSpec], f0_means: dict[str, float | None]
) -> tuple[str, str, str]:
    """The (text_condition, length, speed_condition) cell whose 5 emotions
    show the widest F0-mean spread -- a cheap, pre-registered tie-breaker
    for which cell gets the sonic-3.5 comparison. F0 mean is chosen
    because it needs no model inference and is the most direct acoustic
    correlate of emotional arousal available before any classifier runs."""
    by_cell: dict[tuple[str, str, str], list[float]] = {}
    for spec in grid:
        f0 = f0_means.get(spec.clip_id)
        if f0 is not None:
            by_cell.setdefault(spec.cell, []).append(f0)
    spreads = {cell: (max(vals) - min(vals)) for cell, vals in by_cell.items() if len(vals) >= 2}
    if not spreads:
        raise ValueError("no cell had at least 2 measurable F0 means -- cannot pick a best cell")
    return max(spreads, key=lambda cell: spreads[cell])


def build_model_comparison(best_cell: tuple[str, str, str]) -> list[D1ClipSpec]:
    """5 extra clips: the best cell's 5 emotions, regenerated on
    sonic-3.5. The sonic-3 half already exists in the main grid."""
    text_condition, length, speed_condition = best_cell
    specs: list[D1ClipSpec] = []
    for emotion in EMOTIONS:
        specs.append(
            D1ClipSpec(
                clip_id=f"d1_{emotion}_{text_condition}_{length}_{speed_condition}_sonic35",
                emotion=emotion,
                text_condition=text_condition,
                length=length,
                speed_condition=speed_condition,
                text=_text_for(emotion, text_condition, length),
                speed=_speed_for(emotion, speed_condition),
                model_id=ALT_MODEL_ID,
                carrier_group=_carrier_group(emotion, text_condition, length),
            )
        )
    return specs


def _generate(client, spec: D1ClipSpec) -> Path:
    path = OUT_DIR / f"{spec.clip_id}.wav"
    if not path.exists():
        generate_clip(
            client,
            text=spec.text,
            emotion=spec.emotion,
            voice_id=VOICE_ID,
            out_path=path,
            speed=spec.speed,
            model_id=spec.model_id,
        )
        logger.info("generated %s", spec.clip_id)
    return path


@dataclass(frozen=True, slots=True)
class Measurement:
    clip_id: str
    duration_s: float
    rms: float
    speech_rate_cps: float  # chars of source text / duration
    f0_mean: float | None
    f0_std: float | None
    f0_range: float | None
    egemaps_loudness: float | None
    egemaps_hnr: float | None
    egemaps_jitter: float | None
    egemaps_shimmer_db: float | None
    permissive_valence_proxy: float | None  # P(positive) - P(negative), our WavLM probe
    permissive_predicted_sentiment: str | None
    research_valence: float | None  # audeering VAD, continuity with D0
    research_arousal: float | None


def measure_clip(
    clip: AudioClip,
    text: str,
    acoustic_permissive: AcousticSolution,
    acoustic_research: AcousticSolution,
) -> Measurement:
    rms = float(np.sqrt(np.mean(clip.samples.astype(np.float64) ** 2)))
    pstats = pitch_stats(clip)

    egemaps_loudness = egemaps_hnr = egemaps_jitter = egemaps_shimmer_db = None
    try:
        feats = extract_features(clip)
        egemaps_loudness = feats.loudness_mean
        egemaps_hnr = feats.hnr_mean
        egemaps_jitter = feats.jitter_mean
        egemaps_shimmer_db = feats.shimmer_db_mean
    except Exception:
        logger.warning("eGeMAPS extraction failed for %s", clip.clip_id)

    perm_pred = acoustic_permissive.predict(clip)
    valence_proxy = perm_pred.probs[Sentiment.POSITIVE] - perm_pred.probs[Sentiment.NEGATIVE]

    res_pred = acoustic_research.predict(clip)
    research_valence = res_pred.vad.valence if res_pred.vad is not None else None
    research_arousal = res_pred.vad.arousal if res_pred.vad is not None else None

    return Measurement(
        clip_id=clip.clip_id,
        duration_s=clip.duration_s,
        rms=rms,
        speech_rate_cps=len(text) / clip.duration_s if clip.duration_s > 0 else 0.0,
        f0_mean=pstats.f0_mean,
        f0_std=pstats.f0_std,
        f0_range=pstats.f0_range,
        egemaps_loudness=egemaps_loudness,
        egemaps_hnr=egemaps_hnr,
        egemaps_jitter=egemaps_jitter,
        egemaps_shimmer_db=egemaps_shimmer_db,
        permissive_valence_proxy=valence_proxy,
        permissive_predicted_sentiment=perm_pred.sentiment.value,
        research_valence=research_valence,
        research_arousal=research_arousal,
    )


def recoverability_cv(
    specs: list[D1ClipSpec], measurements: dict[str, Measurement]
) -> dict[str, object]:
    """Leave-one-carrier-out CV, logistic regression on FEATURE_NAMES,
    predicting the intended emotion. Chance = 1/5 = 0.20.

    Explicitly a corroborating signal, not the primary evidence (CLAUDE.md
    rule 6: don't claim measured what was only coarsely fit) -- ~40 samples
    over 5 classes is a small fit. The descriptive stats in the report's
    per_emotion_summary carry the argument; this number either supports or
    complicates that reading."""
    X: list[list[float]] = []
    y: list[str] = []
    groups: list[str] = []
    for spec in specs:
        m = measurements[spec.clip_id]
        row = [getattr(m, name) for name in FEATURE_NAMES]
        if any(v is None for v in row):
            continue
        X_row: list[float] = [float(v) for v in row]
        X.append(X_row)
        y.append(spec.emotion)
        groups.append(spec.carrier_group)

    n_dropped = len(specs) - len(y)
    if len(set(groups)) < 2:
        return {
            "accuracy": None,
            "n": len(y),
            "n_dropped_missing_features": n_dropped,
            "chance": 1 / len(EMOTIONS),
            "note": "fewer than 2 carrier groups -- CV skipped",
        }

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)

    logo = LeaveOneGroupOut()
    correct = 0
    total = 0
    for train_idx, test_idx in logo.split(X_arr, y_arr, groups_arr):
        if len(set(y_arr[train_idx])) < 2:
            continue  # can't fit a classifier with a single training class
        scaler = StandardScaler().fit(X_arr[train_idx])
        clf = LogisticRegression(max_iter=1000).fit(
            scaler.transform(X_arr[train_idx]), y_arr[train_idx]
        )
        preds = clf.predict(scaler.transform(X_arr[test_idx]))
        correct += int((preds == y_arr[test_idx]).sum())
        total += len(test_idx)

    return {
        "accuracy": (correct / total) if total else None,
        "n": total,
        "n_dropped_missing_features": n_dropped,
        "chance": 1 / len(EMOTIONS),
        "n_carrier_groups": len(set(groups)),
    }


def _sentiment_ordering_holds(per_emotion_summary: dict, field: str) -> bool | None:
    """True if this instrument's mean per-emotion reading is ordered
    positive > neutral > negative by D1_EMOTION_SENTIMENT -- the same
    scrambled-order check D0 ran (results/d0_emotion_space.json found
    frustrated > happy and angry > calm on the audeering VAD model). None
    if any group is missing (mixed emotions per sentiment bucket are
    averaged, then compared as group means)."""
    by_sentiment: dict[str, list[float]] = {"positive": [], "neutral": [], "negative": []}
    for emotion, summary in per_emotion_summary.items():
        value = summary.get(field)
        if value is None:
            return None
        by_sentiment[D1_EMOTION_SENTIMENT[emotion].value].append(value)
    pos = float(np.mean(by_sentiment["positive"]))
    neu = float(np.mean(by_sentiment["neutral"]))
    neg = float(np.mean(by_sentiment["negative"]))
    return bool(pos > neu > neg)


def _cell_label(cell: tuple[str, str, str]) -> str:
    text_condition, length, speed_condition = cell
    return f"{text_condition} text, {length}, speed={speed_condition}"


def render_listening_page(
    grid: list[D1ClipSpec],
    comparison: list[D1ClipSpec],
    measurements: dict[str, Measurement],
    best_cell: tuple[str, str, str],
) -> str:
    """Matched A/B pairs the user can play back-to-back: one section per
    grid cell (5 emotions, same words when text_condition=neutral), plus a
    sonic-3 vs sonic-3.5 section for the best cell. This is the real
    arbiter (see module docstring) -- measurement can be wrong in ways
    listening isn't."""

    def audio_row(spec: D1ClipSpec) -> str:
        m = measurements.get(spec.clip_id)
        rel_path = f"../data/probes/d1/{spec.clip_id}.wav"
        stats = (
            f"F0 mean={m.f0_mean:.0f}Hz std={m.f0_std:.0f}Hz | "
            f"permissive={m.permissive_predicted_sentiment} "
            f"(P+ - P-={m.permissive_valence_proxy:+.2f}) | "
            f"research valence={m.research_valence:.2f}"
            if m is not None and m.f0_mean is not None and m.research_valence is not None
            else "(measurement unavailable)"
        )
        return (
            "<tr>"
            f"<td>{spec.emotion}</td>"
            f'<td><audio controls src="{rel_path}"></audio></td>'
            f"<td class='stats'>{stats}</td>"
            "</tr>"
        )

    sections: list[str] = []
    for cell in itertools.product(TEXT_CONDITIONS, LENGTHS, SPEED_CONDITIONS):
        cell_specs = [s for s in grid if s.cell == cell]
        if not cell_specs:
            continue
        cell_specs.sort(key=lambda s: EMOTIONS.index(s.emotion))
        marker = " (chosen for sonic-3.5 comparison below)" if cell == best_cell else ""
        rows = "\n".join(audio_row(s) for s in cell_specs)
        sections.append(
            f"<h2>{_cell_label(cell)}{marker}</h2>\n"
            f'<p class="text">"{cell_specs[0].text if cell[0] == "neutral" else "(per-emotion text, see table)"}"</p>\n'
            f"<table><tr><th>Emotion</th><th>Audio</th><th>Measured</th></tr>\n{rows}\n</table>"
        )

    comparison_html = "\n".join(
        "<tr>"
        f"<td>{emotion}</td>"
        f'<td>sonic-3<br><audio controls src="../data/probes/d1/{s3.clip_id}.wav"></audio></td>'
        f'<td>sonic-3.5<br><audio controls src="../data/probes/d1/{s35.clip_id}.wav"></audio></td>'
        "</tr>"
        for emotion, s3, s35 in (
            (
                emotion,
                next(s for s in grid if s.cell == best_cell and s.emotion == emotion),
                next(s for s in comparison if s.emotion == emotion),
            )
            for emotion in EMOTIONS
        )
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>D1 listening page -- emotion rendering probe</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: .25rem; }}
  .text {{ color: #555; font-style: italic; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .5rem; }}
  th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eee; vertical-align: middle; }}
  td.stats {{ font-family: ui-monospace, monospace; font-size: .8rem; color: #444; }}
  audio {{ height: 32px; }}
</style>
</head>
<body>
<h1>D1: does Cartesia render these 5 emotions audibly?</h1>
<p>Each section below is one (text, length, speed) cell with all 5 emotions on the same
words (when text is "neutral") or on matched-domain per-emotion sentences (when
"congruent"). Play them back-to-back within a section -- that A/B comparison is the real
test, more than any single number below each player. See results/d1_emotion_probe.json
for the full write-up and CLAUDE.md rule 6: none of the numbers here are claimed as more
than what a ~40-clip synthetic probe can support.</p>
{"".join(sections)}
<h2>sonic-3 vs sonic-3.5, on {_cell_label(best_cell)}</h2>
<table><tr><th>Emotion</th><th>sonic-3</th><th>sonic-3.5</th></tr>
{comparison_html}
</table>
</body>
</html>
"""


def main() -> None:
    client = get_client()
    grid = build_grid()

    for spec in grid:
        _generate(client, spec)

    # Cheap F0-only pass (no model loading) to pick the sonic-3.5 comparison cell.
    f0_means: dict[str, float | None] = {}
    for spec in grid:
        clip = load_clip(OUT_DIR / f"{spec.clip_id}.wav", clip_id=spec.clip_id)
        f0_means[spec.clip_id] = pitch_stats(clip).f0_mean

    best_cell = pick_best_cell(grid, f0_means)
    logger.info("best cell for sonic-3.5 comparison: %s", best_cell)
    comparison = build_model_comparison(best_cell)
    for spec in comparison:
        _generate(client, spec)

    all_specs = grid + comparison

    logger.info("loading acoustic solutions (permissive + research)...")
    acoustic_permissive = AcousticSolution(backend="permissive")
    acoustic_research = AcousticSolution(backend="research")

    measurements: dict[str, Measurement] = {}
    for spec in all_specs:
        clip = load_clip(OUT_DIR / f"{spec.clip_id}.wav", clip_id=spec.clip_id)
        measurements[spec.clip_id] = measure_clip(
            clip, spec.text, acoustic_permissive, acoustic_research
        )

    cv_result = recoverability_cv(grid, measurements)

    per_emotion_summary = {}
    for emotion in EMOTIONS:
        emotion_measurements = [measurements[s.clip_id] for s in grid if s.emotion == emotion]
        f0_vals = [m.f0_mean for m in emotion_measurements if m.f0_mean is not None]
        valence_vals = [
            m.permissive_valence_proxy
            for m in emotion_measurements
            if m.permissive_valence_proxy is not None
        ]
        research_vals = [
            m.research_valence for m in emotion_measurements if m.research_valence is not None
        ]
        per_emotion_summary[emotion] = {
            "intended_sentiment": D1_EMOTION_SENTIMENT[emotion].value,
            "mean_f0": float(np.mean(f0_vals)) if f0_vals else None,
            "mean_permissive_valence_proxy": float(np.mean(valence_vals)) if valence_vals else None,
            "mean_research_valence": float(np.mean(research_vals)) if research_vals else None,
            "n": len(emotion_measurements),
        }

    f0_span = None
    means_by_emotion = {e: per_emotion_summary[e]["mean_f0"] for e in EMOTIONS}
    if all(v is not None for v in means_by_emotion.values()):
        f0_span = max(means_by_emotion.values()) - min(means_by_emotion.values())

    # Same check D0 ran and found scrambled (frustrated > happy; angry >
    # calm): does each instrument's valence-like reading, averaged per
    # emotion, come out ordered positive > neutral > negative -- the
    # ordering a working instrument on real emotion should produce?
    valence_ordering = {
        "permissive": _sentiment_ordering_holds(
            per_emotion_summary, "mean_permissive_valence_proxy"
        ),
        "research": _sentiment_ordering_holds(per_emotion_summary, "mean_research_valence"),
    }

    result = {
        "grid_size": len(grid),
        "comparison_size": len(comparison),
        "best_cell": {
            "text_condition": best_cell[0],
            "length": best_cell[1],
            "speed_condition": best_cell[2],
        },
        "recoverability_cv": cv_result,
        "per_emotion_summary": per_emotion_summary,
        "f0_span_across_emotions_hz": f0_span,
        "valence_ordering_matches_intended_sentiment": valence_ordering,
        "per_clip": [
            {**asdict(spec), **{f"m_{k}": v for k, v in asdict(measurements[spec.clip_id]).items()}}
            for spec in all_specs
        ],
        "interpretation": (
            "Coarse, ~40-clip synthetic probe (CLAUDE.md rule 6: not claimed as more than "
            "that). recoverability_cv's accuracy is a leave-one-carrier-out logistic "
            "regression on no-fitting descriptive/eGeMAPS features (chance=0.20, 5-way); "
            "per_emotion_summary's F0/valence means are the primary descriptive evidence "
            "and need no fitting. Listen at report/d1_listening.html -- it is the real "
            "arbiter this probe exists to produce, since measurement can be wrong in ways "
            "listening isn't (see D0's own domain-gap finding, results/d0_emotion_space.json)."
        ),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))

    listening_html = render_listening_page(grid, comparison, measurements, best_cell)
    LISTENING_PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LISTENING_PAGE_PATH.write_text(listening_html)

    logger.info(
        "done: %d clips (%d grid + %d comparison), best_cell=%s, cv_accuracy=%s -> %s, %s",
        len(all_specs),
        len(grid),
        len(comparison),
        best_cell,
        cv_result["accuracy"],
        RESULT_PATH,
        LISTENING_PAGE_PATH,
    )


if __name__ == "__main__":
    main()
