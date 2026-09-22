#!/usr/bin/env python3
"""Falsification test for D1's Cartesia finding, at D1's own scale: does
Hume Octave render these 5 emotions audibly via its `description` field --
documented to control delivery independently of the transcript, which is
exactly the capability Cartesia's docs say it lacks?

Grid (40 clips): 5 emotions (happy/sad/angry/calm/frustrated) x
text_condition (neutral/congruent) x length (short/long) x description
(on/off), one fixed voice. Carrier text is reused verbatim from
`ssa.carriers` -- D1_NEUTRAL_TEXT (identical wording across all 5 emotions,
isolating prosody from lexical content) and D1_CONGRUENT_TEXT (per-emotion
wording) -- so this is the exact same 4-factor design D1 ran on Cartesia,
on the exact same words, just with `description` (present in D1's design
only as "always absent", here varied) as the added factor. That gives:

  - genuine carrier diversity (12 groups, same structure as D1's) for an
    honest leave-one-carrier-out recoverability classifier, run separately
    for the description-on and description-off halves (20 clips/12 groups
    each) -- the small first probe (n=10, 1 carrier) could not support this;
  - a direct test of whether `description` works even on NEUTRAL text,
    the harder and more important case (isolating prosody from wording is
    the whole point of this project, per CLAUDE.md's gold-label rule).

Measured with the same 9 features and method as D1/E3 (F0 via
ssa.voicehealth.pitch_stats, eGeMAPS via ssa.paralinguistic.extract_features,
RMS, speech rate, plus the research VAD model) for direct numeric
comparability across all three probes.

**Requires HUME_API_KEY** (`pip install hume`). Deliberately still small
next to a real production dataset (this is a probe, not a shipped E5) --
scoped up from n=10 to n=40 to make the recoverability number honest, not
scoped up to "a full dataset."

Usage:
    uv run python scripts/probe_hume.py
"""

from __future__ import annotations

import asyncio
import base64
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
    EMOTIONS,
)
from ssa.paralinguistic import extract_features  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.types import AudioClip, Sentiment  # noqa: E402
from ssa.voicehealth import pitch_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "data" / "probes" / "hume"
RESULT_PATH = REPO_ROOT / "results" / "hume_probe.json"

VOICE_NAME = "Ava Song"  # fixed across all clips -- only text_condition/length/description vary
TEXT_CONDITIONS: tuple[str, ...] = ("neutral", "congruent")
LENGTHS: tuple[str, ...] = ("short", "long")

# Short, "precise" emotion descriptions per Hume's own best-practice guidance
# (<=100 chars; concrete emotion words, not broad categories). Delivery
# instructions are independent of the transcript, so the same description
# applies whether the text_condition is neutral or congruent.
DESCRIPTIONS: dict[str, str] = {
    "happy": "happy, upbeat, smiling tone",
    "sad": "sad, downcast, weary tone",
    "angry": "angry, sharp, irritated tone",
    "calm": "calm, relaxed, gentle tone",
    "frustrated": "frustrated, tense, exasperated tone",
}

# Free-tier Hume TTS is rate-limited; fixed spacing between requests plus
# retry-with-backoff on 429 keeps a 40-clip run from crashing partway
# through (it did, once, at clip 28/40 with no handling at all).
REQUEST_SPACING_S = 3.0
MAX_RETRIES = 5
RETRY_BASE_DELAY_S = 10.0

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
class HumeClipSpec:
    clip_id: str
    emotion: str
    text_condition: str  # "neutral" | "congruent"
    length: str  # "short" | "long"
    has_description: bool
    text: str
    carrier_group: str


def _text_for(emotion: str, text_condition: str, length: str) -> str:
    if text_condition == "neutral":
        return D1_NEUTRAL_TEXT[length]
    return D1_CONGRUENT_TEXT[emotion][length]


def _carrier_group(emotion: str, text_condition: str, length: str) -> str:
    """Same definition as D1's -- neutral text is shared across all 5
    emotions at a given length (2 groups total), congruent text is unique
    per emotion (10 groups) -- see ssa.carriers / gen_emotion_probe_d1.py."""
    if text_condition == "neutral":
        return f"neutral_{length}"
    return f"congruent_{emotion}_{length}"


def build_grid() -> list[HumeClipSpec]:
    specs = []
    for emotion, text_condition, length, has_description in itertools.product(
        EMOTIONS, TEXT_CONDITIONS, LENGTHS, (True, False)
    ):
        suffix = "desc" if has_description else "nodesc"
        specs.append(
            HumeClipSpec(
                clip_id=f"hume_{emotion}_{text_condition}_{length}_{suffix}",
                emotion=emotion,
                text_condition=text_condition,
                length=length,
                has_description=has_description,
                text=_text_for(emotion, text_condition, length),
                carrier_group=_carrier_group(emotion, text_condition, length),
            )
        )
    return specs


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()


async def _generate_one(client, spec: HumeClipSpec) -> None:
    from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName

    path = OUT_DIR / f"{spec.clip_id}.wav"
    if path.exists():
        return
    kwargs: dict[str, object] = dict(
        text=spec.text,
        voice=PostedUtteranceVoiceWithName(name=VOICE_NAME, provider="HUME_AI"),
        speed=1.0,
    )
    if spec.has_description:
        kwargs["description"] = DESCRIPTIONS[spec.emotion]
    utterance = PostedUtterance(**kwargs)

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await client.tts.synthesize_json(
                utterances=[utterance], format={"type": "wav"}
            )
            break
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY_S * (2**attempt)
            logger.warning(
                "rate limited on %s (attempt %d/%d), backing off %.0fs",
                spec.clip_id,
                attempt + 1,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

    audio = base64.b64decode(result.generations[0].audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    logger.info("generated %s", spec.clip_id)


async def generate_all(specs: list[HumeClipSpec]) -> None:
    import os

    from hume import AsyncHumeClient

    client = AsyncHumeClient(api_key=os.environ["HUME_API_KEY"])
    for spec in specs:
        await _generate_one(client, spec)
        await asyncio.sleep(REQUEST_SPACING_S)


@dataclass(frozen=True, slots=True)
class Measurement:
    clip_id: str
    f0_mean: float | None
    f0_std: float | None
    f0_range: float | None
    egemaps_loudness: float | None
    egemaps_hnr: float | None
    egemaps_jitter: float | None
    egemaps_shimmer_db: float | None
    rms: float | None
    speech_rate_cps: float | None
    research_valence: float | None
    research_arousal: float | None
    permissive_predicted_sentiment: str | None
    permissive_valence_proxy: float | None


def _measure_one(
    clip: AudioClip,
    text: str,
    acoustic_permissive: AcousticSolution,
    acoustic_research: AcousticSolution,
) -> Measurement:
    pstats = pitch_stats(clip)
    egemaps_loudness = egemaps_hnr = egemaps_jitter = egemaps_shimmer_db = None
    try:
        feats = extract_features(clip)
        egemaps_loudness, egemaps_hnr = feats.loudness_mean, feats.hnr_mean
        egemaps_jitter, egemaps_shimmer_db = feats.jitter_mean, feats.shimmer_db_mean
    except Exception:
        logger.warning("eGeMAPS extraction failed for %s", clip.clip_id)

    perm_pred = acoustic_permissive.predict(clip)
    res_pred = acoustic_research.predict(clip)
    rms = float(np.sqrt(np.mean(clip.samples.astype(np.float64) ** 2)))

    return Measurement(
        clip_id=clip.clip_id,
        f0_mean=pstats.f0_mean,
        f0_std=pstats.f0_std,
        f0_range=pstats.f0_range,
        egemaps_loudness=egemaps_loudness,
        egemaps_hnr=egemaps_hnr,
        egemaps_jitter=egemaps_jitter,
        egemaps_shimmer_db=egemaps_shimmer_db,
        rms=rms,
        speech_rate_cps=len(text) / clip.duration_s if clip.duration_s > 0 else None,
        research_valence=res_pred.vad.valence if res_pred.vad is not None else None,
        research_arousal=res_pred.vad.arousal if res_pred.vad is not None else None,
        permissive_predicted_sentiment=perm_pred.sentiment.value,
        permissive_valence_proxy=(
            perm_pred.probs[Sentiment.POSITIVE] - perm_pred.probs[Sentiment.NEGATIVE]
        ),
    )


def measure_all(specs: list[HumeClipSpec]) -> dict[str, Measurement]:
    acoustic_permissive = AcousticSolution(backend="permissive")
    acoustic_research = AcousticSolution(backend="research")
    out = {}
    for spec in specs:
        clip = load_clip(OUT_DIR / f"{spec.clip_id}.wav", clip_id=spec.clip_id)
        out[spec.clip_id] = _measure_one(clip, spec.text, acoustic_permissive, acoustic_research)
    return out


def recoverability_cv(
    specs: list[HumeClipSpec], measurements: dict[str, Measurement]
) -> dict[str, object]:
    """Leave-one-carrier-out logistic regression on FEATURE_NAMES, same
    method as scripts/gen_emotion_probe_d1.py's recoverability_cv --
    directly comparable to D1's 0.175 (vs 0.200 chance, 5-way)."""
    X: list[list[float]] = []
    y: list[str] = []
    groups: list[str] = []
    for spec in specs:
        m = measurements[spec.clip_id]
        row = [getattr(m, name) for name in FEATURE_NAMES]
        if any(v is None for v in row):
            continue
        X.append([float(v) for v in row])
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

    X_arr, y_arr, groups_arr = np.asarray(X), np.asarray(y), np.asarray(groups)
    logo = LeaveOneGroupOut()
    correct = total = 0
    for train_idx, test_idx in logo.split(X_arr, y_arr, groups_arr):
        if len(set(y_arr[train_idx])) < 2:
            continue
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


def summarize(specs: list[HumeClipSpec], measurements: dict[str, Measurement]) -> dict:
    def group_summary(has_description: bool) -> dict:
        group = [s for s in specs if s.has_description == has_description]

        f0_by_emotion: dict[str, list[float]] = {}
        valence_by_emotion: dict[str, list[float]] = {}
        for s in group:
            m = measurements[s.clip_id]
            if m.f0_mean is not None:
                f0_by_emotion.setdefault(s.emotion, []).append(m.f0_mean)
            if m.research_valence is not None:
                valence_by_emotion.setdefault(s.emotion, []).append(m.research_valence)
        f0_mean_by_emotion = {e: float(np.mean(v)) for e, v in f0_by_emotion.items()}
        valence_mean_by_emotion = {e: float(np.mean(v)) for e, v in valence_by_emotion.items()}

        f0_span = None
        if f0_mean_by_emotion:
            f0_span = max(f0_mean_by_emotion.values()) - min(f0_mean_by_emotion.values())

        by_sentiment: dict[str, list[float]] = {"positive": [], "neutral": [], "negative": []}
        for emotion, v in valence_mean_by_emotion.items():
            by_sentiment[D1_EMOTION_SENTIMENT[emotion].value].append(v)
        ordering_holds = None
        if all(by_sentiment.values()):
            pos, neu, neg = (
                float(np.mean(by_sentiment[k])) for k in ("positive", "neutral", "negative")
            )
            ordering_holds = bool(pos > neu > neg)

        return {
            "n_clips": len(group),
            "f0_mean_by_emotion": f0_mean_by_emotion,
            "f0_span_hz": f0_span,
            "valence_mean_by_emotion": valence_mean_by_emotion,
            "valence_ordering_matches_intended_sentiment": ordering_holds,
            "recoverability_cv": recoverability_cv(group, measurements),
        }

    def text_condition_summary(text_condition: str, has_description: bool) -> dict:
        """The harder, more important cut: does `description` work even on
        NEUTRAL text (isolating prosody from wording), not just congruent
        text where the words themselves already hint at the emotion?"""
        group = [
            s
            for s in specs
            if s.text_condition == text_condition and s.has_description == has_description
        ]
        valences = [
            measurements[s.clip_id].research_valence
            for s in group
            if measurements[s.clip_id].research_valence is not None
        ]
        f0s = [
            measurements[s.clip_id].f0_mean
            for s in group
            if measurements[s.clip_id].f0_mean is not None
        ]
        return {
            "n_clips": len(group),
            "f0_span_hz": (max(f0s) - min(f0s)) if len(f0s) >= 2 else None,
            "valence_range": (max(valences) - min(valences)) if len(valences) >= 2 else None,
        }

    return {
        "with_description": group_summary(True),
        "without_description": group_summary(False),
        "neutral_text_with_description": text_condition_summary("neutral", True),
        "congruent_text_with_description": text_condition_summary("congruent", True),
    }


def main() -> None:
    specs = build_grid()
    logger.info("generating %d Hume clips...", len(specs))
    asyncio.run(generate_all(specs))

    logger.info("loading acoustic solutions...")
    measurements = measure_all(specs)

    summary = summarize(specs, measurements)
    result = {
        "n_clips": len(specs),
        "voice": VOICE_NAME,
        "descriptions_used": DESCRIPTIONS,
        "summary": summary,
        "per_clip": [{**asdict(spec), **asdict(measurements[spec.clip_id])} for spec in specs],
        "interpretation": (
            "Enlarged falsification test of D1's Cartesia finding (results/d1_emotion_probe.json), "
            "at D1's own grid scale (40 clips, same carrier texts, 12 carrier groups) so the "
            "recoverability classifier is honest rather than descriptive-only. "
            "summary.with_description.recoverability_cv is the number directly comparable to "
            "D1's 0.175 (vs 0.200 chance); summary.without_description is the control. "
            "summary.neutral_text_with_description isolates the harder case -- does `description` "
            "work even when the words carry no emotional content, which is what this project's "
            "gold-label rule (prosody_sentiment, not text_sentiment) actually requires."
        ),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info(
        "done: %d clips, with-desc recoverability=%s, without-desc=%s -> %s",
        len(specs),
        summary["with_description"]["recoverability_cv"]["accuracy"],
        summary["without_description"]["recoverability_cv"]["accuracy"],
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()
