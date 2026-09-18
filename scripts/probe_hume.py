#!/usr/bin/env python3
"""Falsification test for D1's Cartesia finding: does Hume Octave render
the same 5 emotions audibly on the SAME neutral text D1 used, via its
`description` (acting-instructions) field -- documented to control
delivery independently of the transcript, which is exactly the capability
Cartesia's docs say it lacks?

Grid: 5 emotions (happy/sad/angry/calm/frustrated) x description
(on/off) = 10 clips, one fixed voice, D1's own short neutral carrier text
(`ssa.carriers.D1_NEUTRAL_TEXT["short"]`) so the words are held constant --
same design principle as D1's neutral-text condition. Only one carrier
text is used (unlike D1's 8-cell grid), so this is descriptive only, no
leave-one-carrier-out classifier: not enough carrier diversity to fit one
honestly.

Measured with D1's same instruments (F0 via ssa.voicehealth.pitch_stats,
the research VAD model) for direct comparability with D1's numbers.

**Requires HUME_API_KEY** (`pip install hume`). Time-boxed by design (10
API calls): either outcome is a real result -- Hume clearing chance-level
differentiation would mean synthetic emotional TTS is achievable with the
right vendor; also failing would mean the D1 finding generalises beyond
Cartesia specifically.

Usage:
    uv run python scripts/probe_hume.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from ssa.audio import load_clip  # noqa: E402
from ssa.carriers import D1_EMOTION_SENTIMENT, D1_NEUTRAL_TEXT, EMOTIONS  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.types import Sentiment  # noqa: E402
from ssa.voicehealth import pitch_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "data" / "probes" / "hume"
RESULT_PATH = REPO_ROOT / "results" / "hume_probe.json"

VOICE_NAME = "Ava Song"  # fixed across all clips, so only `description` varies
CARRIER_TEXT = D1_NEUTRAL_TEXT["short"]  # same words D1 used for its neutral condition

# Short, "precise" emotion descriptions per Hume's own best-practice guidance
# (<=100 chars; concrete emotion words, not broad categories).
DESCRIPTIONS: dict[str, str] = {
    "happy": "happy, upbeat, smiling tone",
    "sad": "sad, downcast, weary tone",
    "angry": "angry, sharp, irritated tone",
    "calm": "calm, relaxed, gentle tone",
    "frustrated": "frustrated, tense, exasperated tone",
}


@dataclass(frozen=True, slots=True)
class HumeClipSpec:
    clip_id: str
    emotion: str
    has_description: bool


def build_grid() -> list[HumeClipSpec]:
    specs = []
    for emotion in EMOTIONS:
        for has_description in (True, False):
            suffix = "desc" if has_description else "nodesc"
            specs.append(
                HumeClipSpec(
                    clip_id=f"hume_{emotion}_{suffix}",
                    emotion=emotion,
                    has_description=has_description,
                )
            )
    return specs


async def _generate_one(client, spec: HumeClipSpec) -> None:
    from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName

    path = OUT_DIR / f"{spec.clip_id}.wav"
    if path.exists():
        return
    kwargs = dict(
        text=CARRIER_TEXT,
        voice=PostedUtteranceVoiceWithName(name=VOICE_NAME, provider="HUME_AI"),
        speed=1.0,
    )
    if spec.has_description:
        kwargs["description"] = DESCRIPTIONS[spec.emotion]
    utterance = PostedUtterance(**kwargs)
    result = await client.tts.synthesize_json(utterances=[utterance], format={"type": "wav"})
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


@dataclass(frozen=True, slots=True)
class Measurement:
    clip_id: str
    f0_mean: float | None
    f0_std: float | None
    research_valence: float | None
    research_arousal: float | None
    permissive_predicted_sentiment: str | None
    permissive_valence_proxy: float | None


def measure_all(specs: list[HumeClipSpec]) -> dict[str, Measurement]:
    acoustic_permissive = AcousticSolution(backend="permissive")
    acoustic_research = AcousticSolution(backend="research")
    out = {}
    for spec in specs:
        clip = load_clip(OUT_DIR / f"{spec.clip_id}.wav", clip_id=spec.clip_id)
        pstats = pitch_stats(clip)
        perm_pred = acoustic_permissive.predict(clip)
        res_pred = acoustic_research.predict(clip)
        out[spec.clip_id] = Measurement(
            clip_id=spec.clip_id,
            f0_mean=pstats.f0_mean,
            f0_std=pstats.f0_std,
            research_valence=res_pred.vad.valence if res_pred.vad is not None else None,
            research_arousal=res_pred.vad.arousal if res_pred.vad is not None else None,
            permissive_predicted_sentiment=perm_pred.sentiment.value,
            permissive_valence_proxy=(
                perm_pred.probs[Sentiment.POSITIVE] - perm_pred.probs[Sentiment.NEGATIVE]
            ),
        )
    return out


def summarize(specs: list[HumeClipSpec], measurements: dict[str, Measurement]) -> dict:
    def group_summary(has_description: bool) -> dict:
        group = [s for s in specs if s.has_description == has_description]
        f0_by_emotion = {
            s.emotion: measurements[s.clip_id].f0_mean
            for s in group
            if measurements[s.clip_id].f0_mean is not None
        }
        valence_by_emotion = {
            s.emotion: measurements[s.clip_id].research_valence
            for s in group
            if measurements[s.clip_id].research_valence is not None
        }
        f0_span = None
        if f0_by_emotion:
            f0_span = max(f0_by_emotion.values()) - min(f0_by_emotion.values())

        # positive > neutral > negative bucket-mean check, same as D1's
        by_sentiment: dict[str, list[float]] = {"positive": [], "neutral": [], "negative": []}
        for emotion, v in valence_by_emotion.items():
            by_sentiment[D1_EMOTION_SENTIMENT[emotion].value].append(v)
        ordering_holds = None
        if all(by_sentiment.values()):
            pos, neu, neg = (
                float(np.mean(by_sentiment[k])) for k in ("positive", "neutral", "negative")
            )
            ordering_holds = bool(pos > neu > neg)

        return {
            "f0_by_emotion": f0_by_emotion,
            "f0_span_hz": f0_span,
            "valence_by_emotion": valence_by_emotion,
            "valence_ordering_matches_intended_sentiment": ordering_holds,
        }

    return {
        "with_description": group_summary(True),
        "without_description": group_summary(False),
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
        "carrier_text": CARRIER_TEXT,
        "voice": VOICE_NAME,
        "descriptions_used": DESCRIPTIONS,
        "summary": summary,
        "per_clip": [{**asdict(spec), **asdict(measurements[spec.clip_id])} for spec in specs],
        "interpretation": (
            "Falsification test of D1's Cartesia finding (results/d1_emotion_probe.json), "
            "using a vendor whose docs claim delivery control independent of the transcript. "
            "Only 1 carrier text -> descriptive only, no leave-one-carrier-out classifier "
            "(not enough carrier diversity to fit one honestly). Compare "
            "summary.with_description's f0_span_hz and valence_ordering against D1's "
            "f0_span_across_emotions_hz (~8 Hz) and scrambled ordering: if Hume's spread is "
            "meaningfully larger and/or the ordering holds, that is real differentiation "
            "Cartesia didn't show; if not, the D1 finding generalises beyond one vendor."
        ),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
