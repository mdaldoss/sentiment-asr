#!/usr/bin/env python3
"""E5: the incongruence dataset, built on Hume Octave.

This is the set E2 was supposed to be and couldn't: **clips where the words
say one thing and the delivery says another**, which is the only condition
that can tell a model that hears prosody apart from one that reads the
transcript. E2 failed at it because Cartesia's own docs say its emotion
parameter only works when the requested emotion already matches the
transcript -- exactly the case that is useless here -- and D1 measured that
failure (intended emotion recoverable at 0.175 vs 0.200 chance).

Hume Octave's `description` field sets delivery **independently of the
text**, and the 40-clip probe in `scripts/probe_hume.py` confirmed it works
on this content (0.5 vs 0.2 chance with `description`, exactly chance
without). E5 spends that capability on the dataset the project actually
needs.

Design -- a full crossing of lexical content against delivery:

    15 carrier texts (5 clearly positive, 5 neutral, 5 clearly negative,
    reused verbatim from `ssa.carriers.CARRIERS`, so lexical polarity is
    unambiguous to a human reader and to Solution A's text classifier)
      x  3 prosody targets (positive / neutral / negative, via `description`)
      x  N voices
    = 45 clips per voice, of which **30 are incongruent** (2 of every 3).

`text_sentiment` comes from which carrier list the sentence belongs to;
`prosody_sentiment` -- the gold label for every evaluation in this repo --
comes from the requested delivery. They disagree on two thirds of the set
by construction.

**Voice choice is a methodological decision, not a cosmetic one.** Hume
ships many character voices ("Sad Old British Man", "Tough Guy") whose
baked-in affect would confound the one variable under test. Only voices
with no emotional loading in their identity are used here, so `description`
is the sole source of emotional variance.

**Requires HUME_API_KEY.** Resumable: already-generated clips are skipped,
so an interrupted or quota-limited run continues where it stopped, and a
partial set is reported as partial rather than silently treated as whole.

Usage:
    uv run python scripts/gen_hume_dataset.py            # default 2 voices
    uv run python scripts/gen_hume_dataset.py --voices 3
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from ssa.carriers import CARRIERS  # noqa: E402
from ssa.manifest import add_congruence, write_manifest  # noqa: E402
from ssa.types import Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "data" / "hume_e5"
MANIFEST_PATH = OUT_DIR / "manifest.csv"

# Same rate-limit handling as scripts/probe_hume.py, which learned it the
# hard way (an unhandled 429 killed a 40-clip run at clip 28).
REQUEST_SPACING_S = 3.0
MAX_RETRIES = 5
RETRY_BASE_DELAY_S = 10.0

# Deliberately emotion-neutral voice identities -- see module docstring.
# Ava Song is carried over from the probe so E5 overlaps with the finding
# that justified building it.
VOICES: tuple[str, ...] = ("Ava Song", "Colton Rivers", "Imani Carter")

# Prosody-only delivery instructions: none of them names or implies any
# subject matter, so the same description applies unchanged to a positive,
# neutral or negative sentence. That independence is the whole experiment.
DESCRIPTIONS: dict[Sentiment, str] = {
    Sentiment.POSITIVE: "cheerful, warm, smiling delivery; bright and lifted tone",
    Sentiment.NEUTRAL: "flat, matter-of-fact delivery; even and unemotional tone",
    Sentiment.NEGATIVE: "sad, heavy, downcast delivery; low and weary tone",
}


@dataclass(frozen=True, slots=True)
class E5Spec:
    clip_id: str
    voice: str
    text: str
    text_sentiment: Sentiment
    prosody_sentiment: Sentiment
    carrier_index: int

    @property
    def is_congruent(self) -> bool:
        return self.text_sentiment == self.prosody_sentiment


def _voice_slug(voice: str) -> str:
    return voice.lower().replace(" ", "_")


def build_grid(voices: tuple[str, ...] = VOICES) -> list[E5Spec]:
    """Full crossing of carrier text x delivery x voice."""
    specs: list[E5Spec] = []
    for voice, (text_sentiment, texts) in itertools.product(voices, CARRIERS.items()):
        for carrier_index, text in enumerate(texts):
            for prosody_sentiment in (
                Sentiment.POSITIVE,
                Sentiment.NEUTRAL,
                Sentiment.NEGATIVE,
            ):
                specs.append(
                    E5Spec(
                        clip_id=(
                            f"e5_{_voice_slug(voice)}_{text_sentiment.value}"
                            f"{carrier_index}_{prosody_sentiment.value}"
                        ),
                        voice=voice,
                        text=text,
                        text_sentiment=text_sentiment,
                        prosody_sentiment=prosody_sentiment,
                        carrier_index=carrier_index,
                    )
                )
    return specs


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()


async def _generate_one(client, spec: E5Spec) -> bool:
    """Returns True if a clip was generated, False if it already existed."""
    from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName

    path = OUT_DIR / f"{spec.clip_id}.wav"
    if path.exists():
        return False

    utterance = PostedUtterance(
        text=spec.text,
        voice=PostedUtteranceVoiceWithName(name=spec.voice, provider="HUME_AI"),
        description=DESCRIPTIONS[spec.prosody_sentiment],
        speed=1.0,
    )
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(result.generations[0].audio))
    logger.info("generated %s", spec.clip_id)
    return True


async def generate_all(specs: list[E5Spec]) -> None:
    import os

    from hume import AsyncHumeClient

    client = AsyncHumeClient(api_key=os.environ["HUME_API_KEY"])
    for i, spec in enumerate(specs, start=1):
        generated = await _generate_one(client, spec)
        if generated:
            await asyncio.sleep(REQUEST_SPACING_S)
        if i % 15 == 0:
            logger.info("progress: %d/%d", i, len(specs))


def build_manifest(specs: list[E5Spec]) -> pd.DataFrame:
    """Manifest rows for the clips that actually exist on disk. A clip the
    API never returned is left out and counted, never emitted as a row
    pointing at a missing file (CLAUDE.md: never silently drop clips --
    count and report them)."""
    rows = []
    for spec in specs:
        if not (OUT_DIR / f"{spec.clip_id}.wav").exists():
            continue
        rows.append(
            {
                "clip_id": spec.clip_id,
                "path": f"data/hume_e5/{spec.clip_id}.wav",
                # One pseudo-speaker per voice: keeps speaker-disjoint
                # splitting meaningful if E5 is ever used for training.
                "speaker_id": f"hume_{_voice_slug(spec.voice)}",
                "source": "hume",
                "text": spec.text,
                "text_sentiment": spec.text_sentiment.value,
                "prosody_sentiment": spec.prosody_sentiment.value,
                "emotion_tag": "",
                "voice_id": spec.voice,
                "split": "test",  # eval-only: a synthetic set never trains the shipped model
            }
        )
    return add_congruence(pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voices", type=int, default=2, help="how many voices from VOICES to use (default 2)"
    )
    args = parser.parse_args()

    voices = VOICES[: args.voices]
    specs = build_grid(voices)
    logger.info("E5 grid: %d clips across %d voice(s): %s", len(specs), len(voices), list(voices))

    asyncio.run(generate_all(specs))

    manifest = build_manifest(specs)
    missing = len(specs) - len(manifest)
    if missing:
        logger.warning(
            "%d/%d clips missing from disk -- manifest covers the rest", missing, len(specs)
        )

    write_manifest(manifest, MANIFEST_PATH)
    n_incongruent = int((~manifest["is_congruent"].astype(bool)).sum())
    logger.info(
        "E5 done: %d clips, %d incongruent (%.0f%%), %d speakers -> %s",
        len(manifest),
        n_incongruent,
        100 * n_incongruent / max(len(manifest), 1),
        manifest["speaker_id"].nunique(),
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()
