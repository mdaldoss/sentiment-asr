#!/usr/bin/env python3
"""E2 (default) / E4 (--augment): synthetic incongruence and augmentation
sets, generated via Cartesia.

E2 is a 3 (text-sentiment) x 3 (prosody-sentiment) x 5 (carrier) x 2 (voice)
= 90-clip crossed design. The 30 diagonal cells (text_sentiment ==
prosody_sentiment) are congruent; the 60 off-diagonal cells are
incongruent -- this is what makes the Prosody Sensitivity Index computable.

Ground-truth prosody_sentiment comes from TTS *intent* (the requested
emotion tag), not from re-verifying with any classifier -- see T8's
findings (results/d0_emotion_space.json): the audeering VAD model doesn't
reliably decode valence from Cartesia's synthetic prosody even though the
tags demonstrably change the audio (duration/RMS confirmed). Filtering
tags by that model's opinion would have been circular and untrustworthy.
Uses only 3 of Cartesia's own documented "best results" tags (neutral,
content, angry) for the same reason -- maximising the chance of genuine
acoustic differentiation rather than picking arbitrarily from all 58.

E4 (--augment) draws more broadly from ssa.mapping.CARTESIA_MAP for
diversity, sampled balanced across sentiment despite the vocabulary's
inherent imbalance (24 negative / 15 positive / 5 neutral labelled tags).

**Requires CARTESIA_API_KEY.** Evaluation itself never depends on this
script -- generated clips and their manifest are committed once produced.

Usage:
    uv run python scripts/gen_synthetic.py            # E2, ~90 clips
    uv run python scripts/gen_synthetic.py --augment --n 300   # E4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from ssa.carriers import CARRIERS  # noqa: E402
from ssa.manifest import add_congruence, write_manifest  # noqa: E402
from ssa.mapping import labelled_tags, map_emotion  # noqa: E402
from ssa.tts import VOICE_IDS, generate_clip, get_client  # noqa: E402
from ssa.types import Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "data" / "synthetic"

# Cartesia's own documentation names only 6 tags as giving best results;
# these 3 span the sentiment axis and were confirmed (results/d0_emotion_space.json)
# to produce acoustically distinct output (duration/RMS) even though a
# real-speech-trained VAD model can't cleanly decode their valence.
E2_PROSODY_TAGS: dict[Sentiment, str] = {
    Sentiment.POSITIVE: "content",
    Sentiment.NEUTRAL: "neutral",
    Sentiment.NEGATIVE: "angry",
}

E2_VOICES = [VOICE_IDS["skylar"], VOICE_IDS["daniel"]]


def build_e2_manifest() -> pd.DataFrame:
    rows = []
    for text_sentiment, sentences in CARRIERS.items():
        for carrier_idx, text in enumerate(sentences):
            for prosody_sentiment, tag in E2_PROSODY_TAGS.items():
                for voice_name, voice_id in zip(["skylar", "daniel"], E2_VOICES, strict=True):
                    clip_id = (
                        f"e2_{text_sentiment.value}_{carrier_idx}_"
                        f"{prosody_sentiment.value}_{voice_name}"
                    )
                    rows.append(
                        {
                            "clip_id": clip_id,
                            "path": str((OUT_DIR / f"{clip_id}.wav").relative_to(REPO_ROOT)),
                            "speaker_id": f"cartesia_{voice_name}",
                            "source": "synthetic",
                            "text": text,
                            "text_sentiment": text_sentiment.value,
                            "prosody_sentiment": prosody_sentiment.value,
                            "emotion_tag": tag,
                            "voice_id": voice_id,
                            "split": "",
                        }
                    )
    df = pd.DataFrame(rows)
    return add_congruence(df)


def generate_from_manifest(df: pd.DataFrame, client) -> None:
    for row in df.itertuples(index=False):
        out_path = REPO_ROOT / row.path
        if out_path.exists():
            continue
        generate_clip(
            client, text=row.text, emotion=row.emotion_tag, voice_id=row.voice_id, out_path=out_path
        )
        logger.info("generated %s", row.clip_id)


def build_e4_manifest(n: int, seed: int = 0) -> pd.DataFrame:
    """Balanced-per-sentiment augmentation set, drawing from the full
    labelled Cartesia vocabulary for diversity."""
    import numpy as np

    rng = np.random.default_rng(seed)
    tags_by_sentiment: dict[Sentiment, list[str]] = {s: [] for s in Sentiment}
    for tag in labelled_tags("cartesia"):
        tags_by_sentiment[map_emotion(tag, "cartesia")].append(tag)

    n_per_sentiment = n // 3
    rows = []
    for sentiment, tags in tags_by_sentiment.items():
        sentences = CARRIERS[sentiment]
        for i in range(n_per_sentiment):
            tag = tags[rng.integers(0, len(tags))]
            text = sentences[rng.integers(0, len(sentences))]
            voice_name = ["skylar", "daniel"][rng.integers(0, 2)]
            voice_id = VOICE_IDS[voice_name]
            clip_id = f"e4_{sentiment.value}_{i}_{tag}_{voice_name}"
            rows.append(
                {
                    "clip_id": clip_id,
                    "path": str((OUT_DIR / "augment" / f"{clip_id}.wav").relative_to(REPO_ROOT)),
                    "speaker_id": f"cartesia_{voice_name}",
                    "source": "synthetic",
                    "text": text,
                    "text_sentiment": sentiment.value,  # carrier's own lexical sentiment
                    "prosody_sentiment": sentiment.value,  # congruent by construction
                    "emotion_tag": tag,
                    "voice_id": voice_id,
                    "split": "train",  # augmentation only ever joins the training split
                }
            )
    df = pd.DataFrame(rows)
    return add_congruence(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augment", action="store_true", help="build E4 instead of E2")
    parser.add_argument("--n", type=int, default=300, help="E4 clip count (default 300)")
    args = parser.parse_args()

    client = get_client()

    if args.augment:
        df = build_e4_manifest(args.n)
        manifest_path = OUT_DIR / "augment" / "manifest.csv"
    else:
        df = build_e2_manifest()
        manifest_path = OUT_DIR / "manifest.csv"

    generate_from_manifest(df, client)
    write_manifest(df, manifest_path)
    n_incongruent = int((~df["is_congruent"]).sum())
    logger.info("done: %d clips (%d incongruent) -> %s", len(df), n_incongruent, manifest_path)


if __name__ == "__main__":
    main()
