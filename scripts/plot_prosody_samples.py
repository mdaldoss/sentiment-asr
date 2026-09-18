#!/usr/bin/env python3
"""results/prosody_samples.json: VAD (valence/arousal/dominance) and F0
extracted from a fixed, deterministic sample of 20 real clips -- 12 from
CREMA-D (the public dataset, 2 per emotion tag) and 8 from E3 (the user's
own recordings, 4 per take, spanning all 3 sentiments) -- so the actual
extracted prosody/emotion signal can be inspected directly, not just its
downstream classification accuracy.

VAD comes from the research backend (audeering/wav2vec2-large-robust-12-ft-
emotion-msp-dim, Wagner et al. 2023, arXiv:2203.07378) -- the model that
actually regresses valence/arousal/dominance end-to-end, as opposed to the
permissive backend's WavLM-embedding + our own probe. F0 (parselmouth) is
included as a second, model-free prosody signal for comparison.

No API key needed -- reads committed CREMA-D cache + E3 recordings, uses
the audeering model (downloads/caches from Hugging Face on first run).

Usage:
    uv run python scripts/plot_prosody_samples.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.audio import load_clip  # noqa: E402
from ssa.encoders.audeering import AudeeringVADEncoder  # noqa: E402
from ssa.voicehealth import pitch_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "prosody_samples.json"

CREMAD_MANIFEST = REPO_ROOT / "data" / "cremad" / "manifest.csv"
E3_MANIFESTS = {
    "e3a": REPO_ROOT / "data" / "recorded0" / "manifest_speaker1.csv",
    "e3b": REPO_ROOT / "data" / "recorded" / "manifest_speaker1.csv",
}

CREMAD_EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
SEED = 0


def select_cremad_sample(n_per_emotion: int = 2, seed: int = SEED) -> pd.DataFrame:
    """2 clips per emotion tag, distinct speakers where possible, seeded
    for reproducibility -- 12 clips spanning all 6 CREMA-D emotions."""
    if not CREMAD_MANIFEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(CREMAD_MANIFEST, dtype={"speaker_id": str})
    rng = np.random.default_rng(seed)
    rows = []
    for emotion in CREMAD_EMOTIONS:
        subset = df[df["emotion_tag"] == emotion]
        n = min(n_per_emotion, len(subset))
        chosen = subset.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        rows.append(chosen)
    sample = pd.concat(rows, ignore_index=True)
    sample["source"] = "CREMA-D (public)"
    sample["label"] = sample["emotion_tag"]
    return sample[["clip_id", "path", "source", "label", "prosody_sentiment"]]


def select_e3_sample(n_per_take: int = 4, seed: int = SEED) -> pd.DataFrame:
    """4 clips per take, one per sentiment where possible (extra clips
    fill from whichever sentiment has the most), non-whisper only,
    seeded -- 8 clips spanning positive/neutral/negative from both takes."""
    frames = []
    rng = np.random.default_rng(seed)
    for take_label, path in E3_MANIFESTS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[~df["clip_id"].str.contains("whisper")]
        picks = []
        for sentiment in ("positive", "neutral", "negative"):
            subset = df[df["prosody_sentiment"] == sentiment]
            if len(subset):
                picks.append(subset.sample(n=1, random_state=int(rng.integers(0, 1_000_000))))
        chosen = pd.concat(picks, ignore_index=True) if picks else df.iloc[:0]
        remaining = n_per_take - len(chosen)
        if remaining > 0:
            leftover = df[~df["clip_id"].isin(chosen["clip_id"])]
            if len(leftover):
                chosen = pd.concat(
                    [chosen, leftover.sample(n=min(remaining, len(leftover)), random_state=seed)],
                    ignore_index=True,
                )
        chosen["source"] = f"E3 (mine, {take_label})"
        chosen["label"] = chosen["prosody_sentiment"]
        frames.append(chosen[["clip_id", "path", "source", "label", "prosody_sentiment"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract(sample: pd.DataFrame, encoder: AudeeringVADEncoder) -> list[dict]:
    rows = []
    for row in sample.itertuples(index=False):
        clip = load_clip(REPO_ROOT / row.path, clip_id=row.clip_id)
        vad = encoder.predict_vad(clip)
        pstats = pitch_stats(clip)
        rows.append(
            {
                "clip_id": row.clip_id,
                "source": row.source,
                "label": row.label,
                "prosody_sentiment": row.prosody_sentiment,
                "valence": vad.valence,
                "arousal": vad.arousal,
                "dominance": vad.dominance,
                "f0_mean": pstats.f0_mean,
                "f0_std": pstats.f0_std,
            }
        )
        logger.info(
            "%s (%s/%s): valence=%.3f arousal=%.3f f0=%s",
            row.clip_id,
            row.source,
            row.label,
            vad.valence,
            vad.arousal,
            f"{pstats.f0_mean:.0f}Hz" if pstats.f0_mean else "n/a",
        )
    return rows


def main() -> None:
    cremad_sample = select_cremad_sample()
    e3_sample = select_e3_sample()
    sample = pd.concat([cremad_sample, e3_sample], ignore_index=True)
    if sample.empty:
        raise RuntimeError(
            "no samples found -- need data/cremad/ (`make data`) and/or E3 recordings"
        )

    logger.info("loading audeering VAD encoder...")
    encoder = AudeeringVADEncoder()
    rows = extract(sample, encoder)

    result = {
        "n_samples": len(rows),
        "n_cremad": int((sample["source"] == "CREMA-D (public)").sum()),
        "n_e3": int(sample["source"].str.startswith("E3").sum()),
        "model": (
            "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim "
            "(Wagner et al. 2023, arXiv:2203.07378)"
        ),
        "samples": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("wrote %d samples -> %s", len(rows), OUT_PATH)


if __name__ == "__main__":
    main()
