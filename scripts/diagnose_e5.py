#!/usr/bin/env python3
"""Is E5's modest PSI the audio's fault or the models'?

E5 showed the permissive backend following delivery on 68% of contradictory
clips and the research backend following the *words* on 79%. Two very
different explanations produce a middling number like 0.682:

  (a) **the models are the limit** -- Hume rendered the requested delivery
      clearly, and the acoustic models simply cannot read it; or
  (b) **the audio is the limit** -- Hume softened or ignored the delivery
      instruction when it contradicted the words, so there is less prosody
      in those clips for any model to find.

Neither the PSI numbers nor UAR can separate these, because both are
model-mediated. This script removes the models from the question: it asks
whether the *requested delivery* is recoverable from raw acoustic
descriptors (F0, loudness, HNR, jitter, shimmer, RMS, speech rate) by a
plain logistic regression, using the exact instruments and method already
applied to Cartesia (D1) and to human speech (E3) in
`scripts/eval_e3.py` -- so the numbers are directly comparable to those.

The decisive cut is **congruent vs incongruent**. If Hume complied equally
in both cases, recoverability should be similar; if it quietly deferred to
the transcript when the two disagreed, recoverability will collapse on the
incongruent clips specifically. That comparison is what makes this a test
rather than a description.

Leave-one-carrier-out CV throughout: the classifier is never tested on a
sentence it was trained on, so it cannot win by memorising a carrier.

No API key needed.

Usage:
    uv run python scripts/diagnose_e5.py
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

from scripts.eval_e3 import f0_rank_consistency, recoverability_cv  # noqa: E402
from ssa.audio import load_clip  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.paralinguistic import extract_features  # noqa: E402
from ssa.voicehealth import pitch_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH = REPO_ROOT / "data" / "hume_e5" / "manifest.csv"
RESULT_PATH = REPO_ROOT / "results" / "e5_diagnostic.json"


def measure(manifest: pd.DataFrame) -> pd.DataFrame:
    """Raw acoustic descriptors per clip -- no learned model involved."""
    rows = []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        clip = load_clip(REPO_ROOT / row.path, clip_id=row.clip_id)
        pstats = pitch_stats(clip)
        feats = extract_features(clip)
        rows.append(
            {
                "clip_id": row.clip_id,
                "carrier": row.text,
                "intended": row.prosody_sentiment,  # the requested delivery
                "text_sentiment": row.text_sentiment,
                "is_congruent": bool(row.is_congruent),
                "voice_id": row.voice_id,
                "f0_mean": pstats.f0_mean,
                "f0_std": pstats.f0_std,
                "f0_range": pstats.f0_range,
                "egemaps_loudness": feats.loudness_mean,
                "egemaps_hnr": feats.hnr_mean,
                "egemaps_jitter": feats.jitter_mean,
                "egemaps_shimmer_db": feats.shimmer_db_mean,
                "rms": float(np.sqrt(np.mean(clip.samples.astype(np.float64) ** 2))),
                "speech_rate_cps": len(row.text) / clip.duration_s if clip.duration_s > 0 else 0.0,
            }
        )
        if i % 30 == 0:
            logger.info("measured %d/%d", i, len(manifest))
    return pd.DataFrame(rows)


def summarise_cut(df: pd.DataFrame, label: str) -> dict:
    """Recoverability + F0 separation for one subset of clips."""
    if len(df) < 6 or df["intended"].nunique() < 2:
        return {"label": label, "n": len(df), "note": "too few clips/classes for CV"}
    cv = recoverability_cv(df)
    f0_by = df.groupby("intended")["f0_mean"].mean()
    return {
        "label": label,
        "n": len(df),
        "recoverability_cv": cv,
        "f0_span_hz": float(f0_by.max() - f0_by.min()),
        "mean_f0_by_intended": f0_by.round(2).to_dict(),
        "f0_rank_consistency": f0_rank_consistency(df),
    }


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found -- run `make gen-hume-e5` first")

    manifest = load_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)
    logger.info("measuring %d E5 clips (no models, raw descriptors only) ...", len(manifest))
    df = measure(manifest)
    df.to_csv(REPO_ROOT / "results" / "e5_measurements.csv", index=False)

    congruent = df[df["is_congruent"]]
    incongruent = df[~df["is_congruent"]]

    result = {
        "dataset": "e5_hume",
        "n_clips": len(df),
        "chance": 1 / 3,
        "all_clips": summarise_cut(df, "all"),
        "congruent_only": summarise_cut(congruent, "congruent (delivery agrees with words)"),
        "incongruent_only": summarise_cut(incongruent, "incongruent (delivery contradicts words)"),
        "per_voice": {str(v): summarise_cut(g, f"voice={v}") for v, g in df.groupby("voice_id")},
        "reference_points": {
            "cartesia_d1_recoverability": 0.175,
            "cartesia_d1_chance": 0.2,
            "human_e3a_recoverability": 0.667,
            "human_e3b_recoverability": 0.741,
            "human_e3_chance": 1 / 3,
        },
    }

    cong = result["congruent_only"].get("recoverability_cv", {}).get("accuracy")
    inco = result["incongruent_only"].get("recoverability_cv", {}).get("accuracy")
    result["interpretation"] = (
        "Separates the two explanations for E5's middling PSI. These numbers involve no "
        "trained emotion model at all -- just raw acoustic descriptors and a logistic "
        "regression -- so they measure whether the requested delivery is PRESENT in the "
        "audio, independently of whether our models can use it. Compare against the same "
        "instruments on Cartesia (0.175 vs 0.200 chance: delivery absent) and on real human "
        "speech (0.667/0.741 vs 0.333 chance: delivery clearly present). The congruent-vs-"
        "incongruent contrast is the key cut: a large drop on the incongruent clips would "
        "mean Hume softened the delivery when it contradicted the transcript, making the "
        "audio the limiting factor; similar values would mean the delivery is equally "
        "present in both and the models are the limiting factor."
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", RESULT_PATH)
    logger.info(
        "recoverability (chance 0.333): all=%s congruent=%s incongruent=%s",
        result["all_clips"].get("recoverability_cv", {}).get("accuracy"),
        cong,
        inco,
    )


if __name__ == "__main__":
    main()
