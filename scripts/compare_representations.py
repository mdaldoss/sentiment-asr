#!/usr/bin/env python3
"""Prosodic features vs WavLM embeddings, with the training domain held constant.

This settles a comparison this project previously got wrong. The E5
diagnostic fitted nine acoustic descriptors *inside* E5 (leave-one-carrier-out)
and scored 0.811; Solution D trained on CREMA-D and met E5 cold scored 0.278.
Those were then compared against each other, which compares two different
problems -- in-domain fitting versus cross-domain transfer -- and the
conclusion drawn ("simple features beat the deep models") did not follow.

The fair test is to give both representations the *same* deal. For each
dataset, fit the same classifier under the same leave-one-carrier-out
cross-validation on:

  prosodic   96 explicit descriptors (eGeMAPS + Praat contour)
  wavlm      the frozen encoder's pooled embedding
  combined   both concatenated

Grouping by carrier text means no model is ever tested on a sentence it
trained on, so nothing can win by memorising wording. Everything is fitted
within-dataset, so neither representation gets a domain advantage.

What each outcome would mean:

  prosodic >> wavlm   the information is in simple physics and the encoder
                      buries it -- feature engineering is the better path
  wavlm >> prosodic   the encoder represents prosody better than hand-built
                      descriptors, and Solution D's failure was the features
  roughly equal       the representation is not the bottleneck; the training
                      corpus is, which is what every cross-domain result here
                      already suggests

Note this is a *ceiling* measurement, not a deployable number: fitting
within the evaluation set is exactly what a shipped model cannot do. It
answers "is the signal reachable by this representation", not "how well
would it work in production".

Usage:
    uv run python scripts/compare_representations.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.audio import load_clip  # noqa: E402
from ssa.encoders.wavlm import WavLMEncoder  # noqa: E402
from ssa.eval.metrics import uar  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.prosodic_features import MemoizingExtractor  # noqa: E402
from ssa.types import Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULT_PATH = REPO_ROOT / "results" / "representation_comparison.json"

DATASETS: dict[str, list[Path]] = {
    "e3_both_takes": [
        REPO_ROOT / "data" / "recorded0" / "manifest_speaker1.csv",
        REPO_ROOT / "data" / "recorded" / "manifest_speaker1.csv",
    ],
    "e5_hume": [REPO_ROOT / "data" / "hume_e5" / "manifest.csv"],
}

SEED = 0


def load_dataset(paths: list[Path]) -> pd.DataFrame:
    frames = [load_manifest(p, repo_root=REPO_ROOT) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df[~df["clip_id"].str.contains("whisper")].reset_index(drop=True)


def build_matrices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Both representations for the same clips, in the same row order."""
    prosodic_extractor = MemoizingExtractor()
    wavlm = WavLMEncoder()

    prosodic, embeddings = [], []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        clip = load_clip(REPO_ROOT / row.path, clip_id=row.clip_id)
        prosodic.append(prosodic_extractor.embed(clip))
        embeddings.append(wavlm.embed(clip))
        if i % 30 == 0:
            logger.info("  featurised %d/%d", i, len(df))

    P = np.vstack(prosodic)
    W = np.vstack(embeddings)
    return {"prosodic": P, "wavlm": W, "combined": np.hstack([P, W])}


def loco_uar(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Leave-one-carrier-out UAR. Groups are carrier sentences, so a model
    is never scored on wording it was fitted on."""
    logo = LeaveOneGroupOut()
    preds = np.empty(len(y), dtype=object)
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(set(y[train_idx])) < 2:
            preds[test_idx] = y[train_idx][0]
            continue
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED),
                ),
            ]
        )
        pipeline.fit(X[train_idx], y[train_idx])
        preds[test_idx] = pipeline.predict(X[test_idx])

    y_true = [Sentiment(s) for s in y]
    y_pred = [Sentiment(s) for s in preds]
    n_classes = len({*y})
    return {
        "uar": uar(y_true, y_pred),
        "chance": 1.0 / n_classes,
        "n": len(y),
        "n_groups": len(set(groups)),
        "predicted_class_share": {str(c): float((preds == c).mean()) for c in sorted(set(preds))},
    }


def main() -> None:
    result: dict[str, object] = {"datasets": {}}

    for name, paths in DATASETS.items():
        if not all(p.exists() for p in paths):
            logger.warning("%s: manifest missing, skipping", name)
            continue
        df = load_dataset(paths)
        logger.info("%s: %d clips, featurising both representations ...", name, len(df))
        matrices = build_matrices(df)

        y = df["prosody_sentiment"].to_numpy()
        groups = df["text"].to_numpy()  # carrier sentence
        per_rep = {rep: loco_uar(X, y, groups) for rep, X in matrices.items()}
        result["datasets"][name] = {
            "n_clips": len(df),
            "n_carriers": len(set(groups)),
            "representations": per_rep,
            "dims": {rep: int(X.shape[1]) for rep, X in matrices.items()},
        }
        for rep, s in per_rep.items():
            logger.info(
                "  %-9s dim=%-5d UAR=%.3f (chance %.3f)",
                rep,
                matrices[rep].shape[1],
                s["uar"],
                s["chance"],
            )

    result["method"] = (
        "Logistic regression, leave-one-carrier-out, fitted within each dataset. Identical "
        "classifier, CV scheme and labels across representations -- only the input features "
        "differ, so the comparison isolates the representation."
    )
    result["interpretation"] = (
        "A ceiling measurement, not a deployable score: fitting inside the evaluation set is "
        "what a shipped model cannot do. It answers whether the prosodic signal is REACHABLE "
        "from each representation, which the earlier comparison between an in-domain fit "
        "(0.811) and a CREMA-D-trained transfer (0.278) could not, because those are "
        "different problems. If the representations land close together, the bottleneck is "
        "the training corpus rather than the features or the encoder."
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
