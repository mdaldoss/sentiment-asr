#!/usr/bin/env python3
"""Fit Solution D: explicit prosodic features -> a transparent classifier.

Trains on CREMA-D's speaker-disjoint train split and selects a classifier on
the validation split. Four candidates are fitted, spanning the transparency
/ flexibility range:

  logreg          linear, fully inspectable coefficients
  linear_svm      linear, max-margin (calibrated to get probabilities)
  svm_rbf         non-linear kernel, still no vocabulary
  hist_gbdt       histogram gradient boosting -- sklearn's LightGBM-family
                  implementation, which is why no new dependency is added

**Every candidate's validation score is recorded, not just the winner's.**
Reporting only the best of four fits, with no note that four were tried, is
how a tuning artefact gets presented as a modelling result -- and a field
replication study this project cites (arXiv:2508.02448) concluded that
hyperparameter search, rather than architecture, explains much of the
reported progress in speech emotion recognition. So the losers stay in the
output file.

`class_weight="balanced"` throughout: CREMA-D is 68% negative, and an
unweighted fit would score well on accuracy while barely predicting the two
minority classes. UAR is the selection criterion for the same reason
(CLAUDE.md rule 7).

Usage:
    uv run python scripts/train_prosodic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.embeddings import embeddings_matrix, extract_and_cache  # noqa: E402
from ssa.eval.metrics import macro_f1, uar  # noqa: E402
from ssa.prosodic_features import FEATURE_NAMES, ProsodicExtractor  # noqa: E402
from ssa.splits import assert_speaker_disjoint  # noqa: E402
from ssa.types import Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH = REPO_ROOT / "data" / "cremad" / "manifest.csv"
CACHE_PATH = REPO_ROOT / "data" / "cache" / "features_prosodic.npz"
MODEL_PATH = REPO_ROOT / "data" / "cache" / "prosodic_model.joblib"
REPORT_PATH = REPO_ROOT / "results" / "prosodic_training.json"

SEED = 0
# Same guard as ssa/train.py: SOTA speaker-independent is <0.90, so anything
# above it means leakage, not success.
SUSPICIOUS_UAR = 0.90


def candidates() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=SEED), cv=3
        ),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
        "hist_gbdt": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED, early_stopping=True
        ),
    }


def build_pipeline(clf: object) -> Pipeline:
    """Impute -> standardise -> classify.

    Imputation is not cosmetic: the extractor emits NaN for a clip whose
    pitch or intensity analysis fails rather than dropping it silently, so
    something downstream has to absorb that.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def split_xy(
    manifest: pd.DataFrame, split: str, cache: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[Sentiment], list[str]]:
    sub = manifest[manifest["split"] == split]
    if len(sub) == 0:
        raise ValueError(f"no rows with split={split!r}")
    X, ids = embeddings_matrix(sub, cache)
    y = [Sentiment(s) for s in sub["prosody_sentiment"]]
    return X, y, ids


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found -- run `make data` first")

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"clip_id": str, "speaker_id": str})
    assert_speaker_disjoint(manifest)  # CLAUDE.md rule 1, checked every run

    logger.info("extracting prosodic features (resumable cache at %s) ...", CACHE_PATH.name)
    cache = extract_and_cache(manifest, ProsodicExtractor(), CACHE_PATH, repo_root=REPO_ROOT)

    X_train, y_train, _ = split_xy(manifest, "train", cache)
    X_val, y_val, _ = split_xy(manifest, "val", cache)
    y_train_str = [s.value for s in y_train]
    logger.info("train %s, val %s, %d features", X_train.shape, X_val.shape, len(FEATURE_NAMES))

    scores: dict[str, dict[str, float]] = {}
    best_name, best_pipeline, best_uar = "", None, -1.0

    for name, clf in candidates().items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train_str)
        pred = [Sentiment(s) for s in pipeline.predict(X_val)]
        val_uar, val_f1 = uar(y_val, pred), macro_f1(y_val, pred)
        scores[name] = {"val_uar": val_uar, "val_macro_f1": val_f1}
        logger.info("%-12s val UAR=%.3f macroF1=%.3f", name, val_uar, val_f1)
        if val_uar > best_uar:
            best_name, best_pipeline, best_uar = name, pipeline, val_uar

    if best_uar > SUSPICIOUS_UAR:
        logger.warning(
            "val UAR=%.3f exceeds the %.2f plausibility ceiling -- check for leakage "
            "before trusting this",
            best_uar,
            SUSPICIOUS_UAR,
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_type": best_name,
            "feature_names": list(FEATURE_NAMES),
        },
        MODEL_PATH,
    )
    logger.info("saved %s (val UAR=%.3f) -> %s", best_name, best_uar, MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "selected": best_name,
                "selection_criterion": "val_uar",
                "n_features": len(FEATURE_NAMES),
                "n_train": int(X_train.shape[0]),
                "n_val": int(X_val.shape[0]),
                "all_candidates": scores,
                "note": (
                    "All four candidates are listed, not only the selected one. Selection on "
                    "a held-out validation split of speaker-disjoint CREMA-D; the spread "
                    "between candidates indicates how much of any result is the classifier "
                    "rather than the features."
                ),
            },
            indent=2,
        )
    )
    logger.info("wrote %s", REPORT_PATH)


if __name__ == "__main__":
    main()
