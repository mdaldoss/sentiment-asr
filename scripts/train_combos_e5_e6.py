#!/usr/bin/env python3
"""Train the permissive backend on E5 and E6, separately and together.

Asked directly: fit the permissive acoustic probe on the Zurich recordings
(E6) and the Hume set (E5), each alone and both combined, then score every
resulting model everywhere.

Five training sets:

  cremad        5,235 clips, 64 actors -- the shipped baseline
  e6              100 clips,  5 speakers
  e5               45 clips,  1 synthetic voice
  e5_e6           145 clips,  6 speakers
  cremad_e5_e6  5,380 clips        -- does adding the small sets to the big one help

**E5 can only contribute one voice.** It has exactly two (Ava Song, Colton
Rivers), so training on it and keeping an honest held-out E5 score means
training on one and testing on the other. Ava trains; Colton is held out of
every training set here and is the only clean E5 evaluation. Ava's own clips
are scored too, clearly marked in-domain for the models that trained on
them, because the gap between the two is the whole point.

**This deliberately breaks a rule the project set for itself**, and that is
worth stating rather than burying. `DESIGN.md` designates E5 eval-only: it
is the instrument that measures prosody sensitivity under contradiction,
and CLAUDE.md rule 8 forbids presenting within-corpus numbers as
generalisation. A model trained on E5 can no longer use E5 as a clean PSI
instrument. The request is reasonable and the experiment is informative, so
it runs -- but every E5 number for an E5-trained model is flagged
in-domain, and the cross-corpus columns are the ones to read.

**E3 is excluded for anything trained on E6.** One E6 training speaker also
recorded E3 and carries E3's `speaker1` id, so scoring those models on E3
would put the same human on both sides.

The probe is the permissive backend's primary path, replicated here rather
than called: `StandardScaler` then `LogisticRegression(max_iter=2000,
class_weight="balanced")` over frozen WavLM embeddings. `train_permissive`
additionally falls back to a small MLP when validation UAR is weak and needs
a `split == "val"` block to select on; neither fits a design where some
training sets are 45 clips and the interesting comparison is the training
data, not the classifier. Holding the classifier fixed is what makes the
five rows comparable.

Usage:
    uv run python scripts/train_combos_e5_e6.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_zurich import bootstrap_ci  # noqa: E402
from ssa.combo_manifests import stratified_subset  # noqa: E402
from ssa.embeddings import embeddings_matrix, extract_and_cache  # noqa: E402
from ssa.encoders.wavlm import WavLMEncoder  # noqa: E402
from ssa.eval.metrics import macro_f1, psi_contested, psi_strict, uar  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.types import Prediction, Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "combos_e5_e6.json"
WAVLM_CACHE = REPO_ROOT / "data" / "cache" / "embeddings_wavlm-base.npz"

CREMAD_MANIFEST = REPO_ROOT / "data" / "cremad" / "manifest.csv"
E5_MANIFEST = REPO_ROOT / "data" / "hume_e5" / "manifest.csv"
E6_MANIFEST = REPO_ROOT / "data" / "zurich" / "manifest.csv"
E3_MANIFESTS = (
    REPO_ROOT / "data" / "recorded0" / "manifest_speaker1.csv",
    REPO_ROOT / "data" / "recorded" / "manifest_speaker1.csv",
)

# E5's two voices. Ava trains; Colton is never trained on by any combo here
# and is therefore the only honest held-out E5 evaluation.
E5_TRAIN_VOICE = "hume_ava_song"
E5_HELDOUT_VOICE = "hume_colton_rivers"

SEED = 0
CREMAD_EVAL_SUBSET_N = 300
SUSPICIOUS_UAR = 0.90


def fit_probe(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """The permissive backend's primary path, held fixed across combos.

    `class_weight="balanced"` matters most for the CREMA-D rows (68%
    negative) but is kept everywhere so the classifier is literally the same
    object in every cell of the grid.
    """
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def predict(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray) -> list[Prediction]:
    proba = clf.predict_proba(scaler.transform(X))
    preds = []
    for row in proba:
        probs = {Sentiment(c): float(p) for c, p in zip(clf.classes_, row, strict=True)}
        for s in Sentiment:  # a class absent from a 45-clip training set gets zero mass
            probs.setdefault(s, 0.0)
        sentiment = max(probs, key=probs.get)  # type: ignore[arg-type]
        preds.append(
            Prediction(
                sentiment=sentiment,
                probs=probs,
                confidence=probs[sentiment],
                latency_ms=0.0,
                solution="permissive-combo",
            )
        )
    return preds


def score(preds: list[Prediction], df: pd.DataFrame) -> dict[str, object]:
    y_true = [Sentiment(s) for s in df["prosody_sentiment"]]
    y_pred = [p.sentiment for p in preds]
    counts = Counter(s.value for s in y_pred)
    _top, top_n = counts.most_common(1)[0]
    out = {
        "n_clips": len(df),
        "n_incongruent": int((~df["is_congruent"]).sum()),
        "uar": uar(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "psi_contested": psi_contested(preds, df),
        "psi_strict": psi_strict(preds, df),
        "predicted_class_share": {k: v / len(preds) for k, v in counts.items()},
        "is_degenerate": top_n / len(preds) >= 0.95,
    }
    out |= bootstrap_ci(np.array([s.value for s in y_pred]), df)
    return out


def main() -> None:
    for path in (CREMAD_MANIFEST, E5_MANIFEST, E6_MANIFEST):
        if not path.exists():
            raise SystemExit(f"{path} not found")

    cremad = load_manifest(CREMAD_MANIFEST, repo_root=REPO_ROOT)
    e5 = load_manifest(E5_MANIFEST, repo_root=REPO_ROOT)
    e6 = load_manifest(E6_MANIFEST, repo_root=REPO_ROOT)
    e3 = pd.concat([load_manifest(p, repo_root=REPO_ROOT) for p in E3_MANIFESTS], ignore_index=True)
    e3 = e3[~e3["clip_id"].str.contains("whisper")].reset_index(drop=True)

    cremad_train = cremad[cremad["split"] == "train"].reset_index(drop=True)
    cremad_test = cremad[cremad["split"] == "test"].reset_index(drop=True)
    e6_train = e6[e6["split"] == "train"].reset_index(drop=True)
    e6_val = e6[e6["split"] == "val"].reset_index(drop=True)
    e6_test = e6[e6["split"] == "test"].reset_index(drop=True)
    e5_train = e5[e5["speaker_id"] == E5_TRAIN_VOICE].reset_index(drop=True)
    e5_heldout = e5[e5["speaker_id"] == E5_HELDOUT_VOICE].reset_index(drop=True)

    everything = pd.concat([cremad, e5, e6, e3], ignore_index=True).drop_duplicates(
        subset=["clip_id"]
    )
    logger.info("extracting WavLM embeddings for %d clips (resumable) ...", len(everything))
    cache = extract_and_cache(everything, WavLMEncoder(), WAVLM_CACHE, repo_root=REPO_ROOT)

    training_sets: dict[str, pd.DataFrame] = {
        "cremad": cremad_train,
        "e6": e6_train,
        "e5": e5_train,
        "e5_e6": pd.concat([e5_train, e6_train], ignore_index=True),
        "cremad_e5_e6": pd.concat([cremad_train, e5_train, e6_train], ignore_index=True),
    }

    eval_sets: dict[str, pd.DataFrame] = {
        "cremad_test": stratified_subset(cremad_test, n=CREMAD_EVAL_SUBSET_N, seed=SEED),
        "e6_val": e6_val,
        "e6_test": e6_test,
        "e5_heldout_voice": e5_heldout,
        "e5_train_voice": e5_train,
        "e3": e3,
    }

    matrices = {name: embeddings_matrix(df, cache)[0] for name, df in eval_sets.items()}

    results: dict[str, object] = {}
    for combo, train_df in training_sets.items():
        trained_speakers = set(train_df["speaker_id"])
        X_train, _ = embeddings_matrix(train_df, cache)
        y_train = train_df["prosody_sentiment"].to_numpy()
        scaler, clf = fit_probe(X_train, y_train)
        logger.info(
            "=== %s: %d clips, %d speakers, classes %s ===",
            combo,
            len(train_df),
            len(trained_speakers),
            dict(Counter(y_train.tolist())),
        )

        per_set: dict[str, object] = {}
        for name, df in eval_sets.items():
            overlap = trained_speakers & set(df["speaker_id"])
            if name == "e3" and overlap:
                # E6's marco is E3's speaker1: same human, both sides.
                per_set[name] = {
                    "excluded_reason": (
                        f"trained on {sorted(overlap)}, who also recorded E3 -- scoring here "
                        "would put the same speaker on both sides of the split"
                    )
                }
                logger.info("  %-18s EXCLUDED (speaker overlap: %s)", name, sorted(overlap))
                continue

            s = score(predict(scaler, clf, matrices[name]), df)
            s["in_domain"] = bool(overlap)
            s["overlapping_speakers"] = sorted(overlap)
            per_set[name] = s
            if np.isfinite(s["uar"]) and s["uar"] > SUSPICIOUS_UAR and not overlap:
                logger.warning(
                    "  !! %s on %s: UAR=%.3f above the %.2f ceiling with no speaker overlap "
                    "-- suspect leakage",
                    combo,
                    name,
                    s["uar"],
                    SUSPICIOUS_UAR,
                )
            logger.info(
                "  %-18s n=%4d UAR=%.3f [%.2f,%.2f] PSI=%.3f%s%s",
                name,
                s["n_clips"],
                s["uar"],
                s["uar_ci95"]["lo"],
                s["uar_ci95"]["hi"],
                s["psi_contested"],
                "  IN-DOMAIN" if overlap else "",
                "  DEGENERATE" if s["is_degenerate"] else "",
            )

        results[combo] = {
            "n_train_clips": len(train_df),
            "n_train_speakers": len(trained_speakers),
            "train_class_counts": dict(Counter(y_train.tolist())),
            "eval": per_set,
        }

    payload = {
        "training_sets": {k: len(v) for k, v in training_sets.items()},
        "results": results,
        "method": (
            "Frozen WavLM-base embeddings into StandardScaler + LogisticRegression"
            "(max_iter=2000, class_weight='balanced') -- the permissive backend's primary "
            "path, held identical across all five training sets so the only thing that "
            "varies is the training data. 95% intervals are a percentile bootstrap over "
            "clips. cremad_test is the 300-clip stratified subset used by the other "
            "comparison tables."
        ),
        "e5_split_note": (
            f"E5 has exactly two voices. {E5_TRAIN_VOICE} trains; {E5_HELDOUT_VOICE} is held "
            "out of every training set here and is the only honest E5 evaluation. "
            "e5_train_voice is reported too, flagged in_domain for the models that trained on "
            "it -- the gap between the two columns is what a 45-clip, single-voice training "
            "set actually buys."
        ),
        "caveats": [
            "TRAINING ON E5 BREAKS THIS PROJECT'S OWN DESIGN RULE. DESIGN.md designates E5 "
            "eval-only: it is the instrument that measures prosody sensitivity under "
            "contradiction. A model trained on it can no longer use it as a clean PSI "
            "instrument, so read the cross-corpus columns for those rows.",
            "Sample sizes are wildly unequal by design: 45 clips and one voice for e5, 100 "
            "clips and five speakers for e6, against CREMA-D's 5,235. A combo losing to "
            "CREMA-D is not surprising; a combo BEATING it on held-out speakers would be.",
            "E3 is excluded for every combo trained on E6, because one E6 training speaker "
            "also recorded E3 under the same speaker id.",
            "e6_test is 20 clips from one speaker and e5_heldout_voice is 45 from one "
            "synthetic voice. Differences of a few points between combos are not resolvable "
            "at those sizes, which is why every cell carries an interval.",
            "The MLP fallback in train_permissive is not reproduced here. It triggers on weak "
            "validation UAR and needs a val block that a 45-clip training set cannot spare; "
            "holding the classifier fixed is what makes the five rows comparable.",
        ],
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("wrote %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
