#!/usr/bin/env python3
"""Does per-speaker feature normalisation rescue cross-domain transfer?

`results/representation_comparison.json` found that explicit prosodic
features have the *higher* in-domain ceiling (0.870 vs WavLM's 0.759 on E3)
but lose roughly twice as much when the classifier is trained on CREMA-D and
meets a new domain cold (-0.46 / -0.52 UAR vs -0.24 / -0.28). The stated
explanation was that absolute pitch, loudness and harmonicity move with
microphone, room, recording level and voice, while a learned representation
is partly invariant to them. That explanation makes a falsifiable
prediction: remove each speaker's own baseline from their features and the
transfer gap should narrow.

This script tests it, on four schemes:

  raw             what Solution D ships today
  dataset_z       z-score each feature over the whole evaluation set: fixes
                  a *domain* offset (one microphone, one room) but not a
                  speaker one
  speaker_z       z-score each feature within speaker_id
  speaker_z_loo   the same, but each clip is standardised using only the
                  speaker's OTHER clips -- so a clip never contributes to
                  its own normalisation

and on both representations, because the claim being tested is comparative:
if normalisation lifts the prosodic features specifically, the fragility
diagnosis was right; if it lifts both or neither, it was not.

**Three caveats that belong next to every number this produces.**

1. It is *transductive*. Normalising within speaker needs a pool of that
   speaker's clips, which a per-clip classifier does not have. Solution D's
   `predict(clip)` contract cannot implement this, which is why the result
   lands here rather than in the shipped model. For Ami -- one user, audio
   accumulating over weeks -- a running per-speaker baseline is realistic,
   and this is a small-sample stand-in for exactly the longitudinal
   baselining DESIGN.md argues for. `speaker_z_loo` removes the
   self-normalisation leak; it does not make the method per-clip.

2. **Mean removal assumes the speaker's clips are class-balanced.** Every
   evaluation set here is balanced by construction (E3: 9 per sentiment per
   take; E5: 10 per delivery per voice; CREMA-D: every actor reads every
   emotion). A speaker who is mostly sad would have their sadness
   normalised away -- the signal and the baseline are the same quantity.
   So these are favourable-case numbers, an upper bound on what the method
   buys, not a deployment estimate.

3. Per-speaker *threshold* recalibration was already tried in this project
   and failed (helped 1 case of 4, `results/calibration.json`). This is a
   different intervention -- on the input features, not the decision
   boundary -- but the prior is not encouraging, and a result here that
   looks too good deserves suspicion rather than celebration.

Usage:
    uv run python scripts/normalise_speaker.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_prosodic import build_pipeline, candidates  # noqa: E402
from ssa.embeddings import embeddings_matrix, extract_and_cache  # noqa: E402
from ssa.encoders.wavlm import WavLMEncoder  # noqa: E402
from ssa.eval.metrics import macro_f1, psi_contested, psi_strict, uar  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.prosodic_features import ProsodicExtractor  # noqa: E402
from ssa.splits import assert_speaker_disjoint  # noqa: E402
from ssa.types import Prediction, Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULT_PATH = REPO_ROOT / "results" / "speaker_normalisation.json"

CREMAD_MANIFEST = REPO_ROOT / "data" / "cremad" / "manifest.csv"
EVAL_MANIFESTS: dict[str, Path] = {
    "e3a": REPO_ROOT / "data" / "recorded0" / "manifest_speaker1.csv",
    "e3b": REPO_ROOT / "data" / "recorded" / "manifest_speaker1.csv",
    "e5_hume": REPO_ROOT / "data" / "hume_e5" / "manifest.csv",
}

PROSODIC_CACHE = REPO_ROOT / "data" / "cache" / "features_prosodic.npz"
WAVLM_CACHE = REPO_ROOT / "data" / "cache" / "embeddings_wavlm-base.npz"

SCHEMES: tuple[str, ...] = ("raw", "dataset_z", "speaker_z", "speaker_z_loo")

# Fewer clips than this and a speaker's mean/std are noise rather than a
# baseline. Groups below it are reported, never silently normalised anyway.
MIN_GROUP = 5
_EPS = 1e-8

# Prosodic gets every candidate (the question "is this the classifier?"
# already bit once). WavLM gets logreg only: it is here as the comparison
# axis, and it is the classifier scripts/compare_representations.py used, so
# the numbers line up with that file.
WAVLM_CANDIDATES: tuple[str, ...] = ("logreg",)


def _group_stats(X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group mean and std, broadcast back to one row per sample."""
    mean = np.zeros_like(X)
    std = np.ones_like(X)
    for g in np.unique(groups):
        m = groups == g
        mean[m] = X[m].mean(axis=0)
        std[m] = X[m].std(axis=0)
    return mean, std


def _loo_stats(X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of each group *excluding the row itself*.

    Closed form from the group's sums, so this costs one pass rather than
    one refit per clip. A group of fewer than 3 rows has no usable
    leave-one-out spread, so those rows fall back to the group statistics
    (and the group is flagged as undersized in the output either way).

    One edge case is worth spelling out, because getting it wrong inverts
    the scheme's meaning. If a speaker's *other* clips are constant on some
    feature -- `voiced_fraction` saturating at 1.0, or a feature the imputer
    filled with the same median for all of them -- the leave-one-out spread
    is zero while the clip being normalised may still differ from them. The
    generic divide-by-zero guard would then map the single most extreme clip
    in the group to 0, i.e. to the value an exactly average clip receives.
    So the scale falls back to the pooled group std in that case (the
    centre, which is the part that matters, stays leave-one-out). That
    readmits a trace of the clip's own contribution, but only through the
    scale and only where the alternative is to destroy the signal outright.
    A feature constant across the *whole* group is genuinely uninformative
    and still ends up at 0.
    """
    mean = np.zeros_like(X)
    std = np.ones_like(X)
    for g in np.unique(groups):
        m = groups == g
        block = X[m]
        n = block.shape[0]
        pooled_std = block.std(axis=0)
        if n < 3:
            mean[m] = block.mean(axis=0)
            std[m] = pooled_std
            continue
        total = block.sum(axis=0)
        total_sq = (block**2).sum(axis=0)
        loo_mean = (total - block) / (n - 1)
        loo_var = (total_sq - block**2) / (n - 1) - loo_mean**2
        loo_std = np.sqrt(np.maximum(loo_var, 0.0))
        mean[m] = loo_mean
        std[m] = np.where(loo_std < _EPS, pooled_std, loo_std)
    return mean, std


def normalise(X: np.ndarray, groups: np.ndarray, scheme: str) -> np.ndarray:
    """Apply one normalisation scheme. Returns a new array.

    A feature that is constant within a group carries no within-group
    information, so it becomes 0 rather than inf -- dividing by a zero std
    is how a "normalisation" quietly turns into garbage.
    """
    if scheme == "raw":
        return X
    if scheme == "dataset_z":
        groups = np.zeros(len(X), dtype=int)  # one pool
        mean, std = _group_stats(X, groups)
    elif scheme == "speaker_z":
        mean, std = _group_stats(X, groups)
    elif scheme == "speaker_z_loo":
        mean, std = _loo_stats(X, groups)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return np.where(std < _EPS, 0.0, (X - mean) / np.maximum(std, _EPS))


def group_sizes(groups: np.ndarray) -> dict[str, object]:
    counts = Counter(groups.tolist())
    undersized = {g: n for g, n in counts.items() if n < MIN_GROUP}
    return {
        "n_groups": len(counts),
        "min_group_size": int(min(counts.values())),
        "median_group_size": float(np.median(list(counts.values()))),
        "undersized_groups": {str(g): int(n) for g, n in sorted(undersized.items())},
    }


def as_predictions(labels: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> list[Prediction]:
    """Wrap raw classifier output so the project's tested PSI code can score
    it, instead of re-implementing PSI here and risking a second definition."""
    preds = []
    for label, row in zip(labels, proba, strict=True):
        probs = {Sentiment(c): float(p) for c, p in zip(classes, row, strict=True)}
        for s in Sentiment:  # a class absent from training gets zero mass
            probs.setdefault(s, 0.0)
        preds.append(
            Prediction(
                sentiment=Sentiment(label),
                probs=probs,
                confidence=probs[Sentiment(label)],
                latency_ms=0.0,
                solution="normalisation-experiment",
            )
        )
    return preds


def score(preds: list[Prediction], df: pd.DataFrame) -> dict[str, object]:
    y_true = [Sentiment(s) for s in df["prosody_sentiment"]]
    y_pred = [p.sentiment for p in preds]
    counts = Counter(s.value for s in y_pred)
    top_label, top_n = counts.most_common(1)[0]
    return {
        "n_clips": len(df),
        "uar": uar(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "psi_contested": psi_contested(preds, df),
        "psi_strict": psi_strict(preds, df),
        "top_class": top_label,
        "top_class_share": top_n / len(preds),
        "is_degenerate": top_n / len(preds) >= 0.95,
    }


def _load_eval(path: Path) -> pd.DataFrame:
    df = load_manifest(path, repo_root=REPO_ROOT)
    # Whisper prompts carry no PSI-design prosody target; excluded the same
    # way scripts/eval_e3.py and scripts/eval_prosodic.py exclude them.
    return df[~df["clip_id"].str.contains("whisper")].reset_index(drop=True)


def build_representation(
    name: str,
    embedder: object,
    cache_path: Path,
    cremad: pd.DataFrame,
    eval_sets: dict[str, pd.DataFrame],
) -> dict[str, object]:
    logger.info("[%s] extracting features (resumable cache %s) ...", name, cache_path.name)
    everything = pd.concat([cremad, *eval_sets.values()], ignore_index=True)
    cache = extract_and_cache(everything, embedder, cache_path, repo_root=REPO_ROOT)

    train = cremad[cremad["split"] == "train"].reset_index(drop=True)
    test = cremad[cremad["split"] == "test"].reset_index(drop=True)
    sets = {"cremad_test": test, **eval_sets}

    X_train_raw, _ = embeddings_matrix(train, cache)
    matrices = {k: embeddings_matrix(df, cache)[0] for k, df in sets.items()}

    # One imputer, fitted on raw training features and applied everywhere:
    # NaNs come from a failed pitch or intensity analysis, and they must be
    # filled BEFORE normalisation or they poison the group statistics.
    imputer = SimpleImputer(strategy="median").fit(X_train_raw)
    X_train_raw = imputer.transform(X_train_raw)
    matrices = {k: imputer.transform(X) for k, X in matrices.items()}

    y_train = train["prosody_sentiment"].to_numpy()
    train_groups = train["speaker_id"].to_numpy()
    group_info = {
        k: group_sizes(df["speaker_id"].to_numpy()) for k, df in ({"train": train} | sets).items()
    }

    model_names = tuple(candidates()) if name == "prosodic" else WAVLM_CANDIDATES
    out: dict[str, object] = {
        "dims": int(X_train_raw.shape[1]),
        "n_train": int(X_train_raw.shape[0]),
        "group_sizes": group_info,
        "schemes": {},
    }

    for scheme in SCHEMES:
        # Training data is normalised with the plain per-speaker statistics
        # under both speaker schemes: leave-one-out exists to keep an
        # *evaluation* clip out of its own normalisation, and there is no
        # such thing to protect on the training side.
        train_scheme = "speaker_z" if scheme == "speaker_z_loo" else scheme
        X_train = normalise(X_train_raw, train_groups, train_scheme)

        per_model: dict[str, object] = {}
        for model_name in model_names:
            clf = candidates()[model_name]
            pipeline = build_pipeline(clf)
            pipeline.fit(X_train, y_train)

            per_dataset = {}
            for set_name, df in sets.items():
                X = normalise(matrices[set_name], df["speaker_id"].to_numpy(), scheme)
                preds = as_predictions(
                    pipeline.predict(X), pipeline.predict_proba(X), pipeline.classes_
                )
                per_dataset[set_name] = score(preds, df)
            per_model[model_name] = per_dataset
            logger.info(
                "[%s] %-13s %-11s " + "  ".join(f"{k}=%.3f" for k in sets),
                name,
                scheme,
                model_name,
                *[per_dataset[k]["uar"] for k in sets],
            )
        out["schemes"][scheme] = per_model
    return out


def deltas(result: dict[str, object]) -> dict[str, object]:
    """Change in UAR from `raw`, per representation / model / dataset.

    The headline the experiment was run to produce, computed here so the
    report never has to subtract two numbers itself and get the sign wrong.
    """
    summary: dict[str, object] = {}
    for rep, rep_data in result.items():
        raw = rep_data["schemes"]["raw"]
        for scheme in SCHEMES:
            if scheme == "raw":
                continue
            for model_name, per_dataset in rep_data["schemes"][scheme].items():
                for set_name, s in per_dataset.items():
                    key = f"{rep}/{model_name}/{set_name}"
                    summary.setdefault(key, {"raw_uar": raw[model_name][set_name]["uar"]})
                    summary[key][f"{scheme}_uar"] = s["uar"]
                    summary[key][f"{scheme}_delta"] = s["uar"] - raw[model_name][set_name]["uar"]
    return summary


def main() -> None:
    cremad = load_manifest(CREMAD_MANIFEST, repo_root=REPO_ROOT)
    assert_speaker_disjoint(cremad)  # CLAUDE.md rule 1, checked every run

    eval_sets = {name: _load_eval(path) for name, path in EVAL_MANIFESTS.items() if path.exists()}
    for name, path in EVAL_MANIFESTS.items():
        if not path.exists():
            logger.warning("%s missing at %s -- skipping", name, path)

    result: dict[str, object] = {}
    result["prosodic"] = build_representation(
        "prosodic", ProsodicExtractor(), PROSODIC_CACHE, cremad, eval_sets
    )
    result["wavlm"] = build_representation("wavlm", WavLMEncoder(), WAVLM_CACHE, cremad, eval_sets)

    payload = {
        "representations": result,
        "uar_deltas_vs_raw": deltas(result),
        "schemes": {
            "raw": "features as Solution D ships them",
            "dataset_z": "z-scored over the whole evaluation set (one pool): corrects a "
            "recording-domain offset, not a speaker one",
            "speaker_z": "z-scored within speaker_id",
            "speaker_z_loo": "z-scored within speaker_id using only that speaker's OTHER "
            "clips, so no clip contributes to its own normalisation",
        },
        "method": (
            "Identical pipeline throughout (median impute -> standardise -> classifier), "
            "fitted on CREMA-D's speaker-disjoint train split and applied cold to each "
            "evaluation set. Only the normalisation applied to the feature vectors "
            "changes between schemes, so any difference is the normalisation."
        ),
        "caveats": [
            "TRANSDUCTIVE: per-speaker statistics need a pool of that speaker's clips, "
            "which a per-clip predict(clip) cannot have. This is why the result is an "
            "experiment and not a change to Solution D. speaker_z_loo removes the "
            "self-normalisation leak but not the pool requirement.",
            "FAVOURABLE CASE: mean removal is only safe when a speaker's clips are "
            "class-balanced, which every set here is by construction (E3 9/sentiment/take, "
            "E5 10/delivery/voice, CREMA-D every actor reads every emotion). A speaker who "
            "is mostly negative would have that negativity normalised away, because the "
            "baseline and the signal are the same quantity. Treat these as an upper bound.",
            "PRIOR NEGATIVE RESULT: per-speaker THRESHOLD recalibration was tried earlier "
            "in this project and helped in 1 of 4 cases. This intervenes on the features "
            "rather than the decision boundary, so it is a different test, but the prior "
            "argues for suspicion of a large gain rather than celebration.",
            "n is small on the sets that matter: E3 is 27 clips from one speaker per take, "
            "E5 is 90 clips from three synthetic voices. No confidence intervals are "
            "computed, and differences of a few points here are not resolvable.",
        ],
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
