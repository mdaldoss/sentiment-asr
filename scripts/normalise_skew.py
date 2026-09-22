#!/usr/bin/env python3
"""What is per-speaker normalisation worth when the speaker is not balanced?

`results/speaker_normalisation.json` found a large gain from z-scoring
features within speaker: on CREMA-D-trained prosodic models, +0.19 to +0.33
UAR out of domain, and 9 of 12 collapsed model/dataset pairs restored to
real predictions. It shipped with a caveat -- that removing a speaker's mean
is only safe when that speaker's clips are class-balanced, which every
evaluation set in this repo is **by construction** (E3: 9 per sentiment per
take; E5: 15 per delivery per voice; CREMA-D: every actor reads every
emotion).

That caveat was an argument. This measures it, because the difference
matters more than the headline does: Ami's users are one person each, over
weeks, and a senior in a low-mood stretch is exactly a speaker whose clips
are *not* balanced. If the gain is really a gift from balanced evaluation
sets, the method is an artefact of how this repo's data was built and must
not be presented as a path forward.

The mechanism to worry about is specific. Under balance, a speaker's mean
sits near the average of the three class means, so centring makes the
classes roughly symmetric about zero -- which is the condition that lets a
boundary fitted on another corpus land correctly. Skew the speaker toward
one class and the mean slides toward that class's centroid, dragging its
clips toward the origin and pushing the others out. The prediction is
therefore that the gain should *shrink monotonically with skew*, and it is
the shape of that decay, not its existence, that says whether the method
survives contact with a real user.

Note this is also the one place where something label-shaped enters a method
that never touches labels: per-speaker mean removal uses no label, but it
implicitly assumes a label *distribution*. That is the subtle leak, and the
only way to size it is to vary the distribution.

Method: draw class-imbalanced subsamples of each evaluation set, per
speaker, at three skew levels; score `raw` and `speaker_z` on the **same**
draw so clip difficulty is held constant; repeat over seeds and over which
class is the majority, so the answer is not an artefact of one easy class.
UAR is class-balanced by definition, so it stays comparable across skew
levels -- the small classes get noisier, but the estimator does not tilt.

Usage:
    uv run python scripts/normalise_skew.py
"""

from __future__ import annotations

import json
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.impute import SimpleImputer  # noqa: E402

from scripts.normalise_speaker import (  # noqa: E402
    CREMAD_MANIFEST,
    EVAL_MANIFESTS,
    PROSODIC_CACHE,
    _load_eval,
    as_predictions,
    normalise,
    score,
)
from scripts.train_prosodic import build_pipeline, candidates  # noqa: E402
from ssa.embeddings import embeddings_matrix, extract_and_cache  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.prosodic_features import ProsodicExtractor  # noqa: E402
from ssa.splits import assert_speaker_disjoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULT_PATH = REPO_ROOT / "results" / "speaker_normalisation_skew.json"

# Fraction of each MINORITY class kept, per speaker; the majority class keeps
# everything. Expressed this way rather than as a target majority share so
# the same three levels apply to sets with different per-class pool sizes,
# and the realised share is reported rather than assumed.
MINORITY_KEEP: dict[str, float] = {"balanced": 1.0, "moderate": 0.45, "strong": 0.22}
MIN_PER_CLASS = 2  # below this, a class's recall is one coin flip

N_DRAWS = 20
MODELS: tuple[str, ...] = ("logreg", "svm_rbf", "hist_gbdt")
SCHEMES: tuple[str, ...] = ("raw", "speaker_z")
CLASSES: tuple[str, ...] = ("negative", "neutral", "positive")


def skewed_indices(
    df: pd.DataFrame, majority: str, keep: float, rng: np.random.Generator
) -> np.ndarray:
    """Row positions of one class-imbalanced draw, applied within each speaker.

    Per speaker so the skew is a property of the *speaker*, which is what
    per-speaker normalisation sees. Skewing the set as a whole while leaving
    each speaker balanced would test nothing.
    """
    picked: list[int] = []
    positions = np.arange(len(df))
    for speaker in df["speaker_id"].unique():
        in_speaker = df["speaker_id"].to_numpy() == speaker
        for cls in CLASSES:
            rows = positions[in_speaker & (df["prosody_sentiment"].to_numpy() == cls)]
            if len(rows) == 0:
                continue
            n = len(rows) if cls == majority else max(MIN_PER_CLASS, round(keep * len(rows)))
            picked.extend(rng.choice(rows, size=min(n, len(rows)), replace=False).tolist())
    return np.sort(np.array(picked, dtype=int))


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n_draws": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_draws": int(arr.size),
    }


def main() -> None:
    cremad = load_manifest(CREMAD_MANIFEST, repo_root=REPO_ROOT)
    assert_speaker_disjoint(cremad)  # CLAUDE.md rule 1, checked every run

    eval_sets = {n: _load_eval(p) for n, p in EVAL_MANIFESTS.items() if p.exists()}
    everything = pd.concat([cremad, *eval_sets.values()], ignore_index=True)
    cache = extract_and_cache(everything, ProsodicExtractor(), PROSODIC_CACHE, repo_root=REPO_ROOT)

    train = cremad[cremad["split"] == "train"].reset_index(drop=True)
    X_train_raw, _ = embeddings_matrix(train, cache)
    imputer = SimpleImputer(strategy="median").fit(X_train_raw)
    X_train_raw = imputer.transform(X_train_raw)
    y_train = train["prosody_sentiment"].to_numpy()
    train_groups = train["speaker_id"].to_numpy()
    matrices = {
        k: imputer.transform(embeddings_matrix(df, cache)[0]) for k, df in eval_sets.items()
    }

    # Fit once per (scheme, model); every draw below only re-normalises and
    # re-predicts, which is what makes 20 draws x 3 majority classes cheap.
    logger.info("fitting %d models ...", len(SCHEMES) * len(MODELS))
    pipelines = {}
    for scheme in SCHEMES:
        X_train = normalise(X_train_raw, train_groups, scheme)
        for model_name in MODELS:
            pipeline = build_pipeline(candidates()[model_name])
            pipeline.fit(X_train, y_train)
            pipelines[scheme, model_name] = pipeline

    result: dict[str, object] = {}
    for set_name, df in eval_sets.items():
        X_full = matrices[set_name]
        per_level: dict[str, object] = {}

        for level, keep in MINORITY_KEEP.items():
            # Draws are shared across schemes and models: the same subsample
            # is scored by everything, so a difference is never a difference
            # in which clips got drawn.
            draws = []
            for majority in CLASSES:
                for seed in range(N_DRAWS):
                    # crc32, not hash(): Python randomises string hashing per
                    # process, so hash() here would make the draws differ
                    # between runs of the same script (CLAUDE.md: every
                    # random operation takes an explicit seed).
                    base = zlib.crc32(f"{set_name}|{level}|{majority}".encode())
                    rng = np.random.default_rng(base + seed)
                    draws.append((majority, skewed_indices(df, majority, keep, rng)))

            shares, sizes = [], []
            for _, idx in draws:
                counts = df.iloc[idx]["prosody_sentiment"].value_counts()
                shares.append(float(counts.max() / counts.sum()))
                sizes.append(len(idx))

            per_model: dict[str, object] = {}
            for model_name in MODELS:
                per_scheme: dict[str, object] = {}
                for scheme in SCHEMES:
                    pipeline = pipelines[scheme, model_name]
                    uars, psis, degen = [], [], []
                    for _, idx in draws:
                        sub = df.iloc[idx].reset_index(drop=True)
                        X = normalise(X_full[idx], sub["speaker_id"].to_numpy(), scheme)
                        preds = as_predictions(
                            pipeline.predict(X), pipeline.predict_proba(X), pipeline.classes_
                        )
                        s = score(preds, sub)
                        uars.append(s["uar"])
                        psis.append(s["psi_contested"])
                        degen.append(bool(s["is_degenerate"]))
                    per_scheme[scheme] = {
                        "uar": aggregate(uars),
                        "psi_contested": aggregate(psis),
                        "degenerate_rate": float(np.mean(degen)),
                    }
                per_scheme["speaker_z_minus_raw_uar"] = (
                    per_scheme["speaker_z"]["uar"]["mean"] - per_scheme["raw"]["uar"]["mean"]
                )
                per_model[model_name] = per_scheme
                logger.info(
                    "%-8s %-9s %-10s raw=%.3f speaker_z=%.3f (delta %+.3f)",
                    set_name,
                    level,
                    model_name,
                    per_scheme["raw"]["uar"]["mean"],
                    per_scheme["speaker_z"]["uar"]["mean"],
                    per_scheme["speaker_z_minus_raw_uar"],
                )

            per_level[level] = {
                "minority_keep": keep,
                "realised_majority_share": {
                    "mean": float(np.mean(shares)),
                    "min": float(np.min(shares)),
                    "max": float(np.max(shares)),
                },
                "clips_per_draw": {"mean": float(np.mean(sizes)), "min": int(np.min(sizes))},
                "n_draws": len(draws),
                "models": per_model,
            }
        result[set_name] = per_level

    payload = {
        "datasets": result,
        "method": (
            f"Class-imbalanced subsamples drawn per speaker at three skew levels, "
            f"{N_DRAWS} seeds x 3 choices of majority class = {N_DRAWS * 3} draws per level. "
            "Each draw is scored by every scheme and model, so a difference between them is "
            "never a difference in which clips were drawn. Models are fitted once on "
            "speaker-disjoint CREMA-D train and applied cold; only the evaluation set's "
            "composition changes."
        ),
        "question": (
            "Per-speaker mean removal uses no labels but implicitly assumes a label "
            "DISTRIBUTION: under balance, a speaker's mean sits near the average of the three "
            "class centroids, which is what lets a boundary fitted elsewhere land correctly. "
            "Every evaluation set in this repo is balanced per speaker by construction, so the "
            "gain reported in results/speaker_normalisation.json was measured in the most "
            "favourable case available. This measures how much of it survives skew."
        ),
        "caveats": [
            "Subsampling shrinks the sets: a strong-skew draw of E3 is ~13 clips from one "
            "speaker. Per-draw UAR is accordingly noisy, which is why means over 60 draws "
            "and their spread are reported rather than single numbers.",
            "UAR is class-balanced by construction, so it stays comparable across skew "
            "levels; the minority classes get noisier but the estimator does not tilt toward "
            "the majority. Plain accuracy would not be comparable here and is not reported.",
            "Only the SPEAKER's balance is varied. CREMA-D's training-side balance is left "
            "alone because it is a fixed, known property of the training corpus -- the "
            "deployment question is about the user's clips, not the corpus's.",
            "Still transductive: this measures whether the gain survives imbalance, not "
            "whether the method can run per-clip. It cannot.",
        ],
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
