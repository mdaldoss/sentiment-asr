#!/usr/bin/env python3
"""Evaluate and retrain on E6, the Zurich multi-speaker recordings.

E6 changes what this project can claim. Until now every human number came
from **one** speaker recorded twice (E3), so "does this generalise across
people" was unanswerable and DESIGN.md said so. E6 has six speakers, and its
train/val/test split is already speaker-disjoint by construction: four
speakers train, Silvia validates, Matteo tests. It is also 70% incongruent,
which makes it the first *human, multi-speaker* set on which PSI means
anything.

Two questions, run in that order:

**1. Zero-shot cross-corpus.** Take the A/B/C/D solutions exactly as they
are -- fitted on CREMA-D, never having heard these voices -- and score them
on E6. This is the cross-corpus generalisation measurement the project has
listed as missing since the beginning. CREMA-D is acted American studio
speech with fixed neutral sentences; E6 is European speakers, many
non-native, recording through laptop microphones on freely-worded
sentences. If the CREMA-D-trained models fall apart here, that is the
training-corpus bottleneck showing up on real data rather than in an
argument.

**2. Does training on these speakers help?** Refit the permissive acoustic
probe and the prosodic classifier under three training sets -- CREMA-D
only, CREMA-D plus E6's 80 training clips, and E6's 80 clips alone -- and
score each on E6's held-out speakers. The third is the interesting one:
80 clips from four speakers against CREMA-D's 5,235 from 64 is a test of
whether *matched* data beats *more* data.

**The leakage trap here is real and specific.** One of E6's training
speakers is the person who recorded E3; `scripts/build_zurich.py` gives him
E3's own `speaker1` id precisely so this cannot pass unnoticed. Any combo
that trains on E6-train therefore must not be scored on E3, and this script
does not score it there. E6's own val and test speakers appear in no
training set.

Usage:
    uv run python scripts/eval_zurich.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.impute import SimpleImputer  # noqa: E402

from scripts.normalise_speaker import as_predictions, score  # noqa: E402
from scripts.train_prosodic import build_pipeline, candidates  # noqa: E402
from ssa.embeddings import embeddings_matrix, extract_and_cache  # noqa: E402
from ssa.eval.runner import evaluate, results_path  # noqa: E402
from ssa.manifest import load_manifest, validate_manifest  # noqa: E402
from ssa.prosodic_features import ProsodicExtractor  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.solutions.fusion import DEFAULT_PARAMS, FusionParams, FusionSolution  # noqa: E402
from ssa.solutions.lexical import LexicalSolution  # noqa: E402
from ssa.solutions.prosodic import ProsodicSolution  # noqa: E402
from ssa.splits import assert_speaker_disjoint  # noqa: E402
from ssa.train import train_permissive  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "e6_summary.json"
CACHE_DIR = REPO_ROOT / "data" / "cache"

E6_MANIFEST = REPO_ROOT / "data" / "zurich" / "manifest.csv"
CREMAD_MANIFEST = REPO_ROOT / "data" / "cremad" / "manifest.csv"
FUSION_PARAMS_PATH = CACHE_DIR / "fusion_params.json"

PROSODIC_CACHE = CACHE_DIR / "features_prosodic.npz"
WAVLM_CACHE = CACHE_DIR / "embeddings_wavlm-base.npz"

PROSODIC_MODELS: tuple[str, ...] = ("logreg", "svm_rbf", "hist_gbdt")

# Same ceiling ssa/train.py and scripts/train_prosodic.py use: SOTA
# speaker-independent is below 0.90, so anything above it here is a bug.
SUSPICIOUS_UAR = 0.90


def e6_eval_sets(e6: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Held-out speakers first; `e6_all` last and clearly named.

    `e6_all` includes the training speakers, so it is only honest for the
    zero-shot pass, where nothing was fitted on any of them. It is excluded
    from every retrained combo below.
    """
    return {
        "e6_test": e6[e6.split == "test"].reset_index(drop=True),
        "e6_val": e6[e6.split == "val"].reset_index(drop=True),
        "e6_all": e6,
    }


def zero_shot(e6: pd.DataFrame) -> dict[str, object]:
    """Every solution as it ships, on voices none of them has heard."""
    lexical = LexicalSolution()
    permissive = AcousticSolution(backend="permissive")
    research = AcousticSolution(backend="research")
    params = (
        FusionParams(**json.loads(FUSION_PARAMS_PATH.read_text()))
        if FUSION_PARAMS_PATH.exists()
        else DEFAULT_PARAMS
    )
    fusion = FusionSolution(lexical=lexical, acoustic=permissive, params=params)
    solutions = [lexical, permissive, research, fusion]

    try:
        prosodic = ProsodicSolution()
        solutions.append(prosodic)
    except FileNotFoundError:
        logger.warning("no prosodic model -- run `make train-prosodic`; skipping Solution D")

    out: dict[str, object] = {}
    for solution in solutions:
        per_set: dict[str, object] = {}
        for name, df in e6_eval_sets(e6).items():
            result = evaluate(solution, df, dataset=name, split_type="all", repo_root=REPO_ROOT)
            result.to_json(results_path(RESULTS_DIR, solution.name, name, "all"))
            per_set[name] = {
                "n_clips": result.n_clips,
                "n_incongruent": result.n_incongruent,
                "uar": result.uar,
                "macro_f1": result.macro_f1,
                "accuracy": result.accuracy,
                "psi_contested": result.psi_contested,
                "psi_strict": result.psi_strict,
            }
            logger.info(
                "zero-shot %-46s %-8s UAR=%.3f PSI=%.3f",
                solution.name[:46],
                name,
                result.uar,
                result.psi_contested,
            )
        out[solution.name] = per_set
    return out


def _flag_if_suspicious(value: float, model: str, dataset: str) -> None:
    """Shout when a speaker-independent score exceeds the plausibility
    ceiling. SOTA speaker-independent is below 0.90 (CLAUDE.md pinned
    facts), so anything above it is a bug -- and this script has already
    produced one. `ssa/train.py` and `scripts/train_prosodic.py` carry the
    same guard; without it here, the leak showed up only as a number that
    looked like very good news."""
    if np.isfinite(value) and value > SUSPICIOUS_UAR:
        logger.warning(
            "  !! %s on %s scored UAR=%.3f, above the %.2f plausibility ceiling -- "
            "suspect leakage before believing it",
            model,
            dataset,
            value,
            SUSPICIOUS_UAR,
        )


def _prosodic_combo(
    train_df: pd.DataFrame, eval_sets: dict[str, pd.DataFrame], cache: dict[str, np.ndarray]
) -> dict[str, object]:
    """Fit every prosodic candidate on `train_df`, score on each eval set.

    Only `split == "train"` rows are fitted on. That restriction is not
    cosmetic: the first run of this script fitted the whole manifest, so the
    `e6train_only` combo trained on the validation speaker and was then
    scored on her, returning UAR 1.000 -- above the 0.90 ceiling CLAUDE.md
    names as the signature of leakage rather than success. `train_permissive`
    has always honoured the split; this had to as well.
    """
    fit_rows = train_df[train_df["split"] == "train"].reset_index(drop=True)
    if len(fit_rows) == 0:
        raise ValueError("no split=='train' rows to fit on")

    X_train_raw, _ = embeddings_matrix(fit_rows, cache)
    imputer = SimpleImputer(strategy="median").fit(X_train_raw)
    X_train = imputer.transform(X_train_raw)
    y_train = fit_rows["prosody_sentiment"].to_numpy()

    per_model: dict[str, object] = {}
    for model_name in PROSODIC_MODELS:
        pipeline = build_pipeline(candidates()[model_name])
        pipeline.fit(X_train, y_train)
        per_dataset = {}
        for name, df in eval_sets.items():
            X = imputer.transform(embeddings_matrix(df, cache)[0])
            preds = as_predictions(
                pipeline.predict(X), pipeline.predict_proba(X), pipeline.classes_
            )
            per_dataset[name] = score(preds, df)
            _flag_if_suspicious(per_dataset[name]["uar"], f"prosodic/{model_name}", name)
        per_model[model_name] = per_dataset
        logger.info(
            "  prosodic %-10s %s",
            model_name,
            "  ".join(
                f"{k}: UAR={v['uar']:.3f} PSI={v['psi_contested']:.3f}"
                for k, v in per_dataset.items()
            ),
        )
    # n_fit_rows is a sibling of the model map, never a member of it: mixing
    # a scalar into a mapping whose values are all per-dataset score dicts is
    # what made the previous run die halfway through the retraining pass.
    return {"n_fit_rows": len(fit_rows), "models": per_model}


def retrained(e6: pd.DataFrame, cremad: pd.DataFrame) -> dict[str, object]:
    """Three training sets, scored on E6's held-out speakers only."""
    e6_train = e6[e6.split == "train"].reset_index(drop=True)
    e6_val = e6[e6.split == "val"].reset_index(drop=True)
    e6_test = e6[e6.split == "test"].reset_index(drop=True)

    combos: dict[str, pd.DataFrame] = {
        "cremad_only": cremad,
        "cremad_plus_e6train": pd.concat([cremad, e6_train], ignore_index=True),
        "e6train_only": pd.concat([e6_train, e6_val], ignore_index=True),
    }

    def eval_sets_for(combo_name: str) -> dict[str, pd.DataFrame]:
        """`e6_test` is held out from everything. `e6_val` is Silvia, and
        `e6train_only` uses her to *select* its model -- the permissive
        probe picks its classifier on her, and she is this combo's only
        validation data. A selection set is not a held-out set, so she is
        dropped from that combo's evaluation rather than reported as one.
        `e6_all` never appears here at all: it contains the training
        speakers, and is honest only in the zero-shot pass.
        """
        sets = {"e6_test": e6_test}
        if combo_name != "e6train_only":
            sets["e6_val"] = e6_val
        return sets

    prosodic_cache = extract_and_cache(
        pd.concat([cremad, e6], ignore_index=True),
        ProsodicExtractor(),
        PROSODIC_CACHE,
        repo_root=REPO_ROOT,
    )

    out: dict[str, object] = {}
    for combo_name, train_manifest in combos.items():
        logger.info("=== combo %s (%d rows) ===", combo_name, len(train_manifest))
        validate_manifest(train_manifest)
        # CLAUDE.md rule 1, per combo rather than once: the E6 training
        # speakers include E3's, and a merged manifest is exactly where an
        # overlap slips through.
        assert_speaker_disjoint(train_manifest)

        eval_sets = eval_sets_for(combo_name)
        combo_out: dict[str, object] = {
            "n_train_rows": len(train_manifest),
            "n_fit_rows": int((train_manifest["split"] == "train").sum()),
            "scored_on": sorted(eval_sets),
        }
        if combo_name == "e6train_only":
            combo_out["e6_val_excluded_reason"] = (
                "e6_val is this combo's model-selection set (it has no other validation "
                "data), so scoring it there would report a selection score as a held-out "
                "one. e6_test is untouched by every combo and is the comparable number."
            )

        # --- Solution B, permissive backend -------------------------------
        manifest_path = CACHE_DIR / "combo_manifests" / f"e6_{combo_name}.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        train_manifest.to_csv(manifest_path, index=False)
        probe_path = CACHE_DIR / "combo_probes" / f"e6_{combo_name}.joblib"
        probe_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fit = train_permissive(manifest_path, probe_path=probe_path, cache_path=WAVLM_CACHE)
            solution = AcousticSolution(backend="permissive", probe_path=probe_path)
            per_set = {}
            for name, df in eval_sets.items():
                result = evaluate(
                    solution,
                    df,
                    dataset=name,
                    split_type=f"combo_{combo_name}",
                    repo_root=REPO_ROOT,
                )
                per_set[name] = {
                    "n_clips": result.n_clips,
                    "uar": result.uar,
                    "macro_f1": result.macro_f1,
                    "psi_contested": result.psi_contested,
                    "psi_strict": result.psi_strict,
                }
                _flag_if_suspicious(result.uar, "permissive", name)
                logger.info(
                    "  permissive %-8s UAR=%.3f PSI=%.3f", name, result.uar, result.psi_contested
                )
            combo_out["permissive"] = {"fit": fit, "eval": per_set}
        except Exception as exc:  # a combo may be too small to fit
            logger.warning("  permissive failed for %s: %s", combo_name, exc)
            combo_out["permissive"] = {"error": str(exc)}

        # --- Solution D, prosodic features --------------------------------
        combo_out["prosodic"] = _prosodic_combo(train_manifest, eval_sets, prosodic_cache)
        out[combo_name] = combo_out

    return out


def previous_zero_shot() -> dict[str, object]:
    """Reuse the last run's zero-shot block verbatim.

    Only valid because the zero-shot pass depends on nothing the
    retraining half touches -- it scores the shipped models on E6 and
    fits nothing. If that ever stops being true, this shortcut has to go.
    """
    if not SUMMARY_PATH.exists():
        raise SystemExit("--retrain-only needs an existing summary to reuse; run once without it")
    return json.loads(SUMMARY_PATH.read_text())["zero_shot"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain-only",
        action="store_true",
        help="Skip the zero-shot pass and reuse its results from the existing summary. "
        "The zero-shot pass runs ASR over every clip and is the slow half; the retraining "
        "half is what changes when a combo or a split is corrected.",
    )
    args = parser.parse_args()

    if not E6_MANIFEST.exists():
        raise SystemExit(f"{E6_MANIFEST} not found -- run `make data-zurich` first")

    e6 = load_manifest(E6_MANIFEST, repo_root=REPO_ROOT)
    assert_speaker_disjoint(e6)
    cremad = load_manifest(CREMAD_MANIFEST, repo_root=REPO_ROOT)

    logger.info(
        "E6: %d clips, %d speakers, %d incongruent (%.0f%%)",
        len(e6),
        e6.speaker_id.nunique(),
        (~e6.is_congruent).sum(),
        100 * (~e6.is_congruent).mean(),
    )

    summary: dict[str, object] = {
        "dataset": "e6_zurich",
        "n_clips": len(e6),
        "n_speakers": int(e6.speaker_id.nunique()),
        "n_incongruent": int((~e6.is_congruent).sum()),
        "speakers_by_split": {
            f"{s}/{sp}": int(n) for (s, sp), n in e6.groupby(["split", "speaker_id"]).size().items()
        },
        "zero_shot": previous_zero_shot() if args.retrain_only else zero_shot(e6),
        "retrained": retrained(e6, cremad),
    }

    summary["interpretation"] = (
        "zero_shot is the cross-corpus generalisation number this project has listed as "
        "missing from the start: models fitted on CREMA-D (acted, American, studio, fixed "
        "neutral sentences) meeting six European speakers on laptop microphones. retrained "
        "asks whether 80 clips from four matched speakers beat 5,235 acted ones. Read PSI "
        "beside UAR: E6 is 70% incongruent, so a model that reads the words rather than "
        "hearing the delivery is penalised here in a way CREMA-D's always-neutral text "
        "cannot penalise it."
    )
    summary["caveats"] = [
        "E6's test split is ONE speaker and 20 clips; its val split is one speaker and 20. "
        "Differences of a few points between combos are not resolvable at that size, and no "
        "confidence intervals are computed.",
        "text_sentiment on E6 is assigned by hand from the sentence text (see "
        "scripts/build_zurich.py) because the dataset labels intended delivery only. PSI "
        "here therefore depends on that labelling in a way it does not for E5, where the "
        "text valence was fixed by design before any audio existed.",
        "One E6 training speaker also recorded E3, and carries E3's speaker1 id. No combo "
        "that trains on E6 is scored on E3 anywhere in this script.",
        "e6_all (120 clips) includes the training speakers and appears only in the zero-shot "
        "pass, where no model has been fitted on any E6 audio.",
    ]

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("wrote %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
