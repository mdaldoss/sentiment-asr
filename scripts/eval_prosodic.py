#!/usr/bin/env python3
"""Score Solution D on every dataset A/B/C are scored on.

The point of adding a fourth solution is comparison, so this deliberately
runs the *same* harness over the *same* manifests rather than inventing a
private evaluation: CREMA-D's speaker-disjoint test split, both E3 takes,
and E5. A number here is directly comparable to the corresponding number
for A, B or C in `results/`.

Two of these sets matter more than the benchmark. E3 is the only real,
unseen human speaker in the repo. E5 is the only set where two thirds of
clips have words and delivery contradicting, which is where PSI separates
a model that hears tone from one that reads text -- and where, on
identical audio, the permissive backend scored 0.682 and the audeering
model 0.211. Solution D exists because a logistic regression on nine
descriptors beat both at recovering the delivery from that same audio
(0.811 vs 0.333 chance, `results/e5_diagnostic.json`). This script asks
whether that advantage survives being turned into an actual classifier
trained on CREMA-D and applied cold to each set.

Also writes the fitted model's feature importances, which is the part no
embedding-based solution in this repo can produce.

Usage:
    uv run python scripts/eval_prosodic.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.combo_manifests import stratified_subset  # noqa: E402
from ssa.eval.runner import evaluate, results_path  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.prosodic_features import MemoizingExtractor  # noqa: E402
from ssa.solutions.prosodic import ProsodicSolution, feature_importances  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "prosodic_summary.json"

CREMAD_MANIFEST = REPO_ROOT / "data" / "cremad" / "manifest.csv"
E3_TAKES = {
    "e3a": REPO_ROOT / "data" / "recorded0" / "manifest_speaker1.csv",
    "e3b": REPO_ROOT / "data" / "recorded" / "manifest_speaker1.csv",
}
E5_MANIFEST = REPO_ROOT / "data" / "hume_e5" / "manifest.csv"

# Matches scripts/eval_backend_combos.py so the CREMA-D column lines up
# with the backend comparison rather than being a third distinct subset.
EVAL_SUBSET_N = 300
EVAL_SUBSET_SEED = 0

# Fitted in scripts/train_prosodic.py; all are scored here, not just the
# one that won on CREMA-D validation.
CANDIDATES: tuple[str, ...] = ("logreg", "linear_svm", "svm_rbf", "hist_gbdt")


def _e3(path: Path) -> pd.DataFrame:
    m = load_manifest(path, repo_root=REPO_ROOT)
    # Whisper prompts carry no PSI-design prosody target, excluded the same
    # way scripts/eval_e3.py excludes them.
    return m[~m["clip_id"].str.contains("whisper")].reset_index(drop=True)


def eval_sets() -> dict[str, tuple[pd.DataFrame, str]]:
    sets: dict[str, tuple[pd.DataFrame, str]] = {}

    cremad = load_manifest(CREMAD_MANIFEST, repo_root=REPO_ROOT)
    test = cremad[cremad["split"] == "test"].reset_index(drop=True)
    sets["cremad"] = (test, "speaker_disjoint_test")
    sets["cremad_subset"] = (
        stratified_subset(test, n=EVAL_SUBSET_N, seed=EVAL_SUBSET_SEED),
        "speaker_disjoint_test_subset",
    )
    for label, path in E3_TAKES.items():
        sets[label] = (_e3(path), "all")
    if E5_MANIFEST.exists():
        sets["e5_hume"] = (load_manifest(E5_MANIFEST, repo_root=REPO_ROOT), "all")
    else:
        logger.warning("E5 manifest missing -- skipping (run `make gen-hume-e5`)")
    return sets


def degeneracy(result) -> dict[str, object]:
    """Flag a model that answers the same thing for (almost) every clip.

    This is not hypothetical: the first run of this script selected the
    in-domain winner (svm_rbf) and it predicted "negative" for 100% of E3
    and E5 clips at 0.88 mean confidence. UAR then lands on exactly 1/3 and
    PSI on exactly 0.5, which look like ordinary weak results rather than a
    collapsed model. Naming it stops that number being read as a finding.
    """
    counts = Counter(p["predicted_sentiment"] for p in result.predictions)
    top_label, top_n = counts.most_common(1)[0]
    share = top_n / max(len(result.predictions), 1)
    return {
        "top_class": top_label,
        "top_class_share": share,
        "is_degenerate": share >= 0.95,
        "predicted_class_counts": dict(counts),
    }


def main() -> None:
    # One extractor shared by every candidate: features are identical across
    # classifiers, and extraction dominates the runtime.
    extractor = MemoizingExtractor()
    solutions = {
        model_type: ProsodicSolution(model_type=model_type, extractor=extractor)
        for model_type in CANDIDATES
    }
    default = ProsodicSolution(extractor=extractor)
    logger.info("scoring %d candidates; training selected %s", len(solutions), default.name)

    summary: dict[str, object] = {
        "selected_by_training": default.name,
        "selection_caveat": (
            "Training selects on CREMA-D's validation split, which is in-domain. Every "
            "candidate is scored on every dataset here because that selection is not "
            "self-evidently the right one -- the first run showed the in-domain winner "
            "collapsing to a single class on both out-of-domain sets."
        ),
        "candidates": {},
    }

    for model_type, solution in solutions.items():
        per_dataset: dict[str, object] = {}
        for name, (manifest, split_type) in eval_sets().items():
            dataset = "cremad" if name.startswith("cremad") else name
            result = evaluate(
                solution, manifest, dataset=dataset, split_type=split_type, repo_root=REPO_ROOT
            )
            if model_type == default._model_type:
                result.to_json(results_path(RESULTS_DIR, solution.name, dataset, split_type))
            per_dataset[name] = {
                "dataset": result.dataset,
                "split_type": result.split_type,
                "n_clips": result.n_clips,
                "n_incongruent": result.n_incongruent,
                "uar": result.uar,
                "macro_f1": result.macro_f1,
                "accuracy": result.accuracy,
                "psi_contested": result.psi_contested,
                "psi_strict": result.psi_strict,
                "degeneracy": degeneracy(result),
            }
            logger.info(
                "%-11s %-14s UAR=%.3f PSI=%.3f %s",
                model_type,
                name,
                result.uar,
                result.psi_contested,
                "DEGENERATE" if per_dataset[name]["degeneracy"]["is_degenerate"] else "",
            )
        summary["candidates"][model_type] = per_dataset

    importances = feature_importances()
    if importances:
        summary["top_features"] = [{"feature": n, "weight": w} for n, w in importances[:20]]
        summary["top_features_note"] = (
            "Importances belong to the training-selected model only; a linear candidate "
            "exposes coefficients and a kernel SVM does not, so this is absent for some."
        )

    summary["interpretation"] = (
        "Solution D uses no pretrained network: eGeMAPS functionals plus pitch/energy "
        "contour descriptors, fed to a classifier fitted on speaker-disjoint CREMA-D. "
        "Nothing in the path can represent a word, so it cannot fail the way the audeering "
        "backend does on E5 (PSI 0.211 -- following the transcript from audio alone). Read "
        "PSI on e5_hume and the two E3 takes rather than CREMA-D UAR: CREMA-D's text is "
        "always neutral, so its PSI does not test genuine word-versus-tone conflict. "
        "top_features lists what the fitted model actually leans on -- an explanation no "
        "embedding-based solution in this repo can offer."
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", SUMMARY_PATH)
    for name, s in summary["datasets"].items():
        logger.info(
            "%-14s UAR=%.3f  PSI=%.3f  (n=%d)", name, s["uar"], s["psi_contested"], s["n_clips"]
        )


if __name__ == "__main__":
    main()
