#!/usr/bin/env python3
"""The research backend on CREMA-D's test split -- the number that was never run.

Every results table in this project shows a dash where the audeering
backend's CREMA-D score should be. The reason is historical rather than
principled: `ssa/train.py`'s `train_research` runs the encoder over the
**validation** split to fit its two valence thresholds, and nothing ever ran
it over the test split. So the backend has a val number (n=737, the split
its thresholds were fitted on) and no held-out one, while the permissive
backend has 0.744 on 1,470 held-out clips. Comparing those two directly --
which the licensing discussion in DESIGN.md effectively does -- compares a
fitted score against a held-out one.

This closes that gap. The same harness, the same speaker-disjoint test
split, the same metrics as every other solution.

Three things are measured, because the interesting part is the gap between
them:

1. **val (n=737)** -- where the two thresholds were fitted. Only two
   parameters on 737 clips, so this should not overfit much; reporting it
   next to test is what shows whether that expectation holds.
2. **test, full (n=1,470)** -- the honest held-out number, directly
   comparable to the permissive backend's 0.744.
3. **test, 300-clip stratified subset** -- the split every other solution's
   headline number uses, so the research backend finally has a row in that
   table.

The permissive backend is scored on the identical splits in the same run.
The licensing trade-off (MIT versus CC-BY-NC-SA-4.0) is one of this
project's better arguments, and it deserves to rest on two numbers measured
the same way rather than on one measured differently.

Note what the research backend is: the audeering model **zero-shot**, with
two thresholds cutting its valence output into three classes. It is not
trained on CREMA-D. A lower score here than the permissive probe's is
expected and is the honest cost of not fine-tuning; a *much* lower one says
something about the domain gap between MSP-Podcast (what it was trained on)
and acted studio speech.

Usage:
    uv run python scripts/eval_research_cremad.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_zurich import bootstrap_ci  # noqa: E402
from ssa.combo_manifests import stratified_subset  # noqa: E402
from ssa.embeddings import extract_and_cache, load_cache  # noqa: E402
from ssa.encoders.audeering import AudeeringVADEncoder  # noqa: E402
from ssa.eval.metrics import confusion_matrix  # noqa: E402
from ssa.eval.runner import evaluate, results_path  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.splits import assert_speaker_disjoint  # noqa: E402
from ssa.train import _VADEmbedAdapter  # noqa: E402
from ssa.types import VAD, AudioClip, Sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "research_cremad.json"
MANIFEST_PATH = REPO_ROOT / "data" / "cremad" / "manifest.csv"
VAD_CACHE_PATH = REPO_ROOT / "data" / "cache" / "vad_audeering.npz"

EVAL_SUBSET_N = 300
EVAL_SUBSET_SEED = 0


class CachedVADEncoder:
    """Serves `predict_vad` from the on-disk VAD cache, by clip_id.

    `AcousticSolution` calls the encoder per clip, which would push 1,470
    clips through a 300M-parameter wav2vec2 on CPU every run. The cache
    that `train_research` already writes holds exactly these vectors, so
    this adapter lets the harness reuse them and makes a re-run instant.

    A clip_id that is not cached falls through to the real encoder rather
    than raising: the point is to avoid recomputation, not to fail when the
    cache is cold.
    """

    def __init__(self, cache: dict[str, np.ndarray], inner: AudeeringVADEncoder) -> None:
        self._cache = cache
        self._inner = inner
        self.hits = 0
        self.misses = 0

    def predict_vad(self, clip: AudioClip) -> VAD:
        cached = self._cache.get(clip.clip_id)
        if cached is None:
            self.misses += 1
            return self._inner.predict_vad(clip)
        self.hits += 1
        # _VADEmbedAdapter fixes the order as (valence, arousal, dominance).
        return VAD(valence=float(cached[0]), arousal=float(cached[1]), dominance=float(cached[2]))


def score(solution, df: pd.DataFrame, dataset: str, split_type: str) -> dict[str, object]:
    result = evaluate(solution, df, dataset=dataset, split_type=split_type, repo_root=REPO_ROOT)
    result.to_json(results_path(RESULTS_DIR, solution.name, dataset, split_type))

    y_pred = np.array([p["predicted_sentiment"] for p in result.predictions])
    y_true = [Sentiment(s) for s in df["prosody_sentiment"]]
    cm = confusion_matrix(y_true, [Sentiment(s) for s in y_pred])
    labels = [s.value for s in Sentiment.ordered()]
    counts = Counter(y_pred.tolist())
    _top_label, top_n = counts.most_common(1)[0]

    out = {
        "dataset": dataset,
        "split_type": split_type,
        "n_clips": result.n_clips,
        "n_incongruent": result.n_incongruent,
        "uar": result.uar,
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "psi_contested": result.psi_contested,
        "psi_strict": result.psi_strict,
        "per_class_recall": {
            label: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": cm.tolist()},
        "predicted_class_share": {k: v / len(y_pred) for k, v in counts.items()},
        "is_degenerate": top_n / len(y_pred) >= 0.95,
    }
    out |= bootstrap_ci(y_pred, df)
    logger.info(
        "%-38s %-22s n=%4d UAR=%.3f [%.2f,%.2f] F1=%.3f acc=%.3f",
        solution.name[:38],
        f"{dataset}/{split_type}",
        out["n_clips"],
        out["uar"],
        out["uar_ci95"]["lo"],
        out["uar_ci95"]["hi"],
        out["macro_f1"],
        out["accuracy"],
    )
    return out


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found -- run `make data` first")

    manifest = load_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)
    assert_speaker_disjoint(manifest)  # CLAUDE.md rule 1, checked every run

    val = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    subset = stratified_subset(test, n=EVAL_SUBSET_N, seed=EVAL_SUBSET_SEED)
    logger.info("val n=%d, test n=%d, subset n=%d", len(val), len(test), len(subset))

    # Warm the VAD cache over everything this script scores. Resumable and
    # checkpointed, so an interrupted run loses at most one checkpoint.
    needed = pd.concat([val, test], ignore_index=True).drop_duplicates(subset=["clip_id"])
    cached_before = len(load_cache(VAD_CACHE_PATH))
    logger.info("warming VAD cache (%d cached, %d needed) ...", cached_before, len(needed))
    cache = extract_and_cache(
        needed, _VADEmbedAdapter(AudeeringVADEncoder()), VAD_CACHE_PATH, repo_root=REPO_ROOT
    )

    vad_encoder = CachedVADEncoder(cache, AudeeringVADEncoder())
    research = AcousticSolution(backend="research", encoder=vad_encoder)
    permissive = AcousticSolution(backend="permissive")

    sets = {
        ("val", "speaker_disjoint_val"): val,
        ("cremad", "speaker_disjoint_test"): test,
        ("cremad", "speaker_disjoint_test_subset"): subset,
    }

    results: dict[str, object] = {"research": {}, "permissive": {}}
    for (dataset, split_type), df in sets.items():
        key = f"{dataset}/{split_type}"
        results["research"][key] = score(research, df, dataset, split_type)
        results["permissive"][key] = score(permissive, df, dataset, split_type)

    test_key = "cremad/speaker_disjoint_test"
    r_test = results["research"][test_key]
    p_test = results["permissive"][test_key]
    val_key = "val/speaker_disjoint_val"

    summary = {
        "thresholds": json.loads((REPO_ROOT / "data/cache/thresholds_research.json").read_text()),
        "results": results,
        "headline": {
            "research_test_uar": r_test["uar"],
            "research_test_uar_ci95": r_test["uar_ci95"],
            "permissive_test_uar": p_test["uar"],
            "permissive_test_uar_ci95": p_test["uar_ci95"],
            "gap_permissive_minus_research": p_test["uar"] - r_test["uar"],
            "research_val_uar": results["research"][val_key]["uar"],
            "research_val_minus_test": results["research"][val_key]["uar"] - r_test["uar"],
        },
        "cache": {"hits": vad_encoder.hits, "misses": vad_encoder.misses},
        "method": (
            "Identical harness, splits and metrics for both backends. The research backend "
            "is the audeering model zero-shot, with two valence thresholds fitted on CREMA-D's "
            "validation split (low/high, shown above); it is never trained on CREMA-D. The "
            "permissive backend is WavLM-base frozen plus a logistic-regression probe trained "
            "on CREMA-D's train split. 95% intervals are a percentile bootstrap over clips."
        ),
        "why_this_exists": (
            "The research backend previously had only a validation number -- the split its "
            "thresholds were fitted on -- while the permissive backend had a held-out test "
            "number. Every comparison between them, including the licensing trade-off "
            "argument, was therefore comparing a fitted score against a held-out one. Both "
            "are now measured the same way."
        ),
        "caveats": [
            "The research backend is ZERO-SHOT on CREMA-D: a lower score than a probe trained "
            "on CREMA-D's own training split is expected, and is the honest cost of the "
            "permissive licence rather than evidence that the model is weak.",
            "It was trained on MSP-Podcast (spontaneous, in-the-wild speech) and CREMA-D is "
            "acted studio speech with fixed neutral sentences, so this is itself a "
            "cross-corpus measurement -- arguably the harder direction, and not the setting "
            "it was built for.",
            "CREMA-D's carrier text is always neutral, so PSI here is close to vacuous for "
            "any acoustic solution: there is no competing lexical signal to resist. Read PSI "
            "on E5 and E6 instead, where the words genuinely disagree with the delivery.",
            "val is the split the two thresholds were fitted on. With only two parameters "
            "over 737 clips the fit is not expected to be tight, and the val-minus-test gap "
            "in headline is what shows whether that held.",
        ],
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("wrote %s", SUMMARY_PATH)
    logger.info(
        "HEADLINE  research test UAR=%.3f  permissive test UAR=%.3f  gap=%+.3f  (val-test=%+.3f)",
        summary["headline"]["research_test_uar"],
        summary["headline"]["permissive_test_uar"],
        summary["headline"]["gap_permissive_minus_research"],
        summary["headline"]["research_val_minus_test"],
    )


if __name__ == "__main__":
    main()
