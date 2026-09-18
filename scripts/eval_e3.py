#!/usr/bin/env python3
"""E3 evaluation: both recorded takes through the standard harness, plus the
D1-style acoustic diagnostics that let Cartesia and human speech be compared
directly.

Two takes exist (`data/recorded0` = take0, `data/recorded` = take1): the same
30 prompts, same speaker, re-recorded independently. That is a free
test-retest experiment, and it is why E3's numbers are reported in pairs
throughout DESIGN.md rather than as one figure -- a single take could be a
lucky or unlucky recording session, and the whole point of this script is to
show whether the findings replicate.

Unlike E1 (CREMA-D, always-neutral text), E3's manifest has real lexical
sentiment crossed against prosody, so **PSI is computable here** -- this is
the first dataset where the words-vs-tone test runs on genuinely competing
signals (see ssa/eval/metrics.py).

No API key needed -- this only reads committed audio and trained artifacts.

Usage:
    uv run python scripts/eval_e3.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.audio import load_clip  # noqa: E402
from ssa.eval.runner import evaluate, results_path  # noqa: E402
from ssa.manifest import load_manifest  # noqa: E402
from ssa.paralinguistic import extract_features  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.solutions.fusion import DEFAULT_PARAMS, FusionParams, FusionSolution  # noqa: E402
from ssa.solutions.lexical import LexicalSolution  # noqa: E402
from ssa.types import Sentiment  # noqa: E402
from ssa.voicehealth import pitch_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
FUSION_PARAMS_PATH = REPO_ROOT / "data" / "cache" / "fusion_params.json"

TAKES: dict[str, Path] = {
    "e3a": REPO_ROOT / "data" / "recorded0" / "manifest_speaker1.csv",
    "e3b": REPO_ROOT / "data" / "recorded" / "manifest_speaker1.csv",
}

FEATURE_NAMES = (
    "f0_mean",
    "f0_std",
    "f0_range",
    "egemaps_loudness",
    "egemaps_hnr",
    "egemaps_jitter",
    "egemaps_shimmer_db",
    "rms",
    "speech_rate_cps",
)


def _build_solutions() -> dict[str, object]:
    lexical = LexicalSolution()
    acoustic_permissive = AcousticSolution(backend="permissive")
    acoustic_research = AcousticSolution(backend="research")
    params = (
        FusionParams.from_json(FUSION_PARAMS_PATH)
        if FUSION_PARAMS_PATH.exists()
        else DEFAULT_PARAMS
    )
    fusion = FusionSolution(lexical, acoustic_permissive, params)
    return {
        "A": lexical,
        "B_permissive": acoustic_permissive,
        "B_research": acoustic_research,
        "C_fusion": fusion,
    }


def run_standard_eval() -> None:
    """A/B(both backends)/C through the standard harness -> results/*.json,
    same schema as E1's, so report.py's existing table renderer needs no
    E3-specific branch."""
    solutions = _build_solutions()
    for dataset, manifest_path in TAKES.items():
        manifest = load_manifest(manifest_path, repo_root=REPO_ROOT)
        # Whisper prompts have no PSI-design prosody target (congruent by
        # convention, see record_prompts.py) -- exclude them here, same as
        # T11's paralinguistic demo does by clip_id.
        manifest = manifest[~manifest["clip_id"].str.contains("whisper")].reset_index(drop=True)
        for solution in solutions.values():
            result = evaluate(
                solution, manifest, dataset=dataset, split_type="all", repo_root=REPO_ROOT
            )
            path = results_path(RESULTS_DIR, solution.name, dataset, "all")
            result.to_json(path)
            logger.info("wrote %s", path)


def _measure_take(manifest_path: Path, label: str) -> pd.DataFrame:
    manifest = load_manifest(manifest_path, repo_root=REPO_ROOT)
    manifest = manifest[~manifest["clip_id"].str.contains("whisper")].reset_index(drop=True)

    acoustic_permissive = AcousticSolution(backend="permissive")
    acoustic_research = AcousticSolution(backend="research")

    rows = []
    for row in manifest.itertuples(index=False):
        clip = load_clip(REPO_ROOT / row.path, clip_id=row.clip_id)
        pstats = pitch_stats(clip)
        feats = extract_features(clip)
        perm_pred = acoustic_permissive.predict(clip)
        res_pred = acoustic_research.predict(clip)
        rows.append(
            {
                "take": label,
                "clip_id": row.clip_id,
                "carrier": row.text,
                "intended": row.prosody_sentiment,
                "text_sentiment": row.text_sentiment,
                "f0_mean": pstats.f0_mean,
                "f0_std": pstats.f0_std,
                "f0_range": pstats.f0_range,
                "egemaps_loudness": feats.loudness_mean,
                "egemaps_hnr": feats.hnr_mean,
                "egemaps_jitter": feats.jitter_mean,
                "egemaps_shimmer_db": feats.shimmer_db_mean,
                "rms": float(np.sqrt(np.mean(clip.samples.astype(np.float64) ** 2))),
                "speech_rate_cps": len(row.text) / clip.duration_s if clip.duration_s > 0 else 0.0,
                "permissive_predicted_sentiment": perm_pred.sentiment.value,
                "permissive_valence_proxy": (
                    perm_pred.probs[Sentiment.POSITIVE] - perm_pred.probs[Sentiment.NEGATIVE]
                ),
                "research_valence": res_pred.vad.valence if res_pred.vad is not None else None,
                "research_arousal": res_pred.vad.arousal if res_pred.vad is not None else None,
            }
        )
    return pd.DataFrame(rows)


def recoverability_cv(df: pd.DataFrame) -> dict:
    """Leave-one-carrier-out logistic regression on FEATURE_NAMES, same
    method as scripts/gen_emotion_probe_d1.py's recoverability_cv -- the
    point is direct comparability with D1's 0.175 (vs 0.200 chance, 5-way).
    Here it's 3-way, chance=0.333."""
    sub = df.dropna(subset=list(FEATURE_NAMES))
    X = sub[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = sub["intended"].to_numpy()
    groups = sub["carrier"].to_numpy()

    if len(set(groups)) < 2:
        return {"accuracy": None, "n": len(y), "chance": 1 / 3, "note": "fewer than 2 carriers"}

    logo = LeaveOneGroupOut()
    correct = total = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(set(y[train_idx])) < 2:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[train_idx]), y[train_idx])
        preds = clf.predict(scaler.transform(X[test_idx]))
        correct += int((preds == y[test_idx]).sum())
        total += len(test_idx)

    return {
        "accuracy": (correct / total) if total else None,
        "n": total,
        "chance": 1 / 3,
        "n_dropped_missing_features": len(df) - len(sub),
    }


def f0_rank_consistency(df: pd.DataFrame) -> dict:
    """Rank the 3 intended-sentiment groups by mean F0 per carrier, average
    each group's rank across carriers -- chance is 2.0 for 3 classes. Same
    method as D1's, for direct comparability."""
    ranks: dict[str, list[int]] = {}
    for _carrier, group in df.groupby("carrier"):
        by_sentiment = group.groupby("intended")["f0_mean"].mean()
        order = by_sentiment.sort_values(ascending=False).index.tolist()
        for rank, sentiment in enumerate(order, start=1):
            ranks.setdefault(sentiment, []).append(rank)
    return {
        "chance_mean_rank": (3 + 1) / 2,
        "mean_rank_by_sentiment": {s: float(np.mean(r)) for s, r in ranks.items()},
    }


def run_control_comparison() -> None:
    """The centrepiece diagnostic: run D1's exact instruments on both E3
    takes, and compute test-retest agreement between them. Written
    alongside D1's own results/d1_emotion_probe.json for direct
    side-by-side reading."""
    take_dfs = {label: _measure_take(path, label) for label, path in TAKES.items()}

    d1_path = RESULTS_DIR / "d1_emotion_probe.json"
    d1 = json.loads(d1_path.read_text()) if d1_path.exists() else None

    result: dict = {
        "cartesia_d1": {
            "recoverability_accuracy": d1["recoverability_cv"]["accuracy"] if d1 else None,
            "recoverability_chance": d1["recoverability_cv"]["chance"] if d1 else None,
            "f0_span_hz": d1["f0_span_across_emotions_hz"] if d1 else None,
        },
        "human_e3": {},
    }

    for label, df in take_dfs.items():
        cv = recoverability_cv(df)
        rank = f0_rank_consistency(df)
        f0_by_sentiment = df.groupby("intended")["f0_mean"].mean()
        result["human_e3"][label] = {
            "n_clips": len(df),
            "recoverability_cv": cv,
            "f0_rank_consistency": rank,
            "f0_span_hz": float(f0_by_sentiment.max() - f0_by_sentiment.min()),
            "mean_f0_by_sentiment": f0_by_sentiment.round(2).to_dict(),
        }

    # Test-retest: same carrier+intended-sentiment pair exists in both
    # takes (the manifests share prompts) -- join on that to correlate
    # each instrument's reading of itself across two independent recordings.
    a = take_dfs["e3a"].copy()
    b = take_dfs["e3b"].copy()
    a["join_key"] = a["carrier"] + "|" + a["intended"]
    b["join_key"] = b["carrier"] + "|" + b["intended"]
    paired = a.merge(b, on="join_key", suffixes=("_a", "_b"))
    test_retest = {}
    for feat in ("f0_mean", "rms", "research_valence", "permissive_valence_proxy"):
        x, y = paired[f"{feat}_a"].to_numpy(float), paired[f"{feat}_b"].to_numpy(float)
        test_retest[feat] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else None
    result["test_retest_correlation_take0_vs_take1"] = {"n_paired": len(paired), **test_retest}

    result["interpretation"] = (
        "Same instruments, same method as D1 (scripts/gen_emotion_probe_d1.py), run on "
        "real human recordings instead of Cartesia TTS. If recoverability/F0-span/rank "
        "come out well above chance here while D1 was at or below chance, that rules out "
        "the instruments as the explanation for D1's negative result -- the audio itself "
        "is what differs. Replicated across two independent takes by the same speaker; "
        "see test_retest_correlation_take0_vs_take1 for how consistent each instrument's "
        "own readings are across takes, which bears on Solution B's reliability, not just "
        "its accuracy. n=1 speaker -- this is a controlled instrument check, not a claim "
        "about speech-emotion recognition in general."
    )

    out_path = RESULTS_DIR / "d1_vs_e3_control.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", out_path)

    for label, df in take_dfs.items():
        df.to_csv(RESULTS_DIR / f"{label}_measurements.csv", index=False)


def main() -> None:
    run_standard_eval()
    run_control_comparison()


if __name__ == "__main__":
    main()
