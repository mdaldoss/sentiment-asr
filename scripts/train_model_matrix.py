#!/usr/bin/env python3
"""Three backends x three training/test configurations, one table.

Asked directly: train the **research** (audeering VAD), **permissive**
(frozen WavLM + probe) and **prosody** (eGeMAPS + contour features)
backends under three data regimes, and report what each one does.

  X1  cremad_hume__zurich        train CREMA-D + Hume (E5)  -> test Zurich humans (E6)
  X2  cremad__hume               train CREMA-D             -> test Hume (E5)
  X3  cremad_hume_zurich__disjoint
                                 train CREMA-D + Hume + Zurich -> test a
                                 speaker-disjoint split of that same union

X1 and X2 are cross-corpus by construction: the test corpus contributes
nothing to training. X3 is the in-corpus-but-speaker-disjoint regime, and it
is the one to read with the most suspicion -- CLAUDE.md rule 8 exists because
a within-corpus number is not a generalisation number, and X3 is presented
here beside X1/X2 precisely so the gap is visible rather than implied.

**Every configuration is speaker-disjoint end to end.** `assert_speaker_disjoint`
runs on each configuration's assembled manifest before anything is fitted
(CLAUDE.md rule 1). Hume's two voices are treated as two speakers: Ava Song
trains, Colton Rivers never does. Zurich's `speaker1` is the same human who
recorded E3, so E3 is not scored anywhere in this script.

**Training on E5 breaks a rule this project set for itself**, and that is
stated rather than buried: DESIGN.md designates E5 eval-only because it is the
instrument that measures prosody sensitivity under contradiction. X1 and X3
train on Ava's half of it, so for those two configurations E5 is no longer a
clean PSI instrument -- which is why Colton (never trained on) is the only E5
column that means anything for them, and why X2, which trains on no E5 at all,
is the honest PSI reading.

**What "training" means differs per backend, and the difference is the
point.** The permissive and prosody backends fit a full classifier over frozen
features. The research backend as this repo ships it fits exactly **two valence
thresholds** -- so training data composition can move it barely at all. To
separate "this backend is insensitive to training data" from "this backend only
has two knobs", a second research variant is fitted: a logistic regression over
the 3-D VAD vector. Both are reported.

Usage:
    uv run python scripts/train_model_matrix.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_zurich import bootstrap_ci  # noqa: E402
from scripts.train_prosodic import build_pipeline, candidates  # noqa: E402
from ssa.combo_manifests import stratified_subset  # noqa: E402
from ssa.embeddings import embeddings_matrix, extract_and_cache, load_cache  # noqa: E402
from ssa.encoders.audeering import AudeeringVADEncoder  # noqa: E402
from ssa.encoders.wavlm import WavLMEncoder  # noqa: E402
from ssa.eval.metrics import (  # noqa: E402
    confusion_matrix,
    macro_f1,
    psi_contested,
    psi_strict,
    uar,
)
from ssa.manifest import load_manifest  # noqa: E402
from ssa.prosodic_features import ProsodicExtractor  # noqa: E402
from ssa.splits import TEST, TRAIN, VAL, assert_speaker_disjoint  # noqa: E402
from ssa.train import _VADEmbedAdapter  # noqa: E402
from ssa.types import Prediction, Sentiment  # noqa: E402
from ssa.vad import ValenceThresholds, fit_valence_thresholds, valence_to_sentiment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "model_matrix.json"
CACHE_DIR = REPO_ROOT / "data" / "cache"

WAVLM_CACHE = CACHE_DIR / "embeddings_wavlm-base.npz"
VAD_CACHE = CACHE_DIR / "vad_audeering.npz"
PROSODIC_CACHE = CACHE_DIR / "features_prosodic.npz"

CREMAD_MANIFEST = REPO_ROOT / "data" / "cremad" / "manifest.csv"
E5_MANIFEST = REPO_ROOT / "data" / "hume_e5" / "manifest.csv"
E6_MANIFEST = REPO_ROOT / "data" / "zurich" / "manifest.csv"

# E5 has exactly two voices. Ava trains wherever E5 is a training corpus;
# Colton is never trained on by any configuration here and is therefore the
# only honest held-out E5 evaluation.
E5_TRAIN_VOICE = "hume_ava_song"
E5_HELDOUT_VOICE = "hume_colton_rivers"

SEED = 0
CREMAD_SUBSET_N = 300  # the project's existing convention for a balanced pooled view
# SOTA speaker-independent is <0.90 on ESD, <0.78 on IEMOCAP (CLAUDE.md).
# Above this on a speaker-disjoint set means a bug, most likely leakage.
SUSPICIOUS_UAR = 0.90
# The prosody backend fits four candidates; this one is the pre-registered
# headline so the grid is not silently reporting the best of four.
PROSODY_HEADLINE = "logreg"


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    question: str
    train: pd.DataFrame
    val: pd.DataFrame
    eval_sets: dict[str, pd.DataFrame]
    notes: tuple[str, ...]


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------


def _relabel(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split
    return out


def build_configs() -> tuple[dict[str, Config], pd.DataFrame]:
    cremad = load_manifest(CREMAD_MANIFEST, repo_root=REPO_ROOT)
    e5 = load_manifest(E5_MANIFEST, repo_root=REPO_ROOT)
    e6 = load_manifest(E6_MANIFEST, repo_root=REPO_ROOT)

    cremad_train = cremad[cremad["split"] == TRAIN].reset_index(drop=True)
    cremad_val = cremad[cremad["split"] == VAL].reset_index(drop=True)
    cremad_test = cremad[cremad["split"] == TEST].reset_index(drop=True)

    e5_ava = e5[e5["speaker_id"] == E5_TRAIN_VOICE].reset_index(drop=True)
    e5_colton = e5[e5["speaker_id"] == E5_HELDOUT_VOICE].reset_index(drop=True)

    e6_train = e6[e6["split"] == TRAIN].reset_index(drop=True)
    e6_val = e6[e6["split"] == VAL].reset_index(drop=True)
    e6_test = e6[e6["split"] == TEST].reset_index(drop=True)

    configs: dict[str, Config] = {}

    # X1: CREMA-D + Hume -> Zurich humans. Zurich contributes nothing to the
    # fit, so all 160 of its clips (8 speakers) are a legitimate test set.
    configs["X1_cremad_hume__zurich"] = Config(
        name="X1_cremad_hume__zurich",
        question="Train CREMA-D + Hume, test on the Zurich human recordings.",
        train=pd.concat([cremad_train, _relabel(e5_ava, TRAIN)], ignore_index=True),
        val=pd.concat([cremad_val, _relabel(e5_colton, VAL)], ignore_index=True),
        eval_sets={"zurich_all": e6, "zurich_test_speaker": e6_test},
        notes=(
            "Hume's Ava Song trains; Colton Rivers is held out into the validation "
            "set so threshold fitting and candidate selection see a second corpus "
            "rather than CREMA-D alone. Neither voice is in any test set here.",
            "All 160 Zurich clips are scored: 8 speakers, none of whom appear in "
            "training. zurich_test_speaker is Zurich's own 20-clip held-out speaker, "
            "reported separately for comparability with X3 and results/e6_summary.json.",
        ),
    )

    # X2: CREMA-D -> Hume. The shipped baseline, scored on E5.
    configs["X2_cremad__hume"] = Config(
        name="X2_cremad__hume",
        question="Train CREMA-D only, test on the Hume incongruence set.",
        train=cremad_train,
        val=cremad_val,
        eval_sets={
            "hume_all": e5,
            "hume_ava": e5_ava,
            "hume_colton": e5_colton,
        },
        notes=(
            "No E5 anywhere in the fit, so this is the only configuration whose E5 "
            "PSI reading is a clean instrument (DESIGN.md designates E5 eval-only).",
            "Both voices are scored together and separately: 45 clips each, and a "
            "two-voice synthetic corpus is not a population.",
        ),
    )

    # X3: everything, split by speaker.
    x3_train = pd.concat(
        [cremad_train, _relabel(e5_ava, TRAIN), _relabel(e6_train, TRAIN)], ignore_index=True
    )
    x3_val = pd.concat([cremad_val, _relabel(e6_val, VAL)], ignore_index=True)
    x3_test_parts = {
        "cremad_test": cremad_test,
        "hume_colton": e5_colton,
        "zurich_test_speaker": e6_test,
    }
    pooled_full = pd.concat(list(x3_test_parts.values()), ignore_index=True)
    pooled_balanced = pd.concat(
        [
            stratified_subset(cremad_test, n=CREMAD_SUBSET_N, seed=SEED),
            e5_colton,
            e6_test,
        ],
        ignore_index=True,
    )
    configs["X3_cremad_hume_zurich__disjoint"] = Config(
        name="X3_cremad_hume_zurich__disjoint",
        question=(
            "Train CREMA-D + Hume + Zurich, test on a speaker-disjoint split of the same union."
        ),
        train=x3_train,
        val=x3_val,
        eval_sets={"pooled_all": pooled_full, "pooled_cremad_subset300": pooled_balanced}
        | x3_test_parts,
        notes=(
            "The union's split is each corpus's own speaker-disjoint split, "
            "concatenated: CREMA-D 64/9/18 speakers, Hume Ava/-/Colton, Zurich "
            "5/2/1 speakers. assert_speaker_disjoint is run on the assembled manifest.",
            "pooled_all is 1,535 clips of which 1,470 are CREMA-D, so it is a "
            "CREMA-D number wearing a union's name. pooled_cremad_subset300 "
            "subsamples CREMA-D test to 300 (class-stratified, seed 0) so the three "
            "corpora are less lopsided. Read the per-corpus rows first.",
            "This is the within-corpus regime for all three corpora at once. Its "
            "numbers are not generalisation numbers -- X1 and X2 are.",
        ),
    )

    everything = pd.concat([cremad, e5, e6], ignore_index=True).drop_duplicates(subset=["clip_id"])
    return configs, everything


def check_disjoint(config: Config) -> None:
    """CLAUDE.md rule 1, verified per configuration rather than once at split time."""
    parts = [_relabel(config.train, TRAIN), _relabel(config.val, VAL)]
    for df in config.eval_sets.values():
        parts.append(_relabel(df, TEST))
    assert_speaker_disjoint(pd.concat(parts, ignore_index=True))


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


def _as_predictions(
    labels: np.ndarray, proba: np.ndarray, classes: np.ndarray, solution: str
) -> list[Prediction]:
    """Wrap raw classifier output so the project's tested PSI code scores it,
    rather than re-implementing PSI here and risking a second definition."""
    preds = []
    for label, row in zip(labels, proba, strict=True):
        probs = {Sentiment(c): float(p) for c, p in zip(classes, row, strict=True)}
        for s in Sentiment:  # a class absent from a small training set gets zero mass
            probs.setdefault(s, 0.0)
        preds.append(
            Prediction(
                sentiment=Sentiment(label),
                probs=probs,
                confidence=probs[Sentiment(label)],
                latency_ms=0.0,
                solution=solution,
            )
        )
    return preds


def _threshold_predictions(
    valences: np.ndarray, thresholds: ValenceThresholds, solution: str
) -> list[Prediction]:
    """A threshold read-out has no probabilities; a one-hot stands in so the
    Prediction contract holds. Nothing downstream here reads calibration off
    these -- ECE on a one-hot would be meaningless and is not reported."""
    preds = []
    for v in valences:
        sentiment = valence_to_sentiment(float(v), thresholds)
        probs = {s: 0.0 for s in Sentiment}
        probs[sentiment] = 1.0
        preds.append(
            Prediction(
                sentiment=sentiment,
                probs=probs,
                confidence=1.0,
                latency_ms=0.0,
                solution=solution,
            )
        )
    return preds


def fit_research_thresholds(
    X_val: np.ndarray, y_val: list[Sentiment]
) -> tuple[ValenceThresholds, dict[str, float]]:
    """The shipped research backend: frozen audeering VAD, two valence
    thresholds grid-searched on the validation split (never train, never
    test -- docs/ARCHITECTURE.md section 4)."""
    thresholds = fit_valence_thresholds(X_val[:, 0], y_val)
    val_pred = [p.sentiment for p in _threshold_predictions(X_val[:, 0], thresholds, "research")]
    return thresholds, {
        "low": thresholds.low,
        "high": thresholds.high,
        "val_uar": uar(y_val, val_pred),
    }


def fit_linear(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """Standardise then multinomial logistic regression, class-weighted.

    This is the permissive backend's primary path (`ssa.train.train_permissive`),
    replicated rather than called: `train_permissive` additionally swaps in an
    MLP when validation UAR is weak, and a classifier that changes identity
    between cells would make the cells incomparable. The same fixed classifier
    is used for the research-VAD head so that backend's rows differ only in
    their input representation.

    `class_weight="balanced"` matters most for CREMA-D (68% negative) and is
    kept everywhere so the object is literally identical in every cell.
    """
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score(preds: list[Prediction], df: pd.DataFrame) -> dict[str, object]:
    y_true = [Sentiment(s) for s in df["prosody_sentiment"]]
    y_pred = [p.sentiment for p in preds]
    counts = Counter(s.value for s in y_pred)
    top_label, top_n = counts.most_common(1)[0]
    cm = confusion_matrix(y_true, y_pred)
    ordered = [s.value for s in Sentiment.ordered()]
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recalls = np.diag(cm) / np.where(support == 0, np.nan, support)

    out: dict[str, object] = {
        "n_clips": len(df),
        "n_incongruent": int((~df["is_congruent"]).sum()),
        "uar": uar(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        # Rule 7: accuracy never appears alone, only beside UAR.
        "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred, strict=True)])),
        "psi_contested": psi_contested(preds, df),
        "psi_strict": psi_strict(preds, df),
        "per_class_recall": {
            c: (float(r) if np.isfinite(r) else None) for c, r in zip(ordered, recalls, strict=True)
        },
        "support": dict(zip(ordered, (int(s) for s in support), strict=True)),
        "confusion_matrix": {"labels": ordered, "rows_true_cols_pred": cm.tolist()},
        "predicted_class_share": {k: v / len(preds) for k, v in counts.items()},
        "top_class": top_label,
        "is_degenerate": top_n / len(preds) >= 0.95,
    }
    out |= bootstrap_ci(np.array([s.value for s in y_pred]), df)
    return out


def majority_baseline(df: pd.DataFrame) -> dict[str, object]:
    """Always predict the training-agnostic majority class of this eval set.

    Its UAR is 1/3 by construction for three classes -- that is the point:
    it is the number any row here has to beat before it means anything, and
    it is deliberately computed rather than asserted.
    """
    top = df["prosody_sentiment"].value_counts().idxmax()
    labels = np.array([top] * len(df))
    proba = np.array([[1.0]] * len(df))
    preds = _as_predictions(labels, proba, np.array([top]), "majority-baseline")
    s = score(preds, df)
    return {k: s[k] for k in ("uar", "macro_f1", "accuracy", "psi_contested", "psi_strict")} | {
        "always_predicts": top
    }


def _flag(name: str, backend: str, s: dict[str, object]) -> None:
    if np.isfinite(s["uar"]) and s["uar"] > SUSPICIOUS_UAR:  # type: ignore[arg-type]
        logger.warning(
            "!! %s/%s UAR=%.3f exceeds the %.2f plausibility ceiling on a speaker-disjoint "
            "set -- suspect a bug or leakage before believing it",
            backend,
            name,
            s["uar"],
            SUSPICIOUS_UAR,
        )


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------


def run_config(
    config: Config,
    reps: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    check_disjoint(config)
    logger.info("=== %s: %s", config.name, config.question)
    logger.info(
        "    train %d clips / %d speakers, val %d / %d",
        len(config.train),
        config.train["speaker_id"].nunique(),
        len(config.val),
        config.val["speaker_id"].nunique(),
    )

    y_train = config.train["prosody_sentiment"].to_numpy()
    y_val_s = [Sentiment(s) for s in config.val["prosody_sentiment"]]
    backends: dict[str, object] = {}

    def matrices(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {"__train__": embeddings_matrix(config.train, cache)[0]}
        out["__val__"] = embeddings_matrix(config.val, cache)[0]
        for name, df in config.eval_sets.items():
            out[name] = embeddings_matrix(df, cache)[0]
        return out

    # --- research: frozen audeering VAD -----------------------------------
    vad = matrices(reps["vad"])
    thresholds, threshold_info = fit_research_thresholds(vad["__val__"], y_val_s)
    logger.info(
        "  research(thresholds): low=%.3f high=%.3f (val UAR=%.3f)",
        thresholds.low,
        thresholds.high,
        threshold_info["val_uar"],
    )
    per_set = {}
    for name, df in config.eval_sets.items():
        s = score(_threshold_predictions(vad[name][:, 0], thresholds, "research"), df)
        _flag(name, "research", s)
        per_set[name] = s
    backends["research_valence_thresholds"] = {
        "description": (
            "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, frozen, with two "
            "valence thresholds grid-searched on this configuration's validation split. "
            "Two free parameters in total. CC-BY-NC-SA-4.0, research use only."
        ),
        "fitted": threshold_info,
        "eval": per_set,
    }

    # --- research variant: a trained head on the 3-D VAD vector ------------
    scaler, clf = fit_linear(vad["__train__"], y_train)
    per_set = {}
    for name, df in config.eval_sets.items():
        proba = clf.predict_proba(scaler.transform(vad[name]))
        labels = clf.classes_[proba.argmax(axis=1)]
        s = score(_as_predictions(labels, proba, clf.classes_, "research-vad-head"), df)
        _flag(name, "research_vad_head", s)
        per_set[name] = s
    backends["research_vad_head"] = {
        "description": (
            "Same frozen audeering encoder, but its (valence, arousal, dominance) "
            "triple fed to the same StandardScaler + balanced LogisticRegression used "
            "for the permissive probe. Present so that 'the research backend barely "
            "moves with training data' can be distinguished from 'the research backend "
            "only has two knobs to move'."
        ),
        "eval": per_set,
    }

    # --- permissive: frozen WavLM + probe ---------------------------------
    wavlm = matrices(reps["wavlm"])
    scaler, clf = fit_linear(wavlm["__train__"], y_train)
    val_pred = [Sentiment(s) for s in clf.predict(scaler.transform(wavlm["__val__"]))]
    per_set = {}
    for name, df in config.eval_sets.items():
        proba = clf.predict_proba(scaler.transform(wavlm[name]))
        labels = clf.classes_[proba.argmax(axis=1)]
        s = score(_as_predictions(labels, proba, clf.classes_, "permissive"), df)
        _flag(name, "permissive", s)
        per_set[name] = s
    backends["permissive"] = {
        "description": (
            "microsoft/wavlm-base frozen (mean+std pooled, 1536-d) into StandardScaler + "
            "LogisticRegression(max_iter=2000, class_weight='balanced'). MIT end to end."
        ),
        "fitted": {"val_uar": uar(y_val_s, val_pred), "val_macro_f1": macro_f1(y_val_s, val_pred)},
        "eval": per_set,
    }

    # --- prosody: eGeMAPS + contour features ------------------------------
    pros = matrices(reps["prosodic"])
    fitted_scores: dict[str, dict[str, float]] = {}
    per_candidate: dict[str, dict[str, object]] = {}
    for cand_name, clf_obj in candidates().items():
        pipeline = build_pipeline(clf_obj)
        pipeline.fit(pros["__train__"], y_train)
        val_pred = [Sentiment(s) for s in pipeline.predict(pros["__val__"])]
        fitted_scores[cand_name] = {
            "val_uar": uar(y_val_s, val_pred),
            "val_macro_f1": macro_f1(y_val_s, val_pred),
        }
        cand_eval = {}
        for name, df in config.eval_sets.items():
            proba = pipeline.predict_proba(pros[name])
            labels = pipeline.classes_[proba.argmax(axis=1)]
            s = score(_as_predictions(labels, proba, pipeline.classes_, "prosody"), df)
            if cand_name == PROSODY_HEADLINE:
                _flag(name, "prosody", s)
            cand_eval[name] = s
        per_candidate[cand_name] = cand_eval

    val_selected = max(fitted_scores, key=lambda k: fitted_scores[k]["val_uar"])
    backends["prosody"] = {
        "description": (
            "eGeMAPS + pitch/loudness contour descriptors (no pretrained network, "
            "nothing in the path can represent a word) into impute -> standardise -> "
            "classifier. All four candidates from scripts/train_prosodic.py are fitted "
            f"and reported; {PROSODY_HEADLINE!r} is the pre-registered headline so this "
            "table is not quietly showing the best of four."
        ),
        "headline_candidate": PROSODY_HEADLINE,
        "val_selected_candidate": val_selected,
        "candidate_val_scores": fitted_scores,
        "eval": per_candidate[PROSODY_HEADLINE],
        "eval_all_candidates": per_candidate,
    }

    for backend, payload in backends.items():
        for name, s in payload["eval"].items():  # type: ignore[index]
            logger.info(
                "  %-28s %-24s n=%4d UAR=%.3f [%.2f,%.2f] F1=%.3f PSI=%.3f%s",
                backend,
                name,
                s["n_clips"],
                s["uar"],
                s["uar_ci95"]["lo"],
                s["uar_ci95"]["hi"],
                s["macro_f1"],
                s["psi_contested"],
                "  DEGENERATE" if s["is_degenerate"] else "",
            )

    return {
        "question": config.question,
        "n_train_clips": len(config.train),
        "n_train_speakers": int(config.train["speaker_id"].nunique()),
        "train_class_counts": dict(Counter(y_train.tolist())),
        "train_sources": dict(Counter(config.train["source"].tolist())),
        "n_val_clips": len(config.val),
        "n_val_speakers": int(config.val["speaker_id"].nunique()),
        "eval_sets": {
            name: {
                "n_clips": len(df),
                "n_speakers": int(df["speaker_id"].nunique()),
                "n_incongruent": int((~df["is_congruent"]).sum()),
                "majority_baseline": majority_baseline(df),
            }
            for name, df in config.eval_sets.items()
        },
        "notes": list(config.notes),
        "backends": backends,
    }


def main() -> None:
    for path in (CREMAD_MANIFEST, E5_MANIFEST, E6_MANIFEST):
        if not path.exists():
            raise SystemExit(f"{path} not found -- run `make data` first")

    configs, everything = build_configs()

    logger.info("ensuring all three representations are cached for %d clips", len(everything))
    reps: dict[str, dict[str, np.ndarray]] = {}
    # Encoders are constructed lazily: a fully warm cache should not pay for
    # loading a 1.2 GB checkpoint just to look up vectors it already has.
    for key, cache_path, make_embedder in (
        ("wavlm", WAVLM_CACHE, WavLMEncoder),
        ("vad", VAD_CACHE, lambda: _VADEmbedAdapter(AudeeringVADEncoder())),
        ("prosodic", PROSODIC_CACHE, ProsodicExtractor),
    ):
        cache = load_cache(cache_path)
        missing = set(everything["clip_id"]) - set(cache)
        if missing:
            logger.info("%s: extracting %d missing clips", key, len(missing))
            cache = extract_and_cache(everything, make_embedder(), cache_path, repo_root=REPO_ROOT)
        reps[key] = cache

    results = {name: run_config(cfg, reps) for name, cfg in configs.items()}

    payload = {
        "configurations": results,
        "method": (
            "Three acoustic backends fitted under three data regimes. The classifier on "
            "top of the permissive and research-head representations is held identical "
            "(StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'), "
            "seed 0) so that what varies between those cells is the representation and the "
            "training data, not the estimator. The prosody backend fits all four candidates "
            f"from scripts/train_prosodic.py; {PROSODY_HEADLINE!r} is the headline and the "
            "others are reported in eval_all_candidates. 95% intervals are a 2,000-sample "
            "percentile bootstrap over clips (scripts/eval_zurich.bootstrap_ci)."
        ),
        "gold_label": (
            "prosody_sentiment on every row, never text_sentiment (CLAUDE.md rule 2). "
            "UAR is the headline; accuracy appears only beside it (rule 7)."
        ),
        "caveats": [
            "TRAINING ON E5 (Hume) BREAKS THIS PROJECT'S OWN DESIGN RULE. DESIGN.md "
            "designates E5 eval-only because it is the instrument that measures prosody "
            "sensitivity under contradiction. X1 and X3 train on Ava Song's 45 clips, so "
            "for those configurations E5 is no longer a clean PSI instrument. X2, which "
            "trains on no E5 at all, is the honest PSI reading.",
            "Sample sizes are wildly unequal by design: 45 Hume clips per voice and 20 "
            "Zurich test clips against CREMA-D's 1,470-clip test split. Differences of a "
            "few points between backends are not resolvable at those sizes, which is why "
            "every cell carries a bootstrap interval. Read the intervals, not the ranks.",
            "The Zurich test speaker is ONE person and the Hume held-out voice is ONE "
            "synthetic voice. A controlled anecdote, not a population claim.",
            "E3 is not scored anywhere here: Zurich's speaker1 is the same human who "
            "recorded E3, and X3 trains on him.",
            "The research backend is CC-BY-NC-SA-4.0 (research use only) and cannot ship "
            "in a commercial product without a separate license from audEERING.",
            "Only three configurations were requested, so there is no 'train on Zurich "
            "alone' or 'train on Hume alone' row here; results/combos_e5_e6.json already "
            "covers those for the permissive backend.",
            "Bootstrap resampling is over clips, which treats them as exchangeable. The "
            "same speakers and carrier sentences recur, so the true intervals are if "
            "anything wider than these.",
        ],
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("wrote %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
