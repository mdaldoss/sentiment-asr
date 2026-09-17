"""Train the acoustic solution's probe on cached WavLM embeddings.

The encoder is frozen; only a small probe on top of its embeddings is
fitted here. That is what "lightweight" means for this project (CLAUDE.md):
no backprop through the encoder, training finishes in seconds once
embeddings are cached, and the CPU cost is dominated by the one-time
embedding extraction, not by training.

Usage:
    uv run python -m ssa.train --backend permissive
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ssa.embeddings import embeddings_matrix, extract_and_cache
from ssa.encoders.audeering import AudeeringVADEncoder
from ssa.encoders.wavlm import WavLMEncoder
from ssa.eval.metrics import macro_f1, uar
from ssa.splits import assert_speaker_disjoint
from ssa.types import AudioClip, Sentiment
from ssa.vad import fit_valence_thresholds

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "cremad" / "manifest.csv"
CACHE_PATH = REPO_ROOT / "data" / "cache" / "embeddings_wavlm-base.npz"
PROBE_PATH = REPO_ROOT / "data" / "cache" / "probe_permissive.joblib"
VAD_CACHE_PATH = REPO_ROOT / "data" / "cache" / "vad_audeering.npz"
THRESHOLDS_PATH = REPO_ROOT / "data" / "cache" / "thresholds_research.json"

# T4 acceptance criteria (docs/TASKS.md): speaker-independent UAR on CREMA-D
# test should land ~0.45-0.70. Above this, investigate for leakage before
# celebrating -- SOTA ceilings are <0.90 on ESD, <0.78 on IEMOCAP.
SUSPICIOUS_UAR_THRESHOLD = 0.90
# "LR underperforms" trigger for trying the MLP fallback (per T4 spec).
WEAK_UAR_THRESHOLD = 0.40


def _split_xy(
    manifest: pd.DataFrame, split: str, cache: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[Sentiment]]:
    sub = manifest[manifest["split"] == split]
    if len(sub) == 0:
        raise ValueError(f"no rows with split={split!r}")
    X, _ids = embeddings_matrix(sub, cache)
    y = [Sentiment(s) for s in sub["prosody_sentiment"]]
    return X, y


def train_permissive(manifest_path: Path = MANIFEST_PATH) -> dict[str, float]:
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run `make data` first")

    manifest = pd.read_csv(manifest_path, dtype={"clip_id": str, "speaker_id": str})
    # CLAUDE.md rule 1: verify before every training run, not just once at split time.
    assert_speaker_disjoint(manifest)

    encoder = WavLMEncoder()
    cache = extract_and_cache(manifest, encoder, CACHE_PATH, repo_root=REPO_ROOT)

    X_train, y_train = _split_xy(manifest, "train", cache)
    X_val, y_val = _split_xy(manifest, "val", cache)
    y_train_str = [s.value for s in y_train]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # class_weight="balanced" matters here: CREMA-D is ~68% negative
    # (CLAUDE.md pinned facts), so an unweighted fit would happily ignore
    # the minority classes and still look decent on plain accuracy.
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train_s, y_train_str)
    logreg_val_pred = [Sentiment(s) for s in logreg.predict(X_val_s)]
    logreg_uar = uar(y_val, logreg_val_pred)
    logreg_f1 = macro_f1(y_val, logreg_val_pred)
    logger.info("logistic regression: val UAR=%.3f macroF1=%.3f", logreg_uar, logreg_f1)

    best_name, best_model, best_uar, best_f1 = "logreg", logreg, logreg_uar, logreg_f1

    if logreg_uar < WEAK_UAR_THRESHOLD:
        logger.info(
            "logreg val UAR=%.3f < %.2f, trying a shallow MLP fallback per T4 spec",
            logreg_uar,
            WEAK_UAR_THRESHOLD,
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=500, early_stopping=True, random_state=0
        )
        mlp.fit(X_train_s, y_train_str)
        mlp_val_pred = [Sentiment(s) for s in mlp.predict(X_val_s)]
        mlp_uar = uar(y_val, mlp_val_pred)
        mlp_f1 = macro_f1(y_val, mlp_val_pred)
        logger.info("shallow MLP: val UAR=%.3f macroF1=%.3f", mlp_uar, mlp_f1)
        if mlp_uar > best_uar:
            best_name, best_model, best_uar, best_f1 = "mlp", mlp, mlp_uar, mlp_f1

    if best_uar > SUSPICIOUS_UAR_THRESHOLD:
        logger.warning(
            "val UAR=%.3f is suspiciously high (SOTA ceiling is <0.90 on ESD, <0.78 on "
            "IEMOCAP) -- check for speaker leakage before trusting this number",
            best_uar,
        )

    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": best_model, "model_type": best_name}, PROBE_PATH)
    logger.info("saved %s probe (val UAR=%.3f) to %s", best_name, best_uar, PROBE_PATH)

    return {"val_uar": best_uar, "val_macro_f1": best_f1, "model_type": best_name}


class _VADEmbedAdapter:
    """Adapts AudeeringVADEncoder to embeddings.py's Embedder protocol
    (.embed(clip) -> np.ndarray) so the same resumable cache mechanism used
    for WavLM embeddings works for VAD predictions too, with no new caching
    code. Order is (valence, arousal, dominance) -- fixed here, not tied to
    the model's own (arousal, dominance, valence) output order (see
    ssa/encoders/audeering.py)."""

    def __init__(self, vad_encoder: AudeeringVADEncoder) -> None:
        self._vad_encoder = vad_encoder

    def embed(self, clip: AudioClip) -> np.ndarray:
        vad = self._vad_encoder.predict_vad(clip)
        return np.array([vad.valence, vad.arousal, vad.dominance], dtype=np.float32)


def train_research(manifest_path: Path = MANIFEST_PATH) -> dict[str, float]:
    """Fit the research backend's two valence thresholds on the validation
    split. No learned parameters otherwise -- the audeering encoder is used
    zero-shot, exactly as published."""
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run `make data` first")

    manifest = pd.read_csv(manifest_path, dtype={"clip_id": str, "speaker_id": str})
    assert_speaker_disjoint(manifest)

    val = manifest[manifest["split"] == "val"]
    if len(val) == 0:
        raise ValueError("no rows with split='val'")

    vad_encoder = AudeeringVADEncoder()
    cache = extract_and_cache(
        val, _VADEmbedAdapter(vad_encoder), VAD_CACHE_PATH, repo_root=REPO_ROOT
    )
    vad_matrix, _ids = embeddings_matrix(val, cache)
    valences = vad_matrix[:, 0]  # (valence, arousal, dominance) per _VADEmbedAdapter
    y_val = [Sentiment(s) for s in val["prosody_sentiment"]]

    thresholds = fit_valence_thresholds(valences, y_val)
    val_pred = [
        Sentiment.NEGATIVE
        if v <= thresholds.low
        else (Sentiment.POSITIVE if v >= thresholds.high else Sentiment.NEUTRAL)
        for v in valences
    ]
    val_uar = uar(y_val, val_pred)
    val_f1 = macro_f1(y_val, val_pred)
    logger.info(
        "research backend: fitted thresholds low=%.3f high=%.3f, val UAR=%.3f macroF1=%.3f",
        thresholds.low,
        thresholds.high,
        val_uar,
        val_f1,
    )

    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(
        json.dumps({"low": thresholds.low, "high": thresholds.high}, indent=2)
    )
    logger.info("saved thresholds to %s", THRESHOLDS_PATH)

    return {"val_uar": val_uar, "val_macro_f1": val_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["permissive", "research"], default="permissive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.backend == "permissive":
        train_permissive()
    elif args.backend == "research":
        train_research()


if __name__ == "__main__":
    main()
