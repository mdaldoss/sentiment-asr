"""The evaluation harness: run a Solution over a manifest, serialise every
number to disk.

CLAUDE.md's rule here: every number that reaches the report comes from one
of these JSON files. report.py only ever renders what's on disk -- it never
recomputes a metric. That is what makes the numbers in the write-up
auditable: re-running `make eval` with the same code and data reproduces
the exact same file.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ssa.audio import load_clip
from ssa.eval.metrics import (
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    psi_contested,
    psi_strict,
    uar,
)
from ssa.types import Sentiment, Solution

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    solution: str
    dataset: str
    split_type: str
    n_clips: int
    n_incongruent: int
    uar: float
    macro_f1: float
    accuracy: float
    ece: float
    psi_contested: float
    psi_strict: float
    confusion_matrix: list[list[int]]
    confusion_labels: list[str]
    abstention_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    git_sha: str
    timestamp: str
    predictions: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False))

    @classmethod
    def from_json(cls, path: Path) -> EvalResult:
        return cls(**json.loads(path.read_text()))


def evaluate(
    solution: Solution,
    manifest: pd.DataFrame,
    *,
    dataset: str,
    split_type: str,
    repo_root: Path = Path("."),
) -> EvalResult:
    """Run `solution` over every row of `manifest`, compute every metric this
    project reports, and return the result (call .to_json() to persist it).
    """
    if len(manifest) == 0:
        raise ValueError("cannot evaluate an empty manifest")

    clips = [
        load_clip(repo_root / row.path, clip_id=row.clip_id)
        for row in manifest.itertuples(index=False)
    ]
    predictions = solution.predict_batch(clips)
    if len(predictions) != len(manifest):
        raise ValueError(
            f"solution {solution.name!r} returned {len(predictions)} predictions "
            f"for {len(manifest)} clips"
        )

    y_true = [Sentiment(s) for s in manifest["prosody_sentiment"]]
    y_pred = [p.sentiment for p in predictions]
    probs_matrix = np.stack([p.prob_vector() for p in predictions])

    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred, strict=True)]))
    abstention_rate = float(np.mean([p.abstained for p in predictions]))
    latencies = np.array([p.latency_ms for p in predictions], dtype=np.float64)
    n_incongruent = int((~manifest["is_congruent"].astype(bool)).sum())

    result = EvalResult(
        solution=solution.name,
        dataset=dataset,
        split_type=split_type,
        n_clips=len(manifest),
        n_incongruent=n_incongruent,
        uar=uar(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred),
        accuracy=accuracy,
        ece=expected_calibration_error(probs_matrix, y_true),
        psi_contested=psi_contested(predictions, manifest),
        psi_strict=psi_strict(predictions, manifest),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        confusion_labels=[s.value for s in Sentiment.ordered()],
        abstention_rate=abstention_rate,
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        git_sha=_git_sha(),
        timestamp=datetime.now(UTC).isoformat(),
        predictions=[
            {
                "clip_id": row.clip_id,
                "prosody_sentiment": row.prosody_sentiment,
                "text_sentiment": row.text_sentiment,
                "is_congruent": bool(row.is_congruent),
                "predicted_sentiment": pred.sentiment.value,
                "confidence": pred.confidence,
                "abstained": pred.abstained,
                "latency_ms": pred.latency_ms,
                "transcript": pred.transcript,
            }
            for row, pred in zip(manifest.itertuples(index=False), predictions, strict=True)
        ],
    )
    logger.info(
        "%s on %s/%s: UAR=%.3f macroF1=%.3f PSI_contested=%.3f (n=%d, %d incongruent)",
        solution.name,
        dataset,
        split_type,
        result.uar,
        result.macro_f1,
        result.psi_contested,
        result.n_clips,
        result.n_incongruent,
    )
    return result


def results_path(results_dir: Path, solution_name: str, dataset: str, split_type: str) -> Path:
    """Deterministic filename for a result, sanitised for the filesystem."""
    safe_solution = solution_name.replace("/", "_").replace(":", "_")
    return results_dir / f"{safe_solution}__{dataset}__{split_type}.json"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"
