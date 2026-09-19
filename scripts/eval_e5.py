#!/usr/bin/env python3
"""Evaluate every solution on E5, the Hume incongruence set.

E5 is the dataset this project was designed around and never had: 2 of
every 3 clips have words and delivery deliberately disagreeing, on a
synthetic voice where the delivery was set independently of the text (see
scripts/gen_hume_dataset.py). That makes **PSI the headline metric here**,
not UAR -- on a set built to be two-thirds contradictory, a model that
quietly reads the transcript is supposed to score badly, and the number
that catches it is PSI.

What each solution's PSI means on E5:

  - **A (lexical)** can only read words, so it should sit near 0.0. It is
    the floor, and if it doesn't land there the labels are wrong.
  - **B (acoustic)** never sees a transcript, so it is the one that can
    score high -- if Hume's delivery control actually produced audible
    prosody, which the 40-clip probe said it does.
  - **C (fusion)** should land between them, since it sees both.

E5 is **eval-only**. It is synthetic, and CLAUDE.md rule 8 forbids
presenting within-corpus numbers as generalisation; it is here to measure
prosody sensitivity under contradiction, not to train anything.

No API key needed -- reads the committed manifest and audio.

Usage:
    uv run python scripts/eval_e5.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.eval.runner import evaluate, results_path  # noqa: E402
from ssa.manifest import load_manifest, summarise  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.solutions.fusion import DEFAULT_PARAMS, FusionParams, FusionSolution  # noqa: E402
from ssa.solutions.lexical import LexicalSolution  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH = REPO_ROOT / "data" / "hume_e5" / "manifest.csv"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "e5_summary.json"
FUSION_PARAMS_PATH = REPO_ROOT / "data" / "cache" / "fusion_params.json"


def build_solutions() -> dict[str, object]:
    lexical = LexicalSolution()
    permissive = AcousticSolution(backend="permissive")
    research = AcousticSolution(backend="research")
    params = (
        FusionParams.from_json(FUSION_PARAMS_PATH)
        if FUSION_PARAMS_PATH.exists()
        else DEFAULT_PARAMS
    )
    return {
        "A_lexical": lexical,
        "B_permissive": permissive,
        "B_research": research,
        "C_fusion": FusionSolution(lexical, permissive, params),
    }


def per_voice_breakdown(manifest: pd.DataFrame) -> dict[str, int]:
    return {str(v): int(n) for v, n in manifest["voice_id"].value_counts().items()}


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found -- run `make gen-hume-e5` first")

    manifest = load_manifest(MANIFEST_PATH, repo_root=REPO_ROOT)
    logger.info("E5: %s", summarise(manifest))

    solutions = build_solutions()
    summary: dict[str, object] = {
        "dataset": "e5_hume",
        "n_clips": len(manifest),
        "n_incongruent": int((~manifest["is_congruent"].astype(bool)).sum()),
        "n_voices": int(manifest["voice_id"].nunique()),
        "clips_per_voice": per_voice_breakdown(manifest),
        "solutions": {},
    }

    for label, solution in solutions.items():
        result = evaluate(
            solution, manifest, dataset="e5_hume", split_type="all", repo_root=REPO_ROOT
        )
        result.to_json(results_path(RESULTS_DIR, solution.name, "e5_hume", "all"))
        summary["solutions"][label] = {
            "solution": result.solution,
            "uar": result.uar,
            "macro_f1": result.macro_f1,
            "accuracy": result.accuracy,
            "psi_contested": result.psi_contested,
            "psi_strict": result.psi_strict,
        }

    summary["interpretation"] = (
        "E5 is the incongruence set E2 could not be: two thirds of clips have words and "
        "delivery deliberately disagreeing, with delivery set independently of the text via "
        "Hume Octave's `description` field (validated first at 40 clips in "
        "results/hume_probe.json: 0.5 vs 0.2 chance emotion recoverability with that field, "
        "exactly chance without it). PSI is the metric to read here, not UAR: the gold label "
        "is prosody_sentiment, so on a set built to contradict the transcript, a model that "
        "reads words scores near 0 by construction and a model that hears tone scores high. "
        "Solution A is the floor and exists to prove the labels behave. Synthetic and "
        "eval-only: one TTS vendor, a small number of voices, so this measures prosody "
        "sensitivity under controlled contradiction, not performance on real speech -- E3 "
        "remains the only real-speaker evidence in this repo."
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", SUMMARY_PATH)
    for label, s in summary["solutions"].items():
        logger.info("%-14s UAR=%.3f  PSI_contested=%.3f", label, s["uar"], s["psi_contested"])


if __name__ == "__main__":
    main()
