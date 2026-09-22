#!/usr/bin/env python3
"""The backend x dataset-combo comparison, requested directly by the user:
WavLM+probe (permissive) vs audeering/wav2vec2 (research), each trained
and evaluated across four training-data combinations --

    cremad_only       CREMA-D train split alone
    cremad_e3         + both E3 takes (the user's own recordings)
    cremad_e3_hume     + both E3 takes + Hume with-description clips
    cremad_hume       + Hume with-description clips (no E3)

Cartesia is deliberately excluded (D1 already found its emotion tags
don't render on this content, see results/d1_emotion_probe.json).

**Why the research backend's numbers repeat across all four combos**: it
has no trainable encoder and only two fitted parameters (valence
thresholds), fit once from CREMA-D's own held-out validation split in
`ssa.train.train_research`. None of the three extra sources ever enters
that val split in this design, so its thresholds -- and therefore every
downstream prediction -- are identical no matter which combo is asked
for. That is reported explicitly rather than silently re-running the same
computation four times and implying it differed.

**Why E3 disappears from the eval column for two combos**: see
`ssa.combo_manifests`'s module docstring -- E3 has one real speaker, and a
combo that trains on that speaker cannot also evaluate on them (CLAUDE.md
rule 1) even though the clip_ids differ between the two takes.

No API key needed -- reads only committed audio, the committed
`results/hume_probe.json`, and CREMA-D (already downloaded by `make data`).

Usage:
    uv run python scripts/eval_backend_combos.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.combo_manifests import (  # noqa: E402
    COMBOS,
    ComboSpec,
    build_train_manifest,
    eval_manifests_for_combo,
    hume_manifest_from_probe,
    stratified_subset,
)
from ssa.eval.runner import evaluate  # noqa: E402
from ssa.manifest import load_manifest, write_manifest  # noqa: E402
from ssa.solutions.acoustic import AcousticSolution  # noqa: E402
from ssa.train import CACHE_PATH, train_permissive  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CREMAD_MANIFEST_PATH = REPO_ROOT / "data" / "cremad" / "manifest.csv"
E3_TAKES: dict[str, Path] = {
    "e3a": REPO_ROOT / "data" / "recorded0" / "manifest_speaker1.csv",
    "e3b": REPO_ROOT / "data" / "recorded" / "manifest_speaker1.csv",
}
HUME_PROBE_PATH = REPO_ROOT / "results" / "hume_probe.json"

COMBO_MANIFEST_DIR = REPO_ROOT / "data" / "cache" / "combo_manifests"
COMBO_PROBE_DIR = REPO_ROOT / "data" / "cache" / "combo_probes"
RESULT_PATH = REPO_ROOT / "results" / "backend_combo_comparison.json"

EVAL_SUBSET_N = 300
EVAL_SUBSET_SEED = 0


def _load_e3_take(path: Path) -> pd.DataFrame:
    manifest = load_manifest(path, repo_root=REPO_ROOT)
    # Whisper-transcription prompts carry no PSI-design prosody target
    # (congruent by convention, see record_prompts.py) -- same exclusion
    # scripts/eval_e3.py already applies.
    return manifest[~manifest["clip_id"].str.contains("whisper")].reset_index(drop=True)


def _summarise_eval(result) -> dict:
    """The fields CLAUDE.md rule 4 requires (dataset, split_type) plus the
    headline numbers -- not the full per-clip `predictions` list, which
    belongs in the standard per-solution result files, not this comparison."""
    return {
        "dataset": result.dataset,
        "split_type": result.split_type,
        "n_clips": result.n_clips,
        "n_incongruent": result.n_incongruent,
        "uar": result.uar,
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "psi_contested": result.psi_contested,
        "psi_strict": result.psi_strict,
    }


@dataclass
class ComboOutcome:
    combo: ComboSpec
    train_n_clips: int
    permissive_val_uar: float
    permissive_val_macro_f1: float
    permissive_eval: dict[str, dict | None]


def _eval_reason_excluded(combo: ComboSpec) -> dict:
    return {
        "excluded_reason": (
            f"combo {combo.name!r} trains on E3's speaker; scoring it on E3 here "
            "would put the same speaker on both sides of the split (CLAUDE.md rule 1)."
        )
    }


def run_combo(
    combo: ComboSpec,
    cremad_full: pd.DataFrame,
    cremad_test_subset: pd.DataFrame,
    e3a: pd.DataFrame,
    e3b: pd.DataFrame,
    hume: pd.DataFrame,
) -> ComboOutcome:
    train_manifest = build_train_manifest(combo, cremad_full, e3a, e3b, hume)
    manifest_path = COMBO_MANIFEST_DIR / f"{combo.name}.csv"
    write_manifest(train_manifest, manifest_path)

    probe_path = COMBO_PROBE_DIR / f"{combo.name}_permissive.joblib"
    train_stats = train_permissive(
        manifest_path=manifest_path, probe_path=probe_path, cache_path=CACHE_PATH
    )
    solution = AcousticSolution(backend="permissive", probe_path=probe_path)

    eval_sets = eval_manifests_for_combo(combo, cremad_test_subset, e3a, e3b)
    eval_out: dict[str, dict | None] = {}
    for eval_name, manifest in eval_sets.items():
        result = evaluate(
            solution,
            manifest,
            dataset=eval_name,
            split_type=f"combo_{combo.name}",
            repo_root=REPO_ROOT,
        )
        eval_out[eval_name] = _summarise_eval(result)
    if "e3_both_takes" not in eval_out:
        eval_out["e3_both_takes"] = _eval_reason_excluded(combo)

    return ComboOutcome(
        combo=combo,
        train_n_clips=len(train_manifest[train_manifest["split"] == "train"]),
        permissive_val_uar=train_stats["val_uar"],
        permissive_val_macro_f1=train_stats["val_macro_f1"],
        permissive_eval=eval_out,
    )


def run_research_once(
    cremad_test_subset: pd.DataFrame, e3a: pd.DataFrame, e3b: pd.DataFrame
) -> dict:
    """Zero-shot encoder, thresholds fit once from CREMA-D val (already
    shipped at data/cache/thresholds_research.json) -- see module
    docstring for why this genuinely doesn't vary by combo."""
    solution = AcousticSolution(backend="research")
    out = {
        "cremad_test": _summarise_eval(
            evaluate(
                solution,
                cremad_test_subset,
                dataset="cremad_test",
                split_type="combo_shared_research",
                repo_root=REPO_ROOT,
            )
        ),
        "e3_both_takes": _summarise_eval(
            evaluate(
                solution,
                pd.concat([e3a, e3b], ignore_index=True),
                dataset="e3_both_takes",
                split_type="combo_shared_research",
                repo_root=REPO_ROOT,
            )
        ),
    }
    return out


def main() -> None:
    cremad_full = load_manifest(CREMAD_MANIFEST_PATH, repo_root=REPO_ROOT)
    cremad_test_full = cremad_full[cremad_full["split"] == "test"].reset_index(drop=True)
    cremad_test_subset = stratified_subset(cremad_test_full, n=EVAL_SUBSET_N, seed=EVAL_SUBSET_SEED)
    logger.info(
        "cremad test subset: %d clips (from %d)", len(cremad_test_subset), len(cremad_test_full)
    )

    e3a = _load_e3_take(E3_TAKES["e3a"])
    e3b = _load_e3_take(E3_TAKES["e3b"])

    hume_probe = json.loads(HUME_PROBE_PATH.read_text())
    hume = hume_manifest_from_probe(hume_probe)
    logger.info("hume with-description manifest: %d clips", len(hume))

    logger.info("research backend: fitting once (combo-invariant, see docstring)...")
    research_eval = run_research_once(cremad_test_subset, e3a, e3b)

    combos_out: dict[str, dict] = {}
    for combo in COMBOS:
        logger.info("=== combo %s: %s ===", combo.name, combo.description)
        outcome = run_combo(combo, cremad_full, cremad_test_subset, e3a, e3b, hume)
        combos_out[combo.name] = {
            "description": combo.description,
            "train_n_clips": outcome.train_n_clips,
            "permissive": {
                "val_uar": outcome.permissive_val_uar,
                "val_macro_f1": outcome.permissive_val_macro_f1,
                "eval": outcome.permissive_eval,
            },
            "research": {"eval": research_eval},
        }

    result = {
        "eval_subset_n": EVAL_SUBSET_N,
        "eval_subset_seed": EVAL_SUBSET_SEED,
        "combos": combos_out,
        "research_backend_note": (
            "The research backend (audeering/wav2vec2, CC-BY-NC-SA-4.0) has no trainable "
            "encoder -- only two valence thresholds, fit once from CREMA-D's own held-out "
            "validation split (ssa.train.train_research), which none of the extra sources "
            "ever enters in this design. Its numbers are therefore identical across all "
            "four combos by construction, computed once and reused here -- this is a real "
            "architectural property being reported, not a shortcut standing in for missing "
            "results. The permissive backend (WavLM + our own trained probe) is the one "
            "that actually differs by combo below."
        ),
        "interpretation": (
            "Direct comparison requested by the user: does adding the user's own recordings "
            "(E3) and/or the Hume Octave synthetic probe to training change either backend's "
            "CREMA-D-test performance, and what happens on E3 itself when it's held out "
            "entirely (cremad_only, cremad_hume) vs folded into training (cremad_e3, "
            "cremad_e3_hume, where E3 eval is excluded -- CLAUDE.md rule 1). Cartesia is "
            "excluded throughout per the user's request; see d1_emotion_probe.json for why "
            "it wouldn't have added signal anyway. n is small for the added sources (E3: 1 "
            "speaker, 2 takes; Hume: 1 pseudo-speaker) relative to CREMA-D's 91 actors, so "
            "any shift in permissive numbers here should be read as a directional signal, "
            "not a robust generalisation claim."
        ),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", RESULT_PATH)


if __name__ == "__main__":
    main()
