"""Command-line entry point.

Defaults to the recommended solution (fusion). The comparison between
solutions is the *evidence* in the report; this CLI is the *deliverable*
-- one clip in, one sentiment out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ssa.audio import load_clip
from ssa.solutions.acoustic import AcousticSolution
from ssa.solutions.fusion import DEFAULT_PARAMS, FusionParams, FusionSolution
from ssa.solutions.lexical import LexicalSolution
from ssa.types import Prediction, Solution

REPO_ROOT = Path(__file__).resolve().parent.parent
FUSION_PARAMS_PATH = REPO_ROOT / "data" / "cache" / "fusion_params.json"


def _build_solution(name: str, backend: str) -> Solution:
    if name == "lexical":
        return LexicalSolution()
    if name == "acoustic":
        return AcousticSolution(backend=backend)
    if name == "fusion":
        lexical = LexicalSolution()
        acoustic = AcousticSolution(backend=backend)
        params = (
            FusionParams.from_json(FUSION_PARAMS_PATH)
            if FUSION_PARAMS_PATH.exists()
            else DEFAULT_PARAMS
        )
        return FusionSolution(lexical, acoustic, params)
    raise ValueError(f"unknown solution {name!r}")


def _print_human(pred: Prediction) -> None:
    print(f"Sentiment:   {pred.sentiment.value}")
    print(f"Confidence:  {pred.confidence:.2f}")
    if pred.abstained:
        print("             (low confidence -- treat this prediction with caution)")
    probs = ", ".join(f"{s.value}={p:.2f}" for s, p in pred.probs.items())
    print(f"Distribution: {probs}")
    if pred.transcript is not None:
        print(f'Transcript:  "{pred.transcript}"')
    if pred.vad is not None:
        print(
            f"VAD:         valence={pred.vad.valence:.2f} "
            f"arousal={pred.vad.arousal:.2f} dominance={pred.vad.dominance:.2f}"
        )
    print(f"Latency:     {pred.latency_ms:.0f} ms")
    print(f"Solution:    {pred.solution}")


def _print_json(pred: Prediction) -> None:
    payload = {
        "sentiment": pred.sentiment.value,
        "confidence": pred.confidence,
        "abstained": pred.abstained,
        "probs": {s.value: p for s, p in pred.probs.items()},
        "transcript": pred.transcript,
        "vad": (
            {
                "valence": pred.vad.valence,
                "arousal": pred.vad.arousal,
                "dominance": pred.vad.dominance,
            }
            if pred.vad is not None
            else None
        ),
        "latency_ms": pred.latency_ms,
        "solution": pred.solution,
    }
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssa", description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="path to an audio file")
    parser.add_argument(
        "--solution",
        default="fusion",
        choices=["lexical", "acoustic", "fusion"],
        help="which solution to run (default: the recommended one)",
    )
    parser.add_argument(
        "--backend",
        default="permissive",
        choices=["permissive", "research"],
        help="acoustic backend; 'research' is CC-BY-NC-SA-4.0, research use only",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 1

    solution = _build_solution(args.solution, args.backend)
    clip = load_clip(args.audio)
    prediction = solution.predict(clip)

    if args.json:
        _print_json(prediction)
    else:
        _print_human(prediction)
    return 0


if __name__ == "__main__":
    sys.exit(main())
