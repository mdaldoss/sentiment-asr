#!/usr/bin/env python3
"""Render results/model_matrix.json as a Markdown table -> results/model_matrix.md.

Separate from the experiment script so re-reading the numbers never risks
re-fitting them. Every row carries its dataset, split type and n, because a
metric without its evaluation condition is meaningless here (CLAUDE.md rule 4).

Usage:
    uv run python scripts/render_model_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUMMARY_PATH = REPO_ROOT / "results" / "model_matrix.json"
OUT_PATH = REPO_ROOT / "results" / "model_matrix.md"

BACKEND_ORDER = ("research_valence_thresholds", "research_vad_head", "permissive", "prosody")
BACKEND_LABEL = {
    "research_valence_thresholds": "research (audeering, 2 thresholds)",
    "research_vad_head": "research (audeering VAD -> logreg)",
    "permissive": "permissive (WavLM + probe)",
    "prosody": "prosody (eGeMAPS+contour, logreg)",
}


def _fmt(x: object, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        f = float(x)
        return "nan" if f != f else f"{f:.{digits}f}"
    return str(x)


def _ci(ci: dict) -> str:
    return f"[{_fmt(ci.get('lo'), 2)}, {_fmt(ci.get('hi'), 2)}]" if ci else ""


def _row(label: str, name: str, s: dict) -> str:
    uar_ci = _ci(s.get("uar_ci95", {}))
    psi_ci = _ci(s.get("psi_contested_ci95", {}))
    flags = f"degenerate ({s['top_class']})" if s.get("is_degenerate") else "-"
    return (
        f"| {label} | {name} | {s['n_clips']} | {_fmt(s['uar'])} {uar_ci} | "
        f"{_fmt(s['macro_f1'])} | {_fmt(s['accuracy'])} | "
        f"{_fmt(s['psi_contested'])} {psi_ci} | {_fmt(s['psi_strict'])} | {flags} |"
    )


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"{SUMMARY_PATH} not found -- run scripts/train_model_matrix.py first")
    payload = json.loads(SUMMARY_PATH.read_text())

    lines: list[str] = [
        "# Model matrix: three backends x three data regimes",
        "",
        "Generated from `results/model_matrix.json` by `scripts/render_model_matrix.py`.",
        "Gold label is `prosody_sentiment` everywhere. UAR is the headline; accuracy",
        "appears only beside it. Chance UAR is 0.333 for three classes; PSI chance is",
        "0.5 (contested) and 0.333 (strict).",
        "",
    ]

    for cfg_name, cfg in payload["configurations"].items():
        lines += [
            f"## {cfg_name}",
            "",
            cfg["question"],
            "",
            f"- train: {cfg['n_train_clips']} clips / {cfg['n_train_speakers']} speakers "
            f"({cfg['train_sources']}), classes {cfg['train_class_counts']}",
            f"- val: {cfg['n_val_clips']} clips / {cfg['n_val_speakers']} speakers "
            "(threshold fitting and candidate selection only)",
            "",
            "| backend | eval set | n | UAR [95% CI] | macro-F1 | acc | PSI contested | "
            "PSI strict | flags |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for backend in BACKEND_ORDER:
            payload_b = cfg["backends"].get(backend)
            if payload_b is None:
                continue
            for name, s in payload_b["eval"].items():
                lines.append(_row(BACKEND_LABEL[backend], name, s))
        for name, meta in cfg["eval_sets"].items():
            b = meta["majority_baseline"]
            lines.append(
                f"| _majority baseline ({b['always_predicts']})_ | {name} | {meta['n_clips']} | "
                f"{_fmt(b['uar'])} | {_fmt(b['macro_f1'])} | {_fmt(b['accuracy'])} | "
                f"{_fmt(b['psi_contested'])} | {_fmt(b['psi_strict'])} | reference |"
            )
        lines += ["", "Notes:", ""]
        lines += [f"- {n}" for n in cfg["notes"]]

        prosody = cfg["backends"].get("prosody")
        if prosody:
            lines += [
                "",
                f"Prosody candidates (val UAR; headline is `{prosody['headline_candidate']}`, "
                f"val-selected would be `{prosody['val_selected_candidate']}`): "
                + ", ".join(
                    f"{k} {_fmt(v['val_uar'])}" for k, v in prosody["candidate_val_scores"].items()
                ),
            ]
        lines.append("")

    lines += ["## Method", "", payload["method"], "", "## Caveats", ""]
    lines += [f"- {c}" for c in payload["caveats"]]
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
