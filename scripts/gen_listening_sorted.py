#!/usr/bin/env python3
"""report/listening_sorted.html: every D1 (Cartesia) and E3 (human) clip in
one table, sorted by a detection score, so a listener can check the
extremes -- "does the top of the ranking actually sound more emotional than
the bottom?" -- without listening to all ~100 clips in an arbitrary order.

Detection score: the research (audeering) model's valence, z-scored within
its own source set (D1 / E3 take0 / E3 take1 have different scales and
recording conditions, so pooling raw valence across them would conflate
"source is different" with "emotion is different"), then signed by the
intended sentiment: +z if intended positive, -z if intended negative,
-|z| if intended neutral (a neutral clip scored as strongly polarised
either way is equally "wrong"). Sorting by this score puts clips the
detector is confident and (if the score is high) probably correct about at
the top, and clips it is confident and probably wrong about at the bottom.

Requires results/d1_emotion_probe.json and results/e3{a,b}_measurements.csv
(scripts/gen_emotion_probe_d1.py and scripts/eval_e3.py). No model loading,
no API key -- pure post-processing of already-computed numbers.

Usage:
    uv run python scripts/gen_listening_sorted.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.carriers import D1_EMOTION_SENTIMENT  # noqa: E402
from ssa.report import render_models_section  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = REPO_ROOT / "report" / "listening_sorted.html"

_SENTIMENT_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _load_d1_rows() -> pd.DataFrame:
    d1 = json.loads((RESULTS_DIR / "d1_emotion_probe.json").read_text())
    rows = []
    for clip in d1["per_clip"]:
        valence = clip.get("m_research_valence")
        if valence is None:
            continue
        rows.append(
            {
                "source": "D1 (Cartesia)",
                "clip_id": clip["clip_id"],
                "audio_path": f"../data/probes/d1/{clip['clip_id']}.wav",
                "intended": D1_EMOTION_SENTIMENT[clip["emotion"]].value,
                "detail": (
                    f"{clip['emotion']} / {clip['text_condition']} / "
                    f"{clip['length']} / speed={clip['speed_condition']}"
                ),
                "research_valence": valence,
                "permissive_predicted": clip.get("m_permissive_predicted_sentiment"),
            }
        )
    return pd.DataFrame(rows)


def _load_e3_rows(label: str, take_dir: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{label}_measurements.csv"
    if not path.exists():
        logger.warning("%s not found -- run scripts/eval_e3.py first; skipping", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for row in df.itertuples(index=False):
        # clip_id is e.g. "e3a_speaker1_positive_0_negative" -- the actual
        # wav on disk kept its original "e3_..." name (only the manifest's
        # clip_id column was prefixed for cross-take uniqueness).
        original_name = "e3_" + row.clip_id.split("_", 1)[1]
        rows.append(
            {
                "source": f"E3 human ({label})",
                "clip_id": row.clip_id,
                "audio_path": f"../{take_dir}/{original_name}.wav",
                "intended": row.intended,
                "detail": (
                    f'"{row.carrier[:40]}..."' if len(row.carrier) > 40 else f'"{row.carrier}"'
                ),
                "research_valence": row.research_valence,
                "permissive_predicted": row.permissive_predicted_sentiment,
            }
        )
    return pd.DataFrame(rows)


def build_scored_table() -> pd.DataFrame:
    frames = [
        _load_d1_rows(),
        _load_e3_rows("e3a", "data/recorded0"),
        _load_e3_rows("e3b", "data/recorded"),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError(
            "no source data found -- run gen_emotion_probe_d1.py and eval_e3.py first"
        )
    df = pd.concat(frames, ignore_index=True)

    # z-score within each source set, then sign by intended sentiment.
    df["z"] = df.groupby("source")["research_valence"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
    )
    sign = df["intended"].map(_SENTIMENT_SIGN)
    df["detection_score"] = np.where(sign == 0.0, -df["z"].abs(), sign * df["z"])
    return df.sort_values("detection_score", ascending=False).reset_index(drop=True)


def render(df: pd.DataFrame) -> str:
    def row_html(r) -> str:
        return (
            "<tr>"
            f"<td>{r.rank}</td>"
            f"<td>{r.source}</td>"
            f"<td>{r.intended}</td>"
            f'<td class="detail">{r.detail}</td>'
            f'<td><audio controls src="{r.audio_path}"></audio></td>'
            f"<td>{r.detection_score:+.2f}</td>"
            f"<td>{r.research_valence:.2f}</td>"
            f"<td>{r.permissive_predicted}</td>"
            "</tr>"
        )

    df = df.copy()
    df["rank"] = range(1, len(df) + 1)
    rows_html = "\n".join(row_html(r) for r in df.itertuples(index=False))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sorted listening check -- D1 + E3</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: .25rem; }}
  p {{ color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #eee; vertical-align: middle; }}
  th {{ position: sticky; top: 0; background: #fff; }}
  td.detail {{ max-width: 220px; font-family: ui-monospace, monospace; font-size: 0.8rem; color: #555; }}
  audio {{ height: 28px; width: 180px; }}
  .unmeasured {{ background: #f5f4f1; border-left: 3px solid #888; padding: .6rem .9rem; border-radius: 4px; font-style: italic; }}
  tr:nth-child(-n+10) {{ background: #f0fff4; }}
  tr:nth-last-child(-n+10) {{ background: #fff5f0; }}
</style>
</head>
<body>
<h1>Sorted listening check: {len(df)} clips, D1 (Cartesia) + E3 (human), ranked by detection score</h1>
<p>Detection score = the research model's valence, z-scored within its own source set, signed by
the intended sentiment (positive/negative/neutral). High score = the detector is confident and
(if correct) should sound clearly emotional; low score = confident in the wrong direction, or a
neutral clip read as strongly polarised. <strong>Top 10 rows (green) and bottom 10 (orange) are
the ones worth listening to</strong> -- do the top ones genuinely sound more emotional than the
bottom ones? If yes, the detector is doing its job and low scorers are inexpressive source audio.
If the ranking sounds arbitrary, the detector -- not the audio -- is the problem.</p>

<h2>Which model actually extracts prosody/emotion (the "Valence" column above)</h2>
{render_models_section()}

<table>
<tr><th>#</th><th>Source</th><th>Intended</th><th>Detail</th><th>Audio</th>
<th>Score</th><th>Valence</th><th>Permissive pred.</th></tr>
{rows_html}
</table>
</body>
</html>
"""


def main() -> None:
    df = build_scored_table()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render(df))
    logger.info("wrote %d rows -> %s", len(df), OUT_PATH)


if __name__ == "__main__":
    main()
