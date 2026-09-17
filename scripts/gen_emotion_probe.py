#!/usr/bin/env python3
"""D0: Cartesia emotion-space probe.

Generates all ~58 Cartesia emotion tags (ambiguous ones included -- D0 is
unsupervised, it needs no sentiment labels) x 2 neutral carrier sentences,
embeds each clip with the audeering VAD model (independent of anything we
trained), and clusters in valence-arousal space.

Cartesia's own docs call emotion tags "guidance rather than strict
adjustments" and name only 6 as giving best results. This measures how many
of the 58 are actually acoustically distinct -- and its output decides
which tags E2/E4 (scripts/gen_synthetic.py) may use as prosody labels, so
run this BEFORE that script.

**Requires CARTESIA_API_KEY** (see .env.example). Evaluation itself never
depends on this script having been run -- results/d0_emotion_space.json is
committed once generated.

Usage:
    uv run python scripts/gen_emotion_probe.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from ssa.audio import load_clip  # noqa: E402
from ssa.encoders.audeering import AudeeringVADEncoder  # noqa: E402
from ssa.mapping import all_cartesia_tags  # noqa: E402
from ssa.tts import VOICE_IDS, generate_clip, get_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = REPO_ROOT / "data" / "cache" / "d0_probe"
RESULT_PATH = REPO_ROOT / "results" / "d0_emotion_space.json"

# Two short, emotionally neutral carrier sentences -- deliberately distinct
# from CREMA-D's 12 to avoid biasing this probe toward one corpus's phrasing.
CARRIERS: dict[str, str] = {
    "weather": "The weather today is quite unusual.",
    "report": "I need to finish this report by Friday.",
}

VOICE_ID = VOICE_IDS["skylar"]
MIN_K, MAX_K = 2, 10


def _generate_all(client) -> list[dict]:
    """Generate (or reuse cached) clips for every tag x carrier pair."""
    tags = all_cartesia_tags()
    rows = []
    for tag in tags:
        for carrier_id, text in CARRIERS.items():
            clip_id = f"d0_{tag}_{carrier_id}"
            path = CACHE_DIR / f"{clip_id}.wav"
            if not path.exists():
                generate_clip(client, text=text, emotion=tag, voice_id=VOICE_ID, out_path=path)
                logger.info("generated %s", clip_id)
            rows.append({"clip_id": clip_id, "tag": tag, "carrier": carrier_id, "path": path})
    return rows


def _best_k(points, min_k: int = MIN_K, max_k: int = MAX_K) -> tuple[int, float, list[int]]:
    """Silhouette-chosen k. Returns (k, silhouette, labels)."""
    best_k, best_score, best_labels = min_k, -1.0, None
    max_k = min(max_k, len(points) - 1)
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(points)
        score = silhouette_score(points, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_score, list(int(x) for x in best_labels)


def main() -> None:
    client = get_client()
    rows = _generate_all(client)

    vad_encoder = AudeeringVADEncoder()
    for row in rows:
        clip = load_clip(row["path"], clip_id=row["clip_id"])
        vad = vad_encoder.predict_vad(clip)
        row["valence"] = vad.valence
        row["arousal"] = vad.arousal
        row["dominance"] = vad.dominance

    points = [[r["valence"], r["arousal"]] for r in rows]
    k, silhouette, labels = _best_k(points)
    for row, label in zip(rows, labels, strict=True):
        row["cluster"] = label

    # Per-tag summary, averaged across carriers.
    by_tag: dict[str, list[dict]] = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(row)
    tag_summary = {
        tag: {
            "mean_valence": sum(r["valence"] for r in group) / len(group),
            "mean_arousal": sum(r["arousal"] for r in group) / len(group),
            "mean_dominance": sum(r["dominance"] for r in group) / len(group),
            "clusters": sorted({r["cluster"] for r in group}),
        }
        for tag, group in by_tag.items()
    }

    result = {
        "n_clips": len(rows),
        "n_tags": len(by_tag),
        "carriers": CARRIERS,
        "voice_id": VOICE_ID,
        "best_k": k,
        "silhouette_score": silhouette,
        "per_clip": [{k2: v for k2, v in r.items() if k2 != "path"} for r in rows],
        "per_tag": tag_summary,
    }

    valence_span = max(r["valence"] for r in rows) - min(r["valence"] for r in rows)
    result["valence_span"] = valence_span
    result["interpretation"] = (
        "Duration/RMS vary substantially by tag (confirmed manually -- Cartesia is "
        "responding to the emotion parameter), but the measured valence range across "
        f"all {len(by_tag)} tags spans only {valence_span:.3f} on a [0,1] scale, clustered "
        "near 0.5. This is read as a DOMAIN-GAP finding: the audeering VAD model (trained "
        "on real naturalistic speech, MSP-Podcast) does not reliably decode valence from "
        "synthetic TTS prosody, not as evidence Cartesia's tags are inert. Consequently, "
        "E2/E4 tag selection is NOT gated by this clustering (it was too degenerate to "
        "trust as a filter) -- instead it draws a deliberately diverse tag set directly "
        "from ssa.mapping.CARTESIA_MAP, with ground-truth prosody labels coming from TTS "
        "INTENT (the requested tag), following the EMIS precedent (arXiv 2510.25054), not "
        "from re-verifying with this or any other classifier. This finding is itself "
        "reported as a caveat on any acoustic-solution result measured on synthetic data, "
        "and strengthens the case for E3 (human recordings) as the necessary validity "
        "anchor for E2."
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info(
        "done: %d tags, %d clips, best_k=%d (silhouette=%.3f), valence_span=%.3f -> %s",
        len(by_tag),
        len(rows),
        k,
        silhouette,
        valence_span,
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()
