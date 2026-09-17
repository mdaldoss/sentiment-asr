#!/usr/bin/env python3
"""Extract and cache WavLM-base embeddings for the CREMA-D manifest.

One-time cost (~20-25 min on a 4-core CPU for the full 7,442 clips) --
resumable via ssa.embeddings.extract_and_cache, so re-running after an
interruption only extracts what's missing.

Usage:
    uv run python scripts/extract_embeddings.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssa.embeddings import extract_and_cache  # noqa: E402
from ssa.encoders.wavlm import MODEL_ID, WavLMEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH = REPO_ROOT / "data" / "cremad" / "manifest.csv"
CACHE_PATH = REPO_ROOT / "data" / "cache" / "embeddings_wavlm-base.npz"


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found -- run `make data` first")

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"clip_id": str, "speaker_id": str})
    logger.info("loaded manifest: %d clips", len(manifest))

    encoder = WavLMEncoder()
    logger.info("encoder ready: %s on %s", MODEL_ID, encoder.device)

    t0 = time.time()
    cache = extract_and_cache(manifest, encoder, CACHE_PATH, repo_root=REPO_ROOT)
    elapsed_min = (time.time() - t0) / 60
    logger.info("done: %d embeddings cached in %.1f min -> %s", len(cache), elapsed_min, CACHE_PATH)


if __name__ == "__main__":
    main()
