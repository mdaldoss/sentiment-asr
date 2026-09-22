"""Embedding cache: clip_id -> pooled encoder vector, persisted to disk.

Extracting WavLM embeddings for thousands of clips on CPU is slow (~0.15-0.2s
per clip) and is re-run often during development, so this cache is
resumable: `extract_and_cache` reads whatever's already on disk, only
computes embeddings for clip_ids not yet present, and checkpoints
periodically so an interrupted run loses at most one checkpoint interval of
work rather than starting over.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ssa.audio import load_clip
from ssa.types import AudioClip

logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 200


class Embedder(Protocol):
    def embed(self, clip: AudioClip) -> np.ndarray: ...


def load_cache(path: Path) -> dict[str, np.ndarray]:
    """Load a cache file if present, else return an empty dict."""
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return {clip_id: data[clip_id] for clip_id in data.files}


def save_cache(cache: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # np.savez silently appends ".npz" to a string/Path target that doesn't
    # already end in it -- passing an open file handle avoids that, so `tmp`
    # (ending in ".tmp") is actually what gets written.
    with open(tmp, "wb") as f:
        np.savez(f, **cache)
    tmp.rename(path)  # atomic-ish: never leaves a half-written cache file


def extract_and_cache(
    manifest: pd.DataFrame,
    embedder: Embedder,
    cache_path: Path,
    *,
    repo_root: Path = Path("."),
) -> dict[str, np.ndarray]:
    """Ensure `cache_path` holds an embedding for every clip_id in `manifest`.

    Returns the full cache dict (existing + newly computed). Safe to call
    repeatedly -- already-cached clips are never re-extracted.
    """
    cache = load_cache(cache_path)
    todo = manifest[~manifest["clip_id"].isin(cache.keys())]
    logger.info("%d/%d clips already cached, extracting %d", len(cache), len(manifest), len(todo))

    since_checkpoint = 0
    for row in todo.itertuples(index=False):
        clip = load_clip(repo_root / row.path, clip_id=row.clip_id)
        cache[row.clip_id] = embedder.embed(clip)
        since_checkpoint += 1

        if since_checkpoint >= CHECKPOINT_EVERY:
            save_cache(cache, cache_path)
            since_checkpoint = 0
            logger.info("checkpoint: %d/%d done", len(cache), len(manifest))

    if since_checkpoint > 0 or not cache_path.exists():
        save_cache(cache, cache_path)

    return cache


def embeddings_matrix(
    manifest: pd.DataFrame, cache: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    """Stack cached embeddings for `manifest`'s rows, in manifest order.

    Raises KeyError (with the missing ids) rather than silently skipping a
    clip whose embedding wasn't extracted.
    """
    missing = [cid for cid in manifest["clip_id"] if cid not in cache]
    if missing:
        raise KeyError(f"{len(missing)} clip_id(s) missing from cache, e.g. {missing[:5]}")
    X = np.stack([cache[cid] for cid in manifest["clip_id"]])
    return X, list(manifest["clip_id"])
