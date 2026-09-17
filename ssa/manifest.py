"""The manifest: one normalised CSV schema for every dataset.

The evaluation harness only ever sees manifests, never raw dataset layouts. Any
new corpus is converted to this schema by a script in `scripts/`, and everything
downstream works unchanged.

The critical column pair is `text_sentiment` vs `prosody_sentiment`: the words
versus the delivery. Separating them is what makes the Prosody Sensitivity Index
computable at all. **`prosody_sentiment` is the gold label for evaluation.**
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ssa.types import Sentiment

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = (
    "clip_id",
    "path",
    "speaker_id",
    "source",
    "text",
    "text_sentiment",
    "prosody_sentiment",
    "emotion_tag",
    "voice_id",
    "is_congruent",
    "split",
)

VALID_SOURCES: frozenset[str] = frozenset({"crema_d", "synthetic", "recorded"})


class ManifestError(ValueError):
    """Raised when a manifest violates the schema."""


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    """Validate then write. Never write an invalid manifest."""
    validate_manifest(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, columns=list(COLUMNS))
    logger.info("wrote %d rows to %s", len(df), path)


def load_manifest(
    path: Path, *, validate: bool = True, repo_root: Path | None = None
) -> pd.DataFrame:
    """Load a manifest CSV. Validates by default."""
    df = pd.read_csv(path, dtype={"clip_id": str, "speaker_id": str, "voice_id": str})
    df["voice_id"] = df["voice_id"].fillna("")
    df["emotion_tag"] = df["emotion_tag"].fillna("")
    df["split"] = df["split"].fillna("")
    if validate:
        validate_manifest(df, repo_root=repo_root)
    return df


def validate_manifest(df: pd.DataFrame, *, repo_root: Path | None = None) -> None:
    """Raise ManifestError on any schema violation.

    Checks, in order of how badly each would corrupt results:
      1. all columns present
      2. clip_id unique
      3. speaker_id non-empty  (leakage protection depends on it)
      4. sentiment values inside the enum
      5. is_congruent consistent with the two sentiment columns
      6. referenced audio files exist  (only when repo_root is given)
    """
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ManifestError(f"missing columns: {sorted(missing)}")

    dupes = df["clip_id"][df["clip_id"].duplicated()].unique()
    if len(dupes):
        raise ManifestError(f"duplicate clip_id: {list(dupes[:5])}")

    blank_spk = df["speaker_id"].isna() | (df["speaker_id"].astype(str).str.strip() == "")
    if blank_spk.any():
        raise ManifestError(
            f"{blank_spk.sum()} row(s) have an empty speaker_id; "
            "speaker-disjoint splitting cannot protect against leakage without it"
        )

    valid = {s.value for s in Sentiment}
    for col in ("text_sentiment", "prosody_sentiment"):
        bad = set(df[col].dropna().unique()) - valid
        if bad:
            raise ManifestError(f"{col} has values outside Sentiment: {sorted(bad)}")

    bad_src = set(df["source"].unique()) - VALID_SOURCES
    if bad_src:
        raise ManifestError(f"unknown source(s): {sorted(bad_src)}")

    expected = df["text_sentiment"] == df["prosody_sentiment"]
    actual = df["is_congruent"].astype(bool)
    if not expected.equals(actual):
        n = int((expected != actual).sum())
        raise ManifestError(f"{n} row(s) have is_congruent inconsistent with the sentiment columns")

    if repo_root is not None:
        absent = [p for p in df["path"] if not (repo_root / p).exists()]
        if absent:
            raise ManifestError(f"{len(absent)} referenced file(s) missing, e.g. {absent[:3]}")


def add_congruence(df: pd.DataFrame) -> pd.DataFrame:
    """Derive `is_congruent` from the two sentiment columns."""
    out = df.copy()
    out["is_congruent"] = out["text_sentiment"] == out["prosody_sentiment"]
    return out


def summarise(df: pd.DataFrame) -> str:
    """One-line human summary, for logs and CLI output."""
    n_incong = int((~df["is_congruent"].astype(bool)).sum())
    return (
        f"{len(df)} clips | {df['speaker_id'].nunique()} speakers | "
        f"{n_incong} incongruent | sources={sorted(df['source'].unique())}"
    )
