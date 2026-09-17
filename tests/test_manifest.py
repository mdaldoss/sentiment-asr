"""Manifest schema invariants."""

from __future__ import annotations

import pandas as pd
import pytest

from ssa.manifest import COLUMNS, ManifestError, add_congruence, summarise, validate_manifest


def test_schema(toy_manifest: pd.DataFrame) -> None:
    validate_manifest(toy_manifest)


def test_all_columns_present(toy_manifest: pd.DataFrame) -> None:
    assert set(COLUMNS) <= set(toy_manifest.columns)


def test_duplicate_clip_id_rejected(toy_manifest: pd.DataFrame) -> None:
    df = pd.concat([toy_manifest, toy_manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(ManifestError, match="duplicate clip_id"):
        validate_manifest(df)


def test_empty_speaker_id_rejected(toy_manifest: pd.DataFrame) -> None:
    df = toy_manifest.copy()
    df.loc[0, "speaker_id"] = ""
    with pytest.raises(ManifestError, match="empty speaker_id"):
        validate_manifest(df)


def test_bad_sentiment_value_rejected(toy_manifest: pd.DataFrame) -> None:
    df = toy_manifest.copy()
    df.loc[0, "prosody_sentiment"] = "ecstatic"
    with pytest.raises(ManifestError, match="outside Sentiment"):
        validate_manifest(df)


def test_inconsistent_congruence_rejected(toy_manifest: pd.DataFrame) -> None:
    df = toy_manifest.copy()
    df.loc[0, "is_congruent"] = not df.loc[0, "is_congruent"]
    with pytest.raises(ManifestError, match="is_congruent inconsistent"):
        validate_manifest(df)


def test_add_congruence_matches_manual(toy_manifest: pd.DataFrame) -> None:
    df = add_congruence(toy_manifest.drop(columns=["is_congruent"]))
    validate_manifest(df)


def test_summarise_counts_incongruent(toy_manifest: pd.DataFrame) -> None:
    assert "3 incongruent" in summarise(toy_manifest)
