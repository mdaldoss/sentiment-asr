"""Splitting invariants.

test_no_speaker_overlap is the single most important test in this repo: if it
fails, every reported number is inflated by speaker leakage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ssa.splits import SplitError, assert_speaker_disjoint, random_split, speaker_disjoint_split


def _many_speakers(n_speakers: int = 20, clips_each: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": [f"s{s}_c{c}" for s in range(n_speakers) for c in range(clips_each)],
            "speaker_id": [f"spk{s}" for s in range(n_speakers) for _ in range(clips_each)],
        }
    )


def test_no_speaker_overlap() -> None:
    df = speaker_disjoint_split(_many_speakers(), test_frac=0.2, val_frac=0.1, seed=0)
    by_split = {s: set(g["speaker_id"]) for s, g in df.groupby("split")}
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["val"].isdisjoint(by_split["test"])


def test_all_splits_non_empty() -> None:
    df = speaker_disjoint_split(_many_speakers(), seed=3)
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_split_is_deterministic_given_seed() -> None:
    a = speaker_disjoint_split(_many_speakers(), seed=7)
    b = speaker_disjoint_split(_many_speakers(), seed=7)
    pd.testing.assert_series_equal(a["split"], b["split"])


def test_different_seeds_give_different_splits() -> None:
    a = speaker_disjoint_split(_many_speakers(), seed=1)
    b = speaker_disjoint_split(_many_speakers(), seed=2)
    assert not a["split"].equals(b["split"])


def test_assert_speaker_disjoint_catches_leakage() -> None:
    df = pd.DataFrame(
        {"clip_id": ["a", "b"], "speaker_id": ["spk1", "spk1"], "split": ["train", "test"]}
    )
    with pytest.raises(SplitError, match="appear in both"):
        assert_speaker_disjoint(df)


def test_random_split_is_leaky_as_documented() -> None:
    """Not a bug -- random_split exists to quantify the leakage it causes."""
    df = random_split(_many_speakers(n_speakers=5, clips_each=20), seed=0)
    by_split = {s: set(g["speaker_id"]) for s, g in df.groupby("split")}
    assert by_split["train"] & by_split["test"], "expected leakage from random_split"


def test_too_few_speakers_raises() -> None:
    with pytest.raises(ValueError, match="too few"):
        speaker_disjoint_split(_many_speakers(n_speakers=2), test_frac=0.4, val_frac=0.4)
