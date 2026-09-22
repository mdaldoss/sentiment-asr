"""Tests for the class-skew robustness experiment.

The experiment's whole claim rests on the draws actually being skewed, and
skewed *within each speaker* -- per-speaker normalisation sees one speaker
at a time, so a set that is imbalanced overall while every speaker inside it
stays balanced would test nothing at all and would look identical in the
output. That is the failure this file exists to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.normalise_skew import CLASSES, MIN_PER_CLASS, aggregate, skewed_indices


def _manifest(n_per_class: int = 9, speakers: tuple[str, ...] = ("s1", "s2")) -> pd.DataFrame:
    rows = [
        {"speaker_id": sp, "prosody_sentiment": cls}
        for sp in speakers
        for cls in CLASSES
        for _ in range(n_per_class)
    ]
    return pd.DataFrame(rows)


class TestSkewedIndices:
    def test_balanced_keep_returns_everything(self) -> None:
        df = _manifest()
        idx = skewed_indices(df, "negative", 1.0, np.random.default_rng(0))
        assert len(idx) == len(df)

    def test_majority_class_dominates_at_strong_skew(self) -> None:
        df = _manifest()
        idx = skewed_indices(df, "positive", 0.22, np.random.default_rng(0))
        counts = df.iloc[idx]["prosody_sentiment"].value_counts()
        assert counts.idxmax() == "positive"
        assert counts["positive"] == 18  # both speakers keep all 9
        assert counts.max() / counts.sum() > 0.6

    def test_skew_holds_inside_every_speaker(self) -> None:
        """The point of the whole design. A draw that is skewed in aggregate
        but balanced within each speaker would leave per-speaker means
        unchanged and silently measure nothing."""
        df = _manifest()
        idx = skewed_indices(df, "negative", 0.22, np.random.default_rng(1))
        sub = df.iloc[idx]
        for speaker, block in sub.groupby("speaker_id"):
            counts = block["prosody_sentiment"].value_counts()
            assert counts.idxmax() == "negative", speaker
            assert counts.max() / counts.sum() > 0.6, speaker

    def test_no_class_is_emptied(self) -> None:
        """A class with zero clips makes UAR undefined for it, which would
        quietly change what the reported mean is an average of."""
        df = _manifest(n_per_class=4)
        for majority in CLASSES:
            idx = skewed_indices(df, majority, 0.01, np.random.default_rng(2))
            counts = df.iloc[idx]["prosody_sentiment"].value_counts()
            assert set(counts.index) == set(CLASSES)
            for cls in CLASSES:
                if cls != majority:
                    assert counts[cls] >= MIN_PER_CLASS * 2  # two speakers

    def test_draws_are_deterministic_given_a_seed(self) -> None:
        df = _manifest()
        a = skewed_indices(df, "neutral", 0.45, np.random.default_rng(7))
        b = skewed_indices(df, "neutral", 0.45, np.random.default_rng(7))
        assert a.tolist() == b.tolist()

    def test_different_seeds_give_different_draws(self) -> None:
        df = _manifest()
        a = skewed_indices(df, "neutral", 0.45, np.random.default_rng(7))
        b = skewed_indices(df, "neutral", 0.45, np.random.default_rng(8))
        assert a.tolist() != b.tolist()

    def test_returned_positions_are_sorted_and_unique(self) -> None:
        df = _manifest()
        idx = skewed_indices(df, "negative", 0.45, np.random.default_rng(3))
        assert idx.tolist() == sorted(set(idx.tolist()))


class TestAggregate:
    def test_nan_draws_are_dropped_not_propagated(self) -> None:
        """PSI is nan on a draw with no contested clips. One such draw must
        not turn the whole cell's mean into nan -- but the count it was
        averaged over has to shrink visibly."""
        out = aggregate([0.5, float("nan"), 0.7])
        assert out["mean"] == pytest.approx(0.6)
        assert out["n_draws"] == 2

    def test_all_nan_reports_zero_draws(self) -> None:
        out = aggregate([float("nan"), float("nan")])
        assert np.isnan(out["mean"])
        assert out["n_draws"] == 0
