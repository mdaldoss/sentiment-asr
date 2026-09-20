"""Tests for the per-speaker normalisation experiment.

The leave-one-out statistics are computed in closed form from each group's
sums rather than by refitting per clip. That is the right thing for runtime
and the wrong thing for trust: an algebra slip there produces numbers that
look entirely plausible and quietly change the experiment's answer. So the
closed form is checked against the definition it is supposed to implement.

The divide-by-zero guard gets the same treatment, because a constant feature
inside one speaker is not hypothetical -- `voiced_fraction` saturates at 1.0
on clean short clips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.normalise_speaker import (
    _group_stats,
    _loo_stats,
    as_predictions,
    group_sizes,
    normalise,
    score,
)
from ssa.types import Sentiment


class TestGroupStats:
    def test_mean_and_std_are_per_group(self) -> None:
        X = np.array([[0.0], [10.0], [100.0], [300.0]])
        groups = np.array(["a", "a", "b", "b"])
        mean, std = _group_stats(X, groups)
        assert mean.ravel().tolist() == [5.0, 5.0, 200.0, 200.0]
        assert std.ravel().tolist() == [5.0, 5.0, 100.0, 100.0]


class TestLeaveOneOutStats:
    def test_matches_the_naive_definition(self) -> None:
        """The closed form must equal actually deleting the row."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 4))
        groups = np.array(["a"] * 7 + ["b"] * 5)

        mean, std = _loo_stats(X, groups)
        for i in range(len(X)):
            others = X[(groups == groups[i]) & (np.arange(len(X)) != i)]
            assert mean[i] == pytest.approx(others.mean(axis=0), abs=1e-9)
            assert std[i] == pytest.approx(others.std(axis=0), abs=1e-9)

    def test_a_clip_does_not_inflate_its_own_spread(self) -> None:
        """The point of the LOO variant: an outlier normalised against the
        group it belongs to is damped by its own contribution to the std.
        Excluded from it, the same clip reads as the outlier it is."""
        X = np.array([[0.0], [0.0], [0.0], [0.0], [9.0]])
        groups = np.array(["a"] * 5)
        z_pooled = normalise(X, groups, "speaker_z")
        z_loo = normalise(X, groups, "speaker_z_loo")
        assert abs(z_loo[4, 0]) > abs(z_pooled[4, 0])

    def test_lone_outlier_is_not_flattened_to_average(self) -> None:
        """Regression test for a sign-of-the-answer bug. When a speaker's
        other clips are constant on a feature, the leave-one-out spread is
        zero; a plain divide-by-zero guard then maps the one clip that
        differs -- the most extreme in the group -- to 0.0, which is what an
        exactly average clip gets. The scale must fall back to the pooled
        std instead, and only a feature constant across the whole group may
        legitimately end at zero."""
        X = np.array([[0.0], [0.0], [0.0], [0.0], [9.0]])
        groups = np.array(["a"] * 5)
        z = normalise(X, groups, "speaker_z_loo")
        assert np.isfinite(z).all()
        assert abs(z[4, 0]) > 1.0, "the outlier must not read as average"
        assert abs(z[4, 0]) > abs(z[0, 0])

    def test_tiny_group_falls_back_instead_of_dividing_by_one(self) -> None:
        X = np.array([[1.0], [3.0]])
        groups = np.array(["solo", "solo"])
        mean, std = _loo_stats(X, groups)
        assert mean.ravel().tolist() == [2.0, 2.0]
        assert std.ravel().tolist() == [1.0, 1.0]
        assert np.isfinite(normalise(X, groups, "speaker_z_loo")).all()


class TestNormalise:
    def test_raw_is_the_identity(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert normalise(X, np.array(["a", "b"]), "raw") is X

    def test_speaker_z_centres_each_speaker_separately(self) -> None:
        """A per-speaker offset -- a louder microphone, a lower voice -- is
        exactly what this is supposed to remove."""
        X = np.array([[1.0], [3.0], [101.0], [103.0]])
        groups = np.array(["a", "a", "b", "b"])
        z = normalise(X, groups, "speaker_z")
        assert z.ravel().tolist() == [-1.0, 1.0, -1.0, 1.0]

    def test_dataset_z_keeps_the_speaker_offset(self) -> None:
        """The contrast that makes dataset_z worth running: it pools, so a
        between-speaker difference survives it."""
        X = np.array([[1.0], [3.0], [101.0], [103.0]])
        groups = np.array(["a", "a", "b", "b"])
        z = normalise(X, groups, "dataset_z")
        assert z[0, 0] < -0.5 and z[3, 0] > 0.5

    def test_constant_feature_becomes_zero_not_infinity(self) -> None:
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        groups = np.array(["a", "a", "a"])
        for scheme in ("dataset_z", "speaker_z", "speaker_z_loo"):
            z = normalise(X, groups, scheme)
            assert np.isfinite(z).all(), scheme
            assert z[:, 0].tolist() == [0.0, 0.0, 0.0], scheme

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scheme"):
            normalise(np.zeros((2, 1)), np.array(["a", "a"]), "per_phase_of_moon")


class TestGroupSizes:
    def test_reports_undersized_groups(self) -> None:
        groups = np.array(["a"] * 10 + ["b"] * 2)
        info = group_sizes(groups)
        assert info["n_groups"] == 2
        assert info["min_group_size"] == 2
        assert info["undersized_groups"] == {"b": 2}


class TestAsPredictions:
    def test_probs_cover_all_sentiments_and_sum_to_one(self) -> None:
        """A classifier fitted on a split missing a class exposes only the
        classes it saw; Prediction requires all three, so the gap is filled
        with zero mass rather than left to raise at scoring time."""
        labels = np.array(["negative", "positive"])
        proba = np.array([[0.7, 0.3], [0.1, 0.9]])
        classes = np.array(["negative", "positive"])
        preds = as_predictions(labels, proba, classes)

        assert [p.sentiment for p in preds] == [Sentiment.NEGATIVE, Sentiment.POSITIVE]
        for p in preds:
            assert set(p.probs) == set(Sentiment)
            assert sum(p.probs.values()) == pytest.approx(1.0)
        assert preds[0].probs[Sentiment.NEUTRAL] == 0.0
        assert preds[1].confidence == pytest.approx(0.9)


class TestScore:
    """The experiment's scoring path, end to end on a hand-checkable set.

    Worth a test of its own: this is the only place in the script where a
    number reaches the output file, and the first run died inside it on an
    attribute error after every fit had already been paid for.
    """

    @staticmethod
    def _df() -> pd.DataFrame:
        # Two incongruent clips (words positive, tone negative) and one
        # congruent neutral clip.
        return pd.DataFrame(
            {
                "prosody_sentiment": ["negative", "negative", "neutral"],
                "text_sentiment": ["positive", "positive", "neutral"],
                "is_congruent": [False, False, True],
            }
        )

    @staticmethod
    def _preds(labels: list[str]) -> list:
        import numpy as np

        classes = np.array(["negative", "neutral", "positive"])
        onehot = np.array([(classes == label).astype(float) for label in labels])
        return as_predictions(np.array(labels), onehot, classes)

    def test_perfect_tone_following_scores_psi_one(self) -> None:
        s = score(self._preds(["negative", "negative", "neutral"]), self._df())
        assert s["n_clips"] == 3
        assert s["psi_contested"] == pytest.approx(1.0)
        assert s["psi_strict"] == pytest.approx(1.0)
        assert s["is_degenerate"] is False

    def test_reading_the_words_scores_psi_zero(self) -> None:
        """Both incongruent clips predicted as their TEXT sentiment."""
        s = score(self._preds(["positive", "positive", "neutral"]), self._df())
        assert s["psi_contested"] == pytest.approx(0.0)
        assert s["psi_strict"] == pytest.approx(0.0)

    def test_collapsed_model_is_flagged(self) -> None:
        s = score(self._preds(["negative"] * 3), self._df())
        assert s["top_class"] == "negative"
        assert s["top_class_share"] == pytest.approx(1.0)
        assert s["is_degenerate"] is True
