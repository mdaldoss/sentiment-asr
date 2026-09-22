"""Tests for the E6 evaluation's bootstrap and its leakage guards.

The bootstrap exists to stop a 20-clip difference being read as a finding,
so it has to be right in the direction that matters: an interval that comes
out too narrow would defeat the entire purpose of adding it. These check
that it widens as the set shrinks, that it brackets the point estimate, and
that it degrades to `nan` rather than to a confident-looking number when
there is nothing to resample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.eval_zurich import SUSPICIOUS_UAR, bootstrap_ci


def _df(prosody: list[str], text: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prosody_sentiment": prosody,
            "text_sentiment": text,
            "is_congruent": [p == t for p, t in zip(prosody, text, strict=True)],
        }
    )


def _balanced(n_per_class: int) -> tuple[pd.DataFrame, list[str]]:
    classes = ["negative", "neutral", "positive"]
    prosody = [c for c in classes for _ in range(n_per_class)]
    # Words always disagree with delivery, so every clip is contested.
    text = [classes[(classes.index(c) + 1) % 3] for c in prosody]
    return _df(prosody, text), prosody


class TestBootstrapCI:
    def test_brackets_a_perfect_score(self) -> None:
        df, y = _balanced(8)
        ci = bootstrap_ci(np.array(y), df, n_boot=500)
        assert ci["uar_ci95"]["hi"] == pytest.approx(1.0)
        assert ci["uar_ci95"]["lo"] > 0.9

    def test_interval_widens_as_the_set_shrinks(self) -> None:
        """The property the E6 write-up depends on. A 20-clip set must not
        produce an interval as tight as a 200-clip one, or the CI provides
        false reassurance exactly where it is most needed."""
        rng = np.random.default_rng(0)
        widths = []
        for n_per_class in (4, 40):
            df, y = _balanced(n_per_class)
            # Same noise rate at both sizes, so only n differs.
            y_pred = [
                c if rng.random() > 0.4 else rng.choice(["negative", "neutral", "positive"])
                for c in y
            ]
            ci = bootstrap_ci(np.array(y_pred), df, n_boot=800)
            widths.append(ci["uar_ci95"]["hi"] - ci["uar_ci95"]["lo"])
        assert widths[0] > widths[1] * 1.5, f"small-n interval not wider: {widths}"

    def test_is_deterministic_for_a_given_seed(self) -> None:
        df, y = _balanced(6)
        a = bootstrap_ci(np.array(y), df, n_boot=300, seed=7)
        b = bootstrap_ci(np.array(y), df, n_boot=300, seed=7)
        assert a == b

    def test_psi_interval_is_nan_when_nothing_is_contested(self) -> None:
        """All-congruent input has no words-vs-tone decision to measure. An
        undefined PSI must stay undefined rather than collapsing to a number
        somebody could quote."""
        df = _df(["negative"] * 6, ["negative"] * 6)
        ci = bootstrap_ci(np.array(["negative"] * 6), df, n_boot=200)
        assert np.isnan(ci["psi_contested_ci95"]["lo"])
        assert np.isnan(ci["psi_contested_ci95"]["hi"])

    def test_psi_of_a_transcript_reader_is_pinned_near_zero(self) -> None:
        """Predicting the TEXT label on every contested clip is PSI 0."""
        df, _ = _balanced(8)
        ci = bootstrap_ci(df["text_sentiment"].to_numpy(), df, n_boot=500)
        assert ci["psi_contested_ci95"]["hi"] == pytest.approx(0.0)


class TestLeakageCeiling:
    def test_ceiling_matches_the_rest_of_the_repo(self) -> None:
        """One number, three places (under two spellings -- `ssa/train.py`
        calls it SUSPICIOUS_UAR_THRESHOLD). If they drift apart the guard
        stops meaning what the docstrings say it means, and the drift would
        be invisible: each module would still look internally consistent."""
        from scripts.train_prosodic import SUSPICIOUS_UAR as train_ceiling
        from ssa.train import SUSPICIOUS_UAR_THRESHOLD as ssa_ceiling

        assert SUSPICIOUS_UAR == train_ceiling == ssa_ceiling == 0.90
