"""Train/val/test splitting.

The single most important correctness property in this repo lives here: **no
speaker may appear on both sides of a split**. Random splits on acted emotion
corpora put the same actor in train and test, and the model learns to recognise
the speaker rather than the emotion. Reported accuracy then overstates
generalisation by a wide margin.

`random_split` is deliberately kept -- clearly named, documented as leaky, and
never used to train the shipped model. The headline table in the report compares
it against `speaker_disjoint_split` to quantify the gap. Do not delete it as
dead code, and do not let it become a default anywhere.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRAIN, VAL, TEST = "train", "val", "test"


class SplitError(AssertionError):
    """Raised when a split violates speaker disjointness."""


def speaker_disjoint_split(
    df: pd.DataFrame,
    *,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign train/val/test so that speaker sets are pairwise disjoint.

    Speakers are shuffled deterministically and allocated whole to a split, so
    the resulting clip-count fractions only approximate the requested ones. That
    is the correct trade: exact fractions are worthless if they leak.

    Returns a copy with the `split` column populated.
    """
    if not 0 < test_frac + val_frac < 1:
        raise ValueError(f"test_frac + val_frac must be in (0, 1), got {test_frac + val_frac}")

    out = df.copy()
    speakers = np.array(sorted(out["speaker_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n_test + n_val >= n:
        raise ValueError(f"{n} speakers is too few for test_frac={test_frac}, val_frac={val_frac}")

    assignment = {}
    for spk in speakers[:n_test]:
        assignment[spk] = TEST
    for spk in speakers[n_test : n_test + n_val]:
        assignment[spk] = VAL
    for spk in speakers[n_test + n_val :]:
        assignment[spk] = TRAIN

    out["split"] = out["speaker_id"].map(assignment)
    assert_speaker_disjoint(out)

    counts = out["split"].value_counts().to_dict()
    logger.info(
        "speaker-disjoint split: %d speakers -> train=%d val=%d test=%d clips",
        n,
        counts.get(TRAIN, 0),
        counts.get(VAL, 0),
        counts.get(TEST, 0),
    )
    return out


def random_split(
    df: pd.DataFrame,
    *,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Split clips at random, ignoring speaker identity.

    LEAKY BY CONSTRUCTION. This exists only so the report can quantify how much
    speaker leakage inflates results. Never use it to train a model we present
    as generalising.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(out))

    n_test = round(len(out) * test_frac)
    n_val = round(len(out) * val_frac)

    split = np.array([TRAIN] * len(out), dtype=object)
    split[idx[:n_test]] = TEST
    split[idx[n_test : n_test + n_val]] = VAL
    out["split"] = split

    logger.warning("random_split used -- results from this split are leaky by construction")
    return out


def assert_speaker_disjoint(df: pd.DataFrame) -> None:
    """Raise SplitError if any speaker appears in more than one split.

    Call this at the top of every training run. It is cheap and it is the
    difference between a trustworthy number and a meaningless one.
    """
    by_split = {s: set(g["speaker_id"]) for s, g in df.groupby("split")}
    for a, b in ((TRAIN, TEST), (TRAIN, VAL), (VAL, TEST)):
        overlap = by_split.get(a, set()) & by_split.get(b, set())
        if overlap:
            raise SplitError(
                f"{len(overlap)} speaker(s) appear in both {a!r} and {b!r}: "
                f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
            )
