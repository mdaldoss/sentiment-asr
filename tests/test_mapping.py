"""Emotion mapping invariants: ambiguity is raised, never defaulted."""

from __future__ import annotations

import pytest

from ssa.mapping import (
    AMBIGUOUS_TAGS,
    CARTESIA_MAP,
    CREMA_D_MAP,
    AmbiguousEmotionError,
    UnknownEmotionError,
    all_cartesia_tags,
    labelled_tags,
    map_emotion,
)
from ssa.types import Sentiment


@pytest.mark.parametrize("tag", sorted(AMBIGUOUS_TAGS))
def test_ambiguous_emotions_excluded(tag: str) -> None:
    """Every ambiguous tag raises rather than silently getting a label."""
    with pytest.raises(AmbiguousEmotionError):
        map_emotion(tag, "cartesia")


def test_unknown_tag_raises_not_defaults() -> None:
    with pytest.raises(UnknownEmotionError):
        map_emotion("definitely_not_an_emotion", "cartesia")


def test_crema_d_maps_all_six() -> None:
    assert set(CREMA_D_MAP) == {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}
    assert map_emotion("ANG", "crema_d") is Sentiment.NEGATIVE
    assert map_emotion("HAP", "crema_d") is Sentiment.POSITIVE
    assert map_emotion("NEU", "crema_d") is Sentiment.NEUTRAL


def test_ambiguous_and_mapped_are_disjoint() -> None:
    """A tag cannot be both labelled and excluded."""
    assert not (set(CARTESIA_MAP) & AMBIGUOUS_TAGS)


def test_labelled_tags_excludes_ambiguous() -> None:
    assert not (set(labelled_tags("cartesia")) & AMBIGUOUS_TAGS)


def test_d0_probe_set_includes_ambiguous() -> None:
    """D0 is unsupervised, so it uses the full vocabulary."""
    assert AMBIGUOUS_TAGS <= set(all_cartesia_tags())


def test_cartesia_vocabulary_size() -> None:
    """Guards against silently dropping tags when editing the table."""
    assert len(all_cartesia_tags()) == len(CARTESIA_MAP) + len(AMBIGUOUS_TAGS)
