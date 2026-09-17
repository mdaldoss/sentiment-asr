"""Emotion label -> Sentiment mapping.

This module encodes one of the project's hard rules: **ambiguous emotions are
never silently bucketed**. Assigning `surprised` to positive (a common shortcut)
injects label noise into exactly the cells this project measures. We exclude
them instead, and say so in the report.

Excluded tags are still usable in D0, the unsupervised emotion-space probe,
which needs no labels at all.
"""

from __future__ import annotations

from ssa.types import Sentiment


class AmbiguousEmotionError(ValueError):
    """Raised when a tag has no defensible single sentiment valence."""


class UnknownEmotionError(ValueError):
    """Raised for a tag absent from the scheme. Never fall back to a default."""


# CREMA-D's six categories map cleanly: there is no `surprise` class, which is
# a genuine convenience of this corpus over RAVDESS.
CREMA_D_MAP: dict[str, Sentiment] = {
    "ANG": Sentiment.NEGATIVE,
    "DIS": Sentiment.NEGATIVE,
    "FEA": Sentiment.NEGATIVE,
    "SAD": Sentiment.NEGATIVE,
    "HAP": Sentiment.POSITIVE,
    "NEU": Sentiment.NEUTRAL,
}

# Cartesia's documented emotion vocabulary, grouped by sentiment valence.
CARTESIA_MAP: dict[str, Sentiment] = {
    # positive
    "happy": Sentiment.POSITIVE,
    "excited": Sentiment.POSITIVE,
    "enthusiastic": Sentiment.POSITIVE,
    "elated": Sentiment.POSITIVE,
    "euphoric": Sentiment.POSITIVE,
    "triumphant": Sentiment.POSITIVE,
    "content": Sentiment.POSITIVE,
    "peaceful": Sentiment.POSITIVE,
    "serene": Sentiment.POSITIVE,
    "grateful": Sentiment.POSITIVE,
    "affectionate": Sentiment.POSITIVE,
    "trust": Sentiment.POSITIVE,
    "proud": Sentiment.POSITIVE,
    "confident": Sentiment.POSITIVE,
    "sympathetic": Sentiment.POSITIVE,
    # negative
    "angry": Sentiment.NEGATIVE,
    "mad": Sentiment.NEGATIVE,
    "outraged": Sentiment.NEGATIVE,
    "frustrated": Sentiment.NEGATIVE,
    "agitated": Sentiment.NEGATIVE,
    "threatened": Sentiment.NEGATIVE,
    "disgusted": Sentiment.NEGATIVE,
    "contempt": Sentiment.NEGATIVE,
    "envious": Sentiment.NEGATIVE,
    "sad": Sentiment.NEGATIVE,
    "dejected": Sentiment.NEGATIVE,
    "melancholic": Sentiment.NEGATIVE,
    "disappointed": Sentiment.NEGATIVE,
    "hurt": Sentiment.NEGATIVE,
    "guilty": Sentiment.NEGATIVE,
    "rejected": Sentiment.NEGATIVE,
    "anxious": Sentiment.NEGATIVE,
    "panicked": Sentiment.NEGATIVE,
    "alarmed": Sentiment.NEGATIVE,
    "scared": Sentiment.NEGATIVE,
    "insecure": Sentiment.NEGATIVE,
    "resigned": Sentiment.NEGATIVE,
    "bored": Sentiment.NEGATIVE,
    "tired": Sentiment.NEGATIVE,
    # neutral
    "neutral": Sentiment.NEUTRAL,
    "hesitant": Sentiment.NEUTRAL,
    "confused": Sentiment.NEUTRAL,
    "distant": Sentiment.NEUTRAL,
    "apologetic": Sentiment.NEUTRAL,
}

# Tags with no agreed valence. `calm` is here on purpose: it reads positive in a
# wellbeing sense but is acoustically near-neutral, and conflating the two is
# how you build a model that calls every quiet senior "content".
AMBIGUOUS_TAGS: frozenset[str] = frozenset(
    {
        "surprised",
        "amazed",
        "nostalgic",
        "wistful",
        "sarcastic",
        "ironic",
        "mysterious",
        "determined",
        "calm",
        "contemplative",
        "anticipation",
        "curious",
        "skeptical",
        "flirtatious",
    }
)

_SCHEMES: dict[str, dict[str, Sentiment]] = {
    "crema_d": CREMA_D_MAP,
    "cartesia": CARTESIA_MAP,
}


def map_emotion(tag: str, scheme: str) -> Sentiment:
    """Map an emotion tag to a Sentiment.

    Raises:
        AmbiguousEmotionError: the tag has no defensible single valence.
        UnknownEmotionError: the tag is not in the scheme.

    Never returns a default. That is the point.
    """
    if scheme not in _SCHEMES:
        raise UnknownEmotionError(f"unknown scheme {scheme!r}; have {sorted(_SCHEMES)}")
    if tag in AMBIGUOUS_TAGS:
        raise AmbiguousEmotionError(
            f"{tag!r} has no agreed sentiment valence and is excluded from labelled sets. "
            "It may still be used in the unsupervised D0 probe."
        )
    table = _SCHEMES[scheme]
    if tag not in table:
        raise UnknownEmotionError(f"{tag!r} not in scheme {scheme!r}")
    return table[tag]


def labelled_tags(scheme: str) -> list[str]:
    """Tags usable as prosody labels, i.e. everything mappable and unambiguous."""
    return sorted(set(_SCHEMES[scheme]) - AMBIGUOUS_TAGS)


def all_cartesia_tags() -> list[str]:
    """Every documented tag, ambiguous ones included -- the D0 probe set."""
    return sorted(set(CARTESIA_MAP) | AMBIGUOUS_TAGS)
