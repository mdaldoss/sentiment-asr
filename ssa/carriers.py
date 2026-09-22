"""Carrier sentences: the text spoken in every generated/recorded clip.

Deliberately dependency-free (stdlib + `ssa.types` only). Both
`scripts/gen_synthetic.py` (E2/E4/E5) and `scripts/record_prompts.py` (E3)
import their carrier text from here, so that recording -- which needs no
API, no key, and no network -- never pulls in `cartesia` transitively.
Before this module existed, `record_prompts.py` imported `CARRIERS` from
`gen_synthetic.py`, which imports `ssa.tts` at module level, which imports
`cartesia` at module level -- so `make record` failed with
`ModuleNotFoundError: No module named 'cartesia'` on a machine that only
ran `make setup` (which installs `.[dev,record]`, not the opt-in
`.[generate]` extra). See the Makefile and pyproject.toml.
"""

from __future__ import annotations

from ssa.types import Sentiment

# E2's carrier text: 5 sentences per text-sentiment, chosen for clear
# lexical polarity so text_sentiment is unambiguous to a human reader (and
# to Solution A's text classifier) independent of how they're spoken.
CARRIERS: dict[Sentiment, list[str]] = {
    Sentiment.POSITIVE: [
        "I'm thrilled about the good news today.",
        "This turned out to be the best day of my life.",
        "I really appreciate everything you've done for me.",
        "What a wonderful surprise this has been.",
        "I'm so proud of how this all turned out.",
    ],
    Sentiment.NEUTRAL: [
        "The meeting is scheduled for three o'clock.",
        "I need to pick up groceries later today.",
        "The report is due next Tuesday afternoon.",
        "Please turn off the lights when you leave.",
        "The train departs from platform four.",
    ],
    Sentiment.NEGATIVE: [
        "This is terrible, I can't believe it happened.",
        "I'm so disappointed in how this went.",
        "Everything about this situation is awful.",
        "I hate how things turned out this time.",
        "This has been a complete disaster for us.",
    ],
}

# D1/E5's five emotions -- the user's proposed set, richer than E2's three
# sentiment-only prosody tags. Canonical order used throughout D1.
EMOTIONS: tuple[str, ...] = ("happy", "sad", "angry", "calm", "frustrated")

# `calm` is globally AMBIGUOUS in ssa.mapping (it reads positive in a
# wellbeing sense but is acoustically near-neutral -- see that module's
# docstring). For THIS dataset only, it is scoped to sentiment=neutral: a
# documented, local override, not a change to the global ambiguous-tag
# scheme. The other four map onto ssa.mapping.CARTESIA_MAP unchanged.
D1_EMOTION_SENTIMENT: dict[str, Sentiment] = {
    "happy": Sentiment.POSITIVE,
    "sad": Sentiment.NEGATIVE,
    "angry": Sentiment.NEGATIVE,
    "calm": Sentiment.NEUTRAL,
    "frustrated": Sentiment.NEGATIVE,
}

# D1's shared neutral text -- IDENTICAL wording across all 5 emotions at a
# given length, so any acoustic difference between emotions in this
# condition can only come from prosody, never from the words. This is the
# condition the project actually wants (CLAUDE.md: prosody_sentiment is the
# gold label) -- D1 measures whether Cartesia can render it at all, since
# Cartesia's own docs say emotion tags "only work when the emotion is
# consistent with the transcript."
D1_NEUTRAL_TEXT: dict[str, str] = {
    "short": "I need to check tomorrow's schedule.",
    "long": (
        "I would like to check what time my appointment is on Thursday, "
        "and whether I need to bring anything with me."
    ),
}

# D1's per-emotion congruent text -- words and intended prosody agree,
# which is the condition Cartesia's docs describe as necessary for its
# emotion tags to work. Elderly-to-voice-assistant themed per the user's
# request; the "long" forms double as E5's carrier content if D1 finds
# neutral text doesn't render (E5's design is gated on D1's result).
D1_CONGRUENT_TEXT: dict[str, dict[str, str]] = {
    "happy": {
        "short": "I'm so happy you called me back today!",
        "long": (
            "My granddaughter is coming to visit on Sunday, and I have "
            "been looking forward to it all month."
        ),
    },
    "sad": {
        "short": "I've been feeling quite lonely lately.",
        "long": (
            "Nobody has telephoned me all week, and the house feels very "
            "quiet without anyone else here."
        ),
    },
    "angry": {
        "short": "This machine never understands me at all!",
        "long": (
            "I have told this machine the same thing three times now, and "
            "it keeps giving me the wrong answer, which makes me quite angry."
        ),
    },
    "calm": {
        "short": "I'm just relaxing here with my tea.",
        "long": (
            "There is no rush with any of this at all, I am just sitting here with a cup of tea."
        ),
    },
    "frustrated": {
        "short": "I've asked you this three times now.",
        "long": (
            "I have asked you three times already and you still are not "
            "understanding me, it is getting quite tiresome."
        ),
    },
}

# Emotion-adjusted speed multipliers for D1/E5's "speed=adjusted" condition,
# within Cartesia's documented 0.6-1.5 bound. Deliberately not touching
# volume (0.5-2.0) -- the user's explicit call was to engineer speed only.
# If this ships, it is labelled as engineered in the report -- the
# no-speed condition stays in as the honest baseline.
D1_SPEED_ADJUSTED: dict[str, float] = {
    "happy": 1.15,
    "angry": 1.15,
    "frustrated": 1.10,
    "sad": 0.85,
    "calm": 0.85,
}
