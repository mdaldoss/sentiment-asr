"""Shared fixtures. Prefer tiny deterministic frames over downloading data."""

from __future__ import annotations

import pandas as pd
import pytest

from ssa.types import Sentiment


@pytest.fixture
def toy_manifest() -> pd.DataFrame:
    """6 clips, 3 speakers, mixed congruence. Enough to exercise every invariant."""
    rows = [
        # clip_id, speaker, text_sent,        prosody_sent
        ("c1", "spk1", Sentiment.NEUTRAL, Sentiment.NEGATIVE),
        ("c2", "spk1", Sentiment.NEUTRAL, Sentiment.NEUTRAL),
        ("c3", "spk2", Sentiment.POSITIVE, Sentiment.NEGATIVE),
        ("c4", "spk2", Sentiment.POSITIVE, Sentiment.POSITIVE),
        ("c5", "spk3", Sentiment.NEGATIVE, Sentiment.POSITIVE),
        ("c6", "spk3", Sentiment.NEGATIVE, Sentiment.NEGATIVE),
    ]
    df = pd.DataFrame(
        {
            "clip_id": [r[0] for r in rows],
            "path": [f"data/fake/{r[0]}.wav" for r in rows],
            "speaker_id": [r[1] for r in rows],
            "source": ["synthetic"] * len(rows),
            "text": ["the book is on the table"] * len(rows),
            "text_sentiment": [r[2].value for r in rows],
            "prosody_sentiment": [r[3].value for r in rows],
            "emotion_tag": [""] * len(rows),
            "voice_id": [""] * len(rows),
            "split": [""] * len(rows),
        }
    )
    df["is_congruent"] = df["text_sentiment"] == df["prosody_sentiment"]
    return df
