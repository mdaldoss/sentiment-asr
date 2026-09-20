"""Tests for the E6 (Zurich multi-speaker) build.

Two invariants here are load-bearing in a way the rest of the script is not.

The speaker mapping decides whether a model can be trained on E6 and scored
on E3 with the same human on both sides. That is CLAUDE.md rule 1, and the
only thing standing between the repo and a silent violation is one
dictionary entry, so it gets an explicit test naming the person's two ids.

The hand-assigned text valence decides `is_congruent`, and therefore decides
what PSI is computed over. A sentence missing from the table, or a table
drifting out of step with the manifest, would quietly move clips into the
"congruent" bucket where PSI ignores them -- shrinking the denominator
rather than raising an error.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_zurich import E3_SPEAKER_ID, SPEAKER_IDS, TEXT_SENTIMENT, speaker_id_for
from ssa.manifest import validate_manifest
from ssa.types import Sentiment


class TestSpeakerIdentity:
    def test_marco_keeps_the_e3_speaker_id(self) -> None:
        """The whole leakage guard in one assertion: E6's `marco` and E3's
        `speaker1` are the same person, so they must carry one id. If this
        ever becomes two ids, `assert_speaker_disjoint` stops being able to
        see the overlap and a model can train on E6 and be scored on E3."""
        assert speaker_id_for("marco_20260920_204857") == E3_SPEAKER_ID
        assert E3_SPEAKER_ID == "speaker1"

    def test_every_other_speaker_is_distinct(self) -> None:
        ids = [speaker_id_for(f"{prefix}_20260920_000000") for prefix in SPEAKER_IDS]
        assert len(set(ids)) == len(ids), "two E6 folders map to the same speaker id"

    def test_unknown_folder_raises_rather_than_inventing_an_id(self) -> None:
        """A new participant must stop the build. Auto-deriving an id from
        the folder name is how a returning speaker silently becomes a new
        one, which breaks the split the other way."""
        with pytest.raises(KeyError, match="unknown speaker folder"):
            speaker_id_for("Someone_New_20270101_120000")


class TestTextSentimentTable:
    def test_every_sentence_has_a_valence(self) -> None:
        assert len(TEXT_SENTIMENT) == 34

    def test_values_are_real_sentiments(self) -> None:
        for sentence, value in TEXT_SENTIMENT.items():
            assert isinstance(value, Sentiment), sentence

    def test_all_three_classes_are_used(self) -> None:
        """A table that collapsed to one class would make every clip either
        congruent or incongruent by construction, and PSI meaningless."""
        assert set(TEXT_SENTIMENT.values()) == set(Sentiment)

    def test_obvious_valences_are_not_hedged_to_neutral(self) -> None:
        """Spot-check the anchors. Conservatism about ambiguous wordings is
        deliberate, but it must not swallow sentences whose valence is not
        in doubt -- that would hide real incongruence."""
        assert TEXT_SENTIMENT["You did a great job today."] is Sentiment.POSITIVE
        assert TEXT_SENTIMENT["That's exactly what I wanted."] is Sentiment.POSITIVE
        assert TEXT_SENTIMENT["That is completely unacceptable."] is Sentiment.NEGATIVE
        assert TEXT_SENTIMENT["Nothing ever goes as planned."] is Sentiment.NEGATIVE

    def test_genuinely_two_sided_wordings_stay_neutral(self) -> None:
        """The other half of the rule: these support either reading, so
        picking a side would manufacture incongruence the text does not
        contain and PSI would score our labelling."""
        assert TEXT_SENTIMENT["I can't believe this happened."] is Sentiment.NEUTRAL
        assert TEXT_SENTIMENT["Whatever, it's fine."] is Sentiment.NEUTRAL
        assert TEXT_SENTIMENT["Are you serious?"] is Sentiment.NEUTRAL


class TestBuiltManifest:
    """Runs against the committed manifest when it exists, so a rebuild that
    drifts from the table is caught rather than assumed correct."""

    @staticmethod
    def _manifest() -> pd.DataFrame:
        from scripts.build_zurich import MANIFEST_PATH

        if not MANIFEST_PATH.exists():
            pytest.skip("E6 manifest not built -- run `make data-zurich`")
        return pd.read_csv(MANIFEST_PATH, dtype={"clip_id": str, "speaker_id": str})

    def test_schema_is_valid(self) -> None:
        df = self._manifest()
        df["voice_id"] = df["voice_id"].fillna("")
        df["emotion_tag"] = df["emotion_tag"].fillna("")
        validate_manifest(df)

    def test_splits_are_speaker_disjoint(self) -> None:
        from ssa.splits import assert_speaker_disjoint

        assert_speaker_disjoint(self._manifest())

    def test_is_congruent_matches_the_two_sentiment_columns(self) -> None:
        df = self._manifest()
        expected = df["text_sentiment"] == df["prosody_sentiment"]
        assert (df["is_congruent"] == expected).all()

    def test_text_sentiment_matches_the_table(self) -> None:
        df = self._manifest()
        for row in df.itertuples(index=False):
            assert row.text_sentiment == TEXT_SENTIMENT[row.text].value, row.text

    def test_the_set_is_substantially_incongruent(self) -> None:
        """E6's value to this project is its incongruence. If a relabelling
        ever drops that below half, PSI stops being the headline it is
        reported as."""
        df = self._manifest()
        assert (~df["is_congruent"]).mean() > 0.5
