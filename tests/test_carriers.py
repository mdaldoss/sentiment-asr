"""Tests for ssa/carriers.py -- pure data, plus the regression test for the
`make record` bug (record_prompts must import with no cartesia installed)."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from ssa.carriers import (
    CARRIERS,
    D1_CONGRUENT_TEXT,
    D1_EMOTION_SENTIMENT,
    D1_NEUTRAL_TEXT,
    D1_SPEED_ADJUSTED,
    EMOTIONS,
)
from ssa.mapping import CARTESIA_MAP
from ssa.types import Sentiment


class TestE2Carriers:
    def test_all_three_sentiments_present(self) -> None:
        assert set(CARRIERS) == set(Sentiment)

    def test_five_sentences_per_sentiment(self) -> None:
        for sentences in CARRIERS.values():
            assert len(sentences) == 5


class TestD1EmotionSentiment:
    def test_covers_all_five_emotions(self) -> None:
        assert set(D1_EMOTION_SENTIMENT) == set(EMOTIONS)

    def test_calm_is_scoped_to_neutral(self) -> None:
        """calm is globally ambiguous in ssa.mapping -- this dataset's
        override to neutral is local to D1_EMOTION_SENTIMENT only."""
        assert D1_EMOTION_SENTIMENT["calm"] == Sentiment.NEUTRAL

    def test_other_four_emotions_agree_with_cartesia_map(self) -> None:
        for emotion in EMOTIONS:
            if emotion == "calm":
                continue
            assert D1_EMOTION_SENTIMENT[emotion] == CARTESIA_MAP[emotion]


class TestD1Text:
    def test_neutral_text_has_short_and_long(self) -> None:
        assert set(D1_NEUTRAL_TEXT) == {"short", "long"}
        assert len(D1_NEUTRAL_TEXT["short"]) < len(D1_NEUTRAL_TEXT["long"])

    def test_congruent_text_covers_all_emotions_with_short_and_long(self) -> None:
        assert set(D1_CONGRUENT_TEXT) == set(EMOTIONS)
        for texts in D1_CONGRUENT_TEXT.values():
            assert set(texts) == {"short", "long"}
            assert len(texts["short"]) < len(texts["long"])

    def test_congruent_text_differs_by_emotion(self) -> None:
        """The whole point of the congruent condition -- unlike the shared
        neutral text, each emotion gets its own words."""
        short_texts = {texts["short"] for texts in D1_CONGRUENT_TEXT.values()}
        assert len(short_texts) == len(EMOTIONS)


class TestD1SpeedAdjusted:
    def test_covers_all_five_emotions(self) -> None:
        assert set(D1_SPEED_ADJUSTED) == set(EMOTIONS)

    def test_within_cartesia_documented_bounds(self) -> None:
        for speed in D1_SPEED_ADJUSTED.values():
            assert 0.6 <= speed <= 1.5

    def test_volume_is_never_touched(self) -> None:
        """Not really testable as an assertion on this dict (there is no
        volume key at all) -- this test documents the invariant by asserting
        the intent explicitly: no key named volume exists anywhere here."""
        assert not any("volume" in str(k).lower() for k in D1_SPEED_ADJUSTED)


class TestRecordPromptsImportsWithoutCartesia:
    def test_import_succeeds_when_cartesia_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression test for the `make record` bug: record_prompts.py must
        never transitively require the `cartesia` package. Simulates
        cartesia being uninstalled (the `generate` extra not being present)
        by making its import fail, then re-imports record_prompts fresh.

        Uses monkeypatch.delitem (not a bare sys.modules.pop) so the real
        modules are restored once this test tears down -- a bare pop would
        leave later tests importing a second, distinct `ssa.tts` module
        object, which breaks unittest.mock.patch("ssa.tts.Cartesia") there
        (patches the new object; get_client's globals still reference the
        old one)."""
        for mod_name in ("scripts.record_prompts", "scripts.gen_synthetic", "cartesia", "ssa.tts"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "cartesia" or name.startswith("cartesia."):
                raise ModuleNotFoundError("No module named 'cartesia'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        module = importlib.import_module("scripts.record_prompts")
        assert hasattr(module, "build_prompts")
